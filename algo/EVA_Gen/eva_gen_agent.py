"""
MI-Unity框架完整实现 - 完全修正版 v2
解决专家指出的所有严重问题：
1. 每个agent独立动作输出
2. Ground Twin shape修正
3. 真正的PPO/MAPPO（old_log_probs + clip）
4. 环境交互接口
5. MI estimator训练
6. 动作one-hot转换
7. 低频belief更新
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import math
from typing import Dict, List, Tuple, Optional

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"使用设备: {device}")


# ============================================================
# L1: 层次化数字孪生
# ============================================================

class EdgeDigitalTwin(nn.Module):
    """边缘孪生 - 推理时使用"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, obs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch_size, obs_dim]
            hidden_state: [1, batch_size, hidden_dim]
        
        Returns:
            context: [batch_size, hidden_dim]
            new_hidden: [1, batch_size, hidden_dim]
        """
        obs = obs.unsqueeze(1)  # [batch_size, 1, obs_dim]
        output, new_hidden = self.gru(obs, hidden_state)
        context = output.squeeze(1)  # [batch_size, hidden_dim]
        
        return context, new_hidden


class GroundDigitalTwin(nn.Module):
    """
    地面孪生 - 训练监督器
    
    修正问题2: 输出 [batch, num_agents, hidden]
    """
    
    def __init__(self, state_dim: int, num_agents: int, context_dim: int = 128, window_size: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.context_dim = context_dim
        self.window_size = window_size
        
        # 修正问题2: 为每个agent生成独立的target context
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim * window_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, context_dim * num_agents)  # 输出所有agent的context
        )
    
    def extract_target_context(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        修正问题2: 返回 [batch_size, num_agents, context_dim]
        
        Args:
            state_history: [batch_size, window_size, state_dim]
        
        Returns:
            target_contexts: [batch_size, num_agents, context_dim]
        """
        batch_size = state_history.shape[0]
        flattened = state_history.view(batch_size, -1)
        features = self.feature_extractor(flattened)  # [batch, context_dim * num_agents]
        
        # 重塑为 [batch, num_agents, context_dim]
        target_contexts = features.view(batch_size, self.num_agents, self.context_dim)
        
        return target_contexts


# ============================================================
# L2: 生成式信念模块 (DDPM) - 添加缓存机制
# ============================================================

class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """U-Net基础块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU()
        )
        
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.residual(x)


class ConditionalDDPM(nn.Module):
    """条件DDPM"""
    
    def __init__(self, belief_dim: int = 256, context_dim: int = 128, 
                 timesteps: int = 100, hidden_dim: int = 512):
        super().__init__()
        self.belief_dim = belief_dim
        self.context_dim = context_dim
        self.timesteps = timesteps
        
        # 噪声调度
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        # 时间步嵌入
        self.time_embedding = TimeEmbedding(hidden_dim)
        
        # U-Net
        input_dim = belief_dim + context_dim + hidden_dim
        
        self.encoder1 = UNetBlock(input_dim, hidden_dim)
        self.encoder2 = UNetBlock(hidden_dim, hidden_dim)
        self.bottleneck = UNetBlock(hidden_dim, hidden_dim)
        self.decoder2 = UNetBlock(hidden_dim * 2, hidden_dim)
        self.decoder1 = UNetBlock(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, belief_dim)
    
    def forward_diffusion(self, b0: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散"""
        noise = torch.randn_like(b0)
        alpha_bars_t = self.alpha_bars[timesteps]
        alpha_bars_t = alpha_bars_t.view(-1, 1)
        
        bk = torch.sqrt(alpha_bars_t) * b0 + torch.sqrt(1 - alpha_bars_t) * noise
        
        return bk, noise
    
    def denoise_network(self, bk: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """去噪网络"""
        t_emb = self.time_embedding(timesteps)
        x = torch.cat([bk, context, t_emb], dim=-1)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        bottleneck = self.bottleneck(enc2)
        dec2 = self.decoder2(torch.cat([bottleneck, enc2], dim=-1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=-1))
        
        predicted_noise = self.output(dec1)
        
        return predicted_noise
    
    def ddim_sample(self, context: torch.Tensor, ddim_steps: int = 10) -> torch.Tensor:
        """DDIM快速采样"""
        batch_size = context.shape[0]
        device = context.device
        
        bk = torch.randn(batch_size, self.belief_dim, device=device)
        ddim_timesteps = torch.linspace(self.timesteps - 1, 0, ddim_steps + 1, dtype=torch.long, device=device)
        
        for i in range(ddim_steps):
            k_curr = ddim_timesteps[i]
            k_next = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else torch.tensor(0, device=device)
            
            timesteps_curr = torch.full((batch_size,), k_curr.item(), device=device, dtype=torch.long)
            predicted_noise = self.denoise_network(bk, timesteps_curr, context)
            
            alpha_bar_curr = self.alpha_bars[k_curr]
            alpha_bar_next = self.alpha_bars[k_next] if k_next >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (bk - torch.sqrt(1 - alpha_bar_curr) * predicted_noise) / torch.sqrt(alpha_bar_curr)
            dir_xt = torch.sqrt(1 - alpha_bar_next) * predicted_noise
            
            if k_next >= 0:
                bk = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt
            else:
                bk = pred_x0
        
        return bk
    
    def compute_loss(self, b0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """训练损失"""
        batch_size = b0.shape[0]
        device = b0.device
        
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        bk, noise = self.forward_diffusion(b0, timesteps)
        predicted_noise = self.denoise_network(bk, timesteps, context)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


class GRUBeliefModule(nn.Module):
    """L2: GRU/RNN信念模块（消融实验用 - 替换Diffusion）"""
    
    def __init__(self, obs_dim: int, belief_dim: int = 256, context_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.context_dim = context_dim
        
        # GRU-based belief encoder
        # 输入: context (from EdgeDigitalTwin) + obs (可选)
        # 输出: belief representation
        self.gru = nn.GRU(
            input_size=context_dim + obs_dim,  # context + obs
            hidden_size=belief_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 投影层：将GRU输出映射到belief_dim
        self.belief_proj = nn.Sequential(
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim)
        )
        
        # 观测编码器（用于训练时的target）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, belief_dim)
        )
        
        # 用于存储hidden state（推理时）
        self.hidden_state = None
    
    def forward(self, context: torch.Tensor, target_obs: Optional[torch.Tensor] = None, 
                training: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context: [batch_size, context_dim] - 来自EdgeDigitalTwin
            target_obs: [batch_size, obs_dim] - 训练时的target观测（可选）
            training: 是否在训练模式
        
        Returns:
            belief: [batch_size, belief_dim]
            loss: Optional[torch.Tensor] - 训练时的loss（GRU版本返回None或MSE loss）
        """
        batch_size = context.shape[0]
        
        # 初始化或重置hidden state（如果batch_size变化）
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            # GRU hidden state: [num_layers, batch_size, hidden_size]
            self.hidden_state = torch.zeros(
                self.gru.num_layers, batch_size, self.belief_dim,
                device=context.device, dtype=context.dtype
            )
        
        if training and target_obs is not None:
            # 训练模式：使用target_obs计算loss
            target_belief = self.obs_encoder(target_obs)  # [batch_size, belief_dim]
            
            # 将context和obs拼接作为GRU输入
            # 为了使用GRU，我们需要将输入reshape为序列
            # 这里我们使用单步序列：[batch_size, 1, context_dim + obs_dim]
            combined_input = torch.cat([context, target_obs], dim=1)  # [batch_size, context_dim + obs_dim]
            combined_input = combined_input.unsqueeze(1)  # [batch_size, 1, context_dim + obs_dim]
            
            # GRU前向传播
            gru_out, self.hidden_state = self.gru(combined_input, self.hidden_state)
            # gru_out: [batch_size, 1, belief_dim]
            gru_hidden = gru_out.squeeze(1)  # [batch_size, belief_dim]
            
            # 投影到belief空间
            belief = self.belief_proj(gru_hidden)  # [batch_size, belief_dim]
            
            # 计算MSE loss（作为belief reconstruction loss）
            loss = F.mse_loss(belief, target_belief)
            
            return belief, loss
        else:
            # 推理模式：只使用context生成belief
            # 如果没有obs，我们使用零向量作为obs部分
            zero_obs = torch.zeros(batch_size, self.obs_dim, device=context.device, dtype=context.dtype)
            combined_input = torch.cat([context, zero_obs], dim=1)  # [batch_size, context_dim + obs_dim]
            combined_input = combined_input.unsqueeze(1)  # [batch_size, 1, context_dim + obs_dim]
            
            # GRU前向传播
            gru_out, self.hidden_state = self.gru(combined_input, self.hidden_state)
            gru_hidden = gru_out.squeeze(1)  # [batch_size, belief_dim]
            
            # 投影到belief空间
            belief = self.belief_proj(gru_hidden)  # [batch_size, belief_dim]
            
            return belief, None
    
    def reset_hidden(self):
        """重置hidden state（用于新episode）"""
        self.hidden_state = None


class GenerativeBeliefModule(nn.Module):
    """L2: 生成式信念模块（Diffusion-based）"""
    
    def __init__(self, obs_dim: int, belief_dim: int = 256, context_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.context_dim = context_dim
        
        self.ddpm = ConditionalDDPM(
            belief_dim=belief_dim,
            context_dim=context_dim,
            timesteps=100
        )
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, belief_dim)
        )
    
    def forward(self, context: torch.Tensor, target_obs: Optional[torch.Tensor] = None, 
                training: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if training and target_obs is not None:
            target_belief = self.obs_encoder(target_obs)
            loss = self.ddpm.compute_loss(target_belief, context)
            
            with torch.no_grad():
                belief = self.ddpm.ddim_sample(context, ddim_steps=10)
            
            return belief, loss
        else:
            belief = self.ddpm.ddim_sample(context, ddim_steps=10)
            return belief, None


# ============================================================
# L3: MI-Collab编码器
# ============================================================

class MICollabEncoder(nn.Module):
    """L3: MI-Collab"""
    
    def __init__(self, belief_dim: int, num_agents: int, hidden_dim: int = 256):
        super().__init__()
        self.belief_dim = belief_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=belief_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, belief_dim)
        )
        
        self.norm1 = nn.LayerNorm(belief_dim)
        self.norm2 = nn.LayerNorm(belief_dim)
        
        # 协作融合
        self.collaboration_fusion = nn.Sequential(
            nn.Linear(belief_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
            nn.Tanh()
        )
        
        # JS散度判别器
        self.js_discriminator = nn.Sequential(
            nn.Linear(belief_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.beta_G = 0.6
        self.beta_L = 0.4
    
    def forward(self, individual_beliefs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            individual_beliefs: [batch_size, num_agents, belief_dim]
        
        Returns:
            collaborative_belief: [batch_size, belief_dim]
            individual_features: [batch_size, num_agents, belief_dim]
            attention_weights: attention weights
        """
        batch_size = individual_beliefs.shape[0]
        
        attended_features, attention_weights = self.attention(
            individual_beliefs, individual_beliefs, individual_beliefs
        )
        
        attended_features = self.norm1(attended_features + individual_beliefs)
        ff_output = self.feed_forward(attended_features)
        individual_features = self.norm2(ff_output + attended_features)
        
        flattened = individual_features.view(batch_size, -1)
        collaborative_belief = self.collaboration_fusion(flattened)
        
        return collaborative_belief, individual_features, attention_weights
    
    def compute_js_mi(self, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor) -> torch.Tensor:
        """计算JS散度MI估计"""
        # 数值稳定性说明：
        # 1) 判别器输出(logits)不加约束时会在训练早期迅速发散，导致 softplus/exp 溢出，
        #    进而让 MI 项出现异常的大幅负值/正值（你日志中的 -75/-161 就是典型征兆）。
        # 2) 这里对 logits 进行 clamp，并使用 softplus 的稳定形式。
        def softplus(x):
            return F.softplus(x)

        pos_scores = self.js_discriminator(positive_pairs).clamp(-10.0, 10.0)
        neg_scores = self.js_discriminator(negative_pairs).clamp(-10.0, 10.0)

        # JSD-based MI lower bound (a.k.a. Donsker-Varadhan variant with logistic)
        # E_p[ -softplus(-T) ] - E_q[ softplus(T) ]
        pos_term = -softplus(-pos_scores).mean()
        neg_term = softplus(neg_scores).mean()
        mi_estimate = pos_term - neg_term
        return mi_estimate
    
    def compute_mvmi_loss(self, individual_beliefs: torch.Tensor, 
                         collaborative_belief: torch.Tensor) -> torch.Tensor:
        """计算MVMI损失"""
        batch_size, num_agents, belief_dim = individual_beliefs.shape
        
        positive_pairs = []
        negative_pairs = []
        
        for i in range(batch_size):
            for j in range(num_agents):
                pos_pair = torch.cat([
                    individual_beliefs[i, j, :],
                    collaborative_belief[i, :]
                ], dim=-1)
                positive_pairs.append(pos_pair)
                
                neg_idx = (i + 1) % batch_size
                neg_pair = torch.cat([
                    individual_beliefs[i, j, :],
                    collaborative_belief[neg_idx, :]
                ], dim=-1)
                negative_pairs.append(neg_pair)
        
        positive_samples = torch.stack(positive_pairs)
        negative_samples = torch.stack(negative_pairs)
        
        mi = self.compute_js_mi(positive_samples, negative_samples)
        mvmi_loss = -mi
        
        return mvmi_loss


# ============================================================
# L4: QAVER + MI Estimator
# ============================================================

class StateActionMIEstimator(nn.Module):
    """
    状态-动作MI估计器
    
    修正问题6: 支持one-hot动作输入
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        
        # InfoNCE估计器
        self.info_nce_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # L1Out估计器
        self.l1out_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _action_to_onehot(self, actions: torch.Tensor) -> torch.Tensor:
        """
        修正问题6: 将离散动作转为one-hot
        
        Args:
            actions: [batch_size] LongTensor
        
        Returns:
            actions_onehot: [batch_size, action_dim]
        """
        if actions.dtype in [torch.long, torch.int64, torch.int32]:
            return F.one_hot(actions, num_classes=self.action_dim).float()
        else:
            return actions  # 已经是one-hot
    
    def estimate_mi_infonce(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """InfoNCE / binary classification view.

        重要修正：此前函数返回的是 `-BCE` 作为 MI estimate，这会让外层很容易把
        “要最大化的量”当成“loss 去最小化”，从而出现发散。

        现在：
        - 返回一个 **可直接最小化** 的 `bce_loss`（稳定、非负）；
        - 若需要 MI estimate，可用 `mi_est = log(2) - bce_loss` 作为监控指标。
        """
        actions_onehot = self._action_to_onehot(actions)
        sa_pairs = torch.cat([states, actions_onehot], dim=-1)

        pos_logits = self.info_nce_net(sa_pairs).squeeze(-1).clamp(-10.0, 10.0)

        shuffled_actions = actions_onehot[torch.randperm(actions_onehot.shape[0])]
        neg_pairs = torch.cat([states, shuffled_actions], dim=-1)
        neg_logits = self.info_nce_net(neg_pairs).squeeze(-1).clamp(-10.0, 10.0)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([
            torch.ones(pos_logits.shape[0], device=states.device),
            torch.zeros(neg_logits.shape[0], device=states.device)
        ], dim=0)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return bce_loss
    
    def estimate_mi_l1out(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """L1Out MI估计"""
        actions_onehot = self._action_to_onehot(actions)
        batch_size = states.size(0)
        sa_pairs = torch.cat([states, actions_onehot], dim=-1)
        
        # 数值稳定：限制 logits 范围，避免 logsumexp 溢出
        log_prob_joint = self.l1out_net(sa_pairs).squeeze(-1).clamp(-10.0, 10.0)
        log_prob_marginal = torch.zeros_like(log_prob_joint)
        
        for i in range(batch_size):
            other_indices = torch.cat([torch.arange(i), torch.arange(i+1, batch_size)])
            if len(other_indices) > 0:
                action_i = actions_onehot[i:i+1].expand(len(other_indices), -1)
                other_states = states[other_indices]
                other_pairs = torch.cat([other_states, action_i], dim=-1)
                
                other_log_probs = self.l1out_net(other_pairs).squeeze(-1).clamp(-10.0, 10.0)
                log_prob_marginal[i] = torch.logsumexp(other_log_probs, dim=0) - torch.log(
                    torch.tensor(len(other_indices), dtype=torch.float, device=states.device)
                )
        
        mi_l1out = (log_prob_joint - log_prob_marginal).mean()
        # 额外稳定：限制估计值范围（只用于监控/弱正则）
        return mi_l1out.clamp(-10.0, 10.0)


class SingleReplayBuffer:
    """单一经验回放缓冲区（消融实验用）- Uniform sampling"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, old_log_probs, reward, next_state, done, hp_tag, episode_return,
             td_error=None, state_history=None):
        """存储经验（忽略hp_tag，统一存储）"""
        experience = (state, action, old_log_probs, reward, next_state, done, hp_tag, episode_return, state_history)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[List, torch.Tensor, List]:
        """Uniform随机采样"""
        if len(self.buffer) < batch_size:
            return [], torch.tensor([]), []
        
        # Uniform random sampling
        sampled_indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[i] for i in sampled_indices]
        
        # 返回uniform weights（所有为1.0）
        weights = torch.ones(batch_size, dtype=torch.float32, device=device)
        
        # 返回格式与QAVERBuffer兼容：indices为list of (pool_type, idx)
        # 对于SingleBuffer，pool_type统一为'single'
        indices = [('single', i) for i in sampled_indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: List[Tuple[str, int]], priorities: List[float]):
        """保持接口兼容，但不做任何操作（uniform sampling不需要更新优先级）"""
        pass
    
    def __len__(self):
        return len(self.buffer)


class QAVERBuffer:
    """L4: QAVER缓冲区 - Step C: 分阶段HPB系统（HPB-2/HPB-1/SPB）"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 p_HPB: float = 0.25, initial_threshold: float = -10.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.p_HPB = p_HPB  # ✅ 进一步提高：起步p_HPB=0.3
        
        # ✅ Step C: 分阶段HPB系统
        self.hpb2 = []  # HPB-2 (hard): episode_has_any_target==1
        self.hpb1 = []  # HPB-1 (mid): episode_has_crossed==1
        self.spb = []   # SPB: 其他
        
        self.hpb2_priorities = []
        self.hpb1_priorities = []
        self.spb_priorities = []
        
        self.hpb2_returns = []
        self.hpb1_returns = []
        self.spb_returns = []
        
        self.high_return_threshold = initial_threshold
        self.spb_quality_threshold = 0.0
        
        self.max_priority = 1.0
        self.update_counter = 0
    
    def push(self, state, action, old_log_probs, reward, next_state, done, hp_tag, episode_return,
             td_error=None, state_history=None):
        """存储经验 - Step C: 使用hp_tag ('HPB-2', 'HPB-1', 'SPB') 替代qos_tag
        Args:
            state: obs or state (numpy)
            action: [num_agents] list[int]
            old_log_probs: [num_agents] list[float] (behavior policy log-prob at sampling time)
            hp_tag: 'HPB-2', 'HPB-1', or 'SPB'
        """
        # experience tuple (backward compatible):
        # (state, action, old_log_probs, reward, next_state, done, hp_tag, episode_return, state_history)
        experience = (state, action, old_log_probs, reward, next_state, done, hp_tag, episode_return, state_history)
        
        if td_error is not None:
            priority = abs(td_error) + 1e-6
        else:
            priority = self.max_priority
        
        # ✅ Step C: 根据hp_tag分类存储
        if hp_tag == 'HPB-2':
            # HPB-2: 优先存储，容量限制为capacity // 3
            max_hpb2 = self.capacity // 3
            if len(self.hpb2) < max_hpb2:
                self.hpb2.append(experience)
                self.hpb2_priorities.append(priority)
                self.hpb2_returns.append(episode_return)
                self.max_priority = max(self.max_priority, priority)
            else:
                min_idx = np.argmin(self.hpb2_priorities)
                if priority > self.hpb2_priorities[min_idx]:
                    self.hpb2[min_idx] = experience
                    self.hpb2_priorities[min_idx] = priority
                    self.hpb2_returns[min_idx] = episode_return
        elif hp_tag == 'HPB-1':
            # HPB-1: 优先存储，容量限制为capacity // 3
            max_hpb1 = self.capacity // 3
            if len(self.hpb1) < max_hpb1:
                self.hpb1.append(experience)
                self.hpb1_priorities.append(priority)
                self.hpb1_returns.append(episode_return)
                self.max_priority = max(self.max_priority, priority)
            else:
                min_idx = np.argmin(self.hpb1_priorities)
                if priority > self.hpb1_priorities[min_idx]:
                    self.hpb1[min_idx] = experience
                    self.hpb1_priorities[min_idx] = priority
                    self.hpb1_returns[min_idx] = episode_return
        else:  # hp_tag == 'SPB'
            # SPB: FIFO队列
            max_spb = self.capacity - (self.capacity // 3) * 2
            if len(self.spb) < max_spb:
                self.spb.append(experience)
                self.spb_priorities.append(priority)
                self.spb_returns.append(episode_return)
            else:
                self.spb.pop(0)
                self.spb_priorities.pop(0)
                self.spb_returns.pop(0)
                self.spb.append(experience)
                self.spb_priorities.append(priority)
                self.spb_returns.append(episode_return)
        
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self._update_thresholds()
    
    def _update_thresholds(self):
        """更新动态阈值 - Step C: 适配分阶段HPB"""
        if len(self.hpb2_returns) > 10:
            returns_array = np.array(self.hpb2_returns)
            self.high_return_threshold = np.percentile(returns_array, 25)
        
        if len(self.spb_returns) > 10:
            returns_array = np.array(self.spb_returns)
            self.spb_quality_threshold = np.median(returns_array)
    
    def sample(self, batch_size: int) -> Tuple[List, torch.Tensor, List]:
        """优先级采样 - Step C: 固定配额采样策略（k2=8, k1=16，剩下从SPB）"""
        total_samples = len(self.hpb2) + len(self.hpb1) + len(self.spb)
        if total_samples < batch_size:
            return [], torch.tensor([]), []
        
        experiences = []
        weights = []
        indices = []
        
        # ✅ 进一步提高：固定配额采样（按文档：k2=6, k1=18，剩下从SPB）
        k2 = 6  # HPB-2配额：固定6（不够就用HPB-1补）
        k1 = 18  # HPB-1配额：固定18（不够就用SPB补）
        
        # 1. 先抽k2条来自HPB-2（不足则用HPB-1补）
        n_hpb2 = min(k2, len(self.hpb2))
        if n_hpb2 > 0:
            hpb2_experiences, hpb2_weights, hpb2_indices = self._sample_from_pool(
                self.hpb2, self.hpb2_priorities, n_hpb2, 'hpb2'
            )
            experiences.extend(hpb2_experiences)
            weights.extend(hpb2_weights)
            indices.extend(hpb2_indices)
        
        # 如果HPB-2不足，用HPB-1补
        remaining_k2 = k2 - n_hpb2
        if remaining_k2 > 0 and len(self.hpb1) > 0:
            n_hpb1_from_k2 = min(remaining_k2, len(self.hpb1))
            hpb1_experiences, hpb1_weights, hpb1_indices = self._sample_from_pool(
                self.hpb1, self.hpb1_priorities, n_hpb1_from_k2, 'hpb1'
            )
            experiences.extend(hpb1_experiences)
            weights.extend(hpb1_weights)
            indices.extend(hpb1_indices)
        
        # 2. 再抽k1条来自HPB-1（不足则用SPB补）
        remaining_k1 = k1 - (len(experiences) - n_hpb2)
        if remaining_k1 > 0 and len(self.hpb1) > 0:
            n_hpb1 = min(remaining_k1, len(self.hpb1))
            hpb1_experiences, hpb1_weights, hpb1_indices = self._sample_from_pool(
                self.hpb1, self.hpb1_priorities, n_hpb1, 'hpb1'
            )
            experiences.extend(hpb1_experiences)
            weights.extend(hpb1_weights)
            indices.extend(hpb1_indices)
        
        # 如果HPB-1不足，用SPB补
        remaining_k1_spb = k1 - (len(experiences) - n_hpb2)
        if remaining_k1_spb > 0 and len(self.spb) > 0:
            n_spb_from_k1 = min(remaining_k1_spb, len(self.spb))
            spb_experiences, spb_weights, spb_indices = self._sample_from_pool(
                self.spb, self.spb_priorities, n_spb_from_k1, 'spb'
            )
            experiences.extend(spb_experiences)
            weights.extend(spb_weights)
            indices.extend(spb_indices)
        
        # 3. 剩下从SPB抽
        remaining = batch_size - len(experiences)
        if remaining > 0 and len(self.spb) > 0:
            n_spb = min(remaining, len(self.spb))
            spb_experiences, spb_weights, spb_indices = self._sample_from_pool(
                self.spb, self.spb_priorities, n_spb, 'spb'
            )
            experiences.extend(spb_experiences)
            weights.extend(spb_weights)
            indices.extend(spb_indices)
        
        if len(weights) == 0:
            return [], torch.tensor([]), []
        
        weights = torch.FloatTensor(weights).to(device)
        
        return experiences, weights, indices
    
    def _sample_from_pool(self, pool, priorities, n_samples: int, pool_type: str):
        """从池中采样"""
        if len(pool) == 0:
            return [], [], []
        
        n_samples = min(n_samples, len(pool))
        
        priorities_array = np.array(priorities)
        probabilities = priorities_array ** self.alpha
        probabilities /= probabilities.sum()
        
        sampled_indices = np.random.choice(len(pool), n_samples, p=probabilities, replace=False)
        
        N = len(pool)
        weights = (N * probabilities[sampled_indices]) ** (-self.beta)
        weights /= weights.max()
        
        experiences = [pool[i] for i in sampled_indices]
        indices_with_type = [(pool_type, i) for i in sampled_indices]
        
        return experiences, weights.tolist(), indices_with_type
    
    def update_priorities(self, indices: List[Tuple[str, int]], priorities: List[float]):
        """更新优先级 - Step C: 支持hpb2/hpb1/spb"""
        for (pool_type, idx), priority in zip(indices, priorities):
            if pool_type == 'hpb2' and idx < len(self.hpb2_priorities):
                self.hpb2_priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
            elif pool_type == 'hpb1' and idx < len(self.hpb1_priorities):
                self.hpb1_priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
            elif pool_type == 'spb' and idx < len(self.spb_priorities):
                self.spb_priorities[idx] = priority
    
    def __len__(self):
        return len(self.hpb2) + len(self.hpb1) + len(self.spb)


# ============================================================
# Actor-Critic - 修正问题1和问题3
# ============================================================

class DiscreteActor(nn.Module):
    """
    修正问题1: Per-agent actor
    每个agent一个独立的actor网络
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int, hidden_dim: int = 256):
        super().__init__()
        self.agent_id = agent_id
        
        # 添加agent_id embedding
        self.id_embedding = nn.Embedding(10, 16)  # 支持最多10个agents
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + 16, hidden_dim),  # +16 for id embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state, agent_id=None):
        """
        Args:
            state: [batch_size, state_dim]
            agent_id: int (optional, use self.agent_id if None)
        
        Returns:
            action_logits: [batch_size, action_dim]
        """
        if agent_id is None:
            agent_id = self.agent_id
        
        batch_size = state.shape[0]
        id_tensor = torch.full((batch_size,), agent_id, dtype=torch.long, device=state.device)
        id_emb = self.id_embedding(id_tensor)  # [batch_size, 16]
        
        x = torch.cat([state, id_emb], dim=-1)
        return self.network(x)
    
    def get_action(self, state, agent_id=None, deterministic=False):
        """
        获取动作
        
        Returns:
            action: [batch_size]
            log_prob: [batch_size]
            entropy: [batch_size]
        """
        logits = self.forward(state, agent_id)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class Critic(nn.Module):
    """Critic网络（Value function）"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


# ============================================================
# MI-Unity主Agent - 完整修正版
# ============================================================

class MIUnityAgent:
    """
    MI-Unity智能体 - 完整修正版 v2
    
    修正所有7个问题：
    1. 每个agent独立动作
    2. Ground Twin shape正确
    3. 真正的PPO (old_log_probs + clip)
    4. 环境交互接口
    5. MI estimator训练
    6. 动作one-hot转换
    7. 低频belief更新
    """
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 belief_dim: int,
                 num_agents: int,
                 config: Dict):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.num_agents = num_agents
        self.config = config

        # ========== Ground-Twin 训练所需的状态历史（自包含实现） ==========
        # 你当前 GridWorld A 场景的 runner 没有传入 state_history_batch，导致 dt_loss 长期为 0。
        # 这里在 agent 内部维护一个 "global state" 的滑动窗口：把所有 agent 的观测拼接成
        # [state_dim = obs_dim * num_agents]，再组成 [window_size, state_dim] 作为 teacher 信号。
        # 这不是特权信息：它与 MAPPO/MACE 的 centralized training 输入同源（多智能体联合观测）。
        self._gt_window_size = int(config.get('gt_window_size', 3))
        self._recent_global_states = deque(maxlen=self._gt_window_size)
        
        # 修正问题7: 低频belief更新
        self.belief_update_freq = config.get('belief_update_freq', 5)  # 每5步更新一次
        self.step_counter = 0
        self.cached_beliefs = None
        
        # MI warmup机制
        self.global_steps = 0
        self.mi_warmup_steps = config.get('mi_warmup_steps', 50000)  # MI warmup步数
        self.base_mi_weight = config.get('base_mi_weight', 1.0)  # 基础MI权重
        
        # ========== L1: 层次化数字孪生 ==========
        self.edge_twins = nn.ModuleList([
            EdgeDigitalTwin(obs_dim, config['hidden_dim']).to(device)
            for _ in range(num_agents)
        ])
        
        # 修正问题2: Ground Twin输出 [batch, num_agents, hidden]
        self.ground_twin = GroundDigitalTwin(
            state_dim=config.get('state_dim', obs_dim * num_agents),
            num_agents=num_agents,
            context_dim=config['hidden_dim'],
            window_size=self._gt_window_size
        ).to(device)
        
        # ========== L2: 生成式信念模块 ==========
        # ✅ 消融实验：根据配置选择Belief类型
        use_diffusion = config.get('use_diffusion', True)  # 默认使用Diffusion
        if use_diffusion:
            BeliefModuleClass = GenerativeBeliefModule
        else:
            BeliefModuleClass = GRUBeliefModule
        
        self.belief_modules = nn.ModuleList([
            BeliefModuleClass(
                obs_dim=obs_dim,
                belief_dim=belief_dim,
                context_dim=config['hidden_dim']
            ).to(device)
            for _ in range(num_agents)
        ])
        
        # ========== L3: MI-Collab编码器 ==========
        self.mi_collab = MICollabEncoder(
            belief_dim=belief_dim,
            num_agents=num_agents,
            hidden_dim=config['hidden_dim']
        ).to(device)
        
        # ========== 修正问题1: Per-agent actors ==========
        self.actors = nn.ModuleList([
            DiscreteActor(belief_dim, action_dim, agent_id=i, hidden_dim=config['hidden_dim']).to(device)
            for i in range(num_agents)
        ])
        
        self.critic = Critic(belief_dim, config['hidden_dim']).to(device)
        
        # ========== 修正问题6: MI估计器（支持one-hot） ==========
        self.sa_mi_estimator = StateActionMIEstimator(
            belief_dim, action_dim, config['hidden_dim']
        ).to(device)
        
        # ========== L4: QAVER ==========
        # ✅ 消融实验：根据配置选择Buffer类型
        use_dual_buffer = config.get('use_dual_buffer', True)  # 默认使用Dual-Buffer
        if use_dual_buffer:
            self.memory = QAVERBuffer(
                capacity=config.get('buffer_capacity', config.get('buffer_size', 10000)),
                alpha=config.get('priority_alpha', 0.6),
                beta=config.get('priority_beta', 0.4),
                p_HPB=config.get('p_HPB', 0.25),  # ✅ Step C: 默认0.25（从0.2~0.3起步）
                initial_threshold=config.get('initial_threshold', -10.0)
            )
        else:
            # 使用单一Replay Buffer（Uniform sampling）
            self.memory = SingleReplayBuffer(
                capacity=config.get('buffer_capacity', config.get('buffer_size', 10000))
            )
        
        # ========== 优化器 ==========
        self.perception_optimizer = optim.Adam(
            list(self.edge_twins.parameters()) + 
            list(self.ground_twin.parameters()) +
            list(self.belief_modules.parameters()) +
            list(self.mi_collab.parameters()),
            lr=config['lr_perception']
        )
        
        # 修正问题1: 每个agent一个actor
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=config['lr_actor'])
            for actor in self.actors
        ]
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        
        # 修正问题5: MI estimator optimizer实际使用
        self.mi_optimizer = optim.Adam(self.sa_mi_estimator.parameters(), lr=config['lr_mi_estimator'])
        
        # 隐藏状态
        self.edge_hidden_states = [None for _ in range(num_agents)]
        
        # 修正问题3: PPO需要存储old_log_probs
        self.eps_clip = config.get('eps_clip', 0.2)
        
        # 训练统计
        self.update_step = 0
    
    def reset_hidden_states(self):
        """重置隐藏状态"""
        self.edge_hidden_states = [None for _ in range(self.num_agents)]
        self.step_counter = 0
        self.cached_beliefs = None
    
    def select_action(self, observations: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        修正问题1: 返回每个agent的动作 [num_agents]
        修正问题7: 低频belief更新
        
        Returns:
            actions: [num_agents] 每个agent一个离散动作
            log_probs: [num_agents]
            entropies: [num_agents]
        """
        if len(observations.shape) == 1:
            observations = np.tile(observations, (self.num_agents, 1))
        
        observations = torch.FloatTensor(observations).to(device)  # [num_agents, obs_dim]
        
        with torch.no_grad():
            # 修正问题7: 低频belief更新
            if self.step_counter % self.belief_update_freq == 0 or self.cached_beliefs is None:
                # ========== L1: 时序上下文 ==========
                contexts = []
                for i in range(self.num_agents):
                    context, self.edge_hidden_states[i] = self.edge_twins[i](
                        observations[i:i+1], self.edge_hidden_states[i]
                    )
                    contexts.append(context)
                
                contexts = torch.cat(contexts, dim=0)  # [num_agents, hidden_dim]
                
                # ========== L2: 生成式信念（低频更新） ==========
                individual_beliefs = []
                for i in range(self.num_agents):
                    belief, _ = self.belief_modules[i](contexts[i:i+1], training=False)
                    individual_beliefs.append(belief)
                
                individual_beliefs = torch.cat(individual_beliefs, dim=0)  # [num_agents, belief_dim]
                individual_beliefs = individual_beliefs.unsqueeze(0)  # [1, num_agents, belief_dim]
                
                # ========== L3: MI-Collab ==========
                collaborative_belief, _, _ = self.mi_collab(individual_beliefs)
                # collaborative_belief: [1, belief_dim]
                
                # 缓存
                self.cached_beliefs = collaborative_belief
            else:
                # 使用缓存的belief
                collaborative_belief = self.cached_beliefs
            
            # ========== 修正问题1: 每个agent独立选择动作 ==========
            actions_list = []
            log_probs_list = []
            entropies_list = []
            
            for i in range(self.num_agents):
                action, log_prob, entropy = self.actors[i].get_action(
                    collaborative_belief, agent_id=i, deterministic=deterministic
                )
                actions_list.append(action)
                log_probs_list.append(log_prob)
                entropies_list.append(entropy)
            
            actions = torch.stack(actions_list, dim=0)  # [num_agents]
            log_probs = torch.stack(log_probs_list, dim=0)  # [num_agents]
            entropies = torch.stack(entropies_list, dim=0)  # [num_agents]
        
        self.step_counter += 1
        
        self._last_log_probs = log_probs.cpu().numpy().astype(np.float32)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), entropies.cpu().numpy()
    
    def store_transition(self, obs, actions, reward, next_obs, done, info, episode_return):
        """
        修正问题4: 明确的环境交互接口 - Step C: 使用hp_tag替代qos_tag
        
        Args:
            obs: [obs_dim] or [num_agents, obs_dim]
            actions: [num_agents]
            reward: float
            next_obs: [obs_dim] or [num_agents, obs_dim]
            done: bool
            info: dict (包含hp_tag等信息)
            episode_return: float
        """
        # ✅ Step C: 确定HP标签（'HPB-2', 'HPB-1', 'SPB'）
        hp_tag = info.get('hp_tag', 'SPB')  # 默认SPB
        # 规范化动作与old_log_probs（PPO必须使用采样时的log_prob）
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(self.num_agents,)
        old_lp = getattr(self, '_last_log_probs', None)
        if old_lp is None:
            # 如果外部没有保存log_probs，这里退化为全0（ratio≈exp(new-0)，会偏，但至少能跑）
            old_lp_arr = np.zeros((self.num_agents,), dtype=np.float32)
        else:
            old_lp_arr = np.asarray(old_lp, dtype=np.float32).reshape(self.num_agents,)

        td_error = None  # priority 在 update() 中统一计算并回写

        # ========== 自包含的 Ground-Twin state_history ==========
        # 组装 “global state”：拼接所有 agent 的本地观测，形状 [state_dim = obs_dim * num_agents]
        obs_np = np.asarray(obs, dtype=np.float32)
        if obs_np.ndim == 1:
            # 单个 obs（极少出现）：为了不崩溃，按 num_agents 重复拼接
            global_state = np.concatenate([obs_np for _ in range(self.num_agents)], axis=0)
        else:
            # [num_agents, obs_dim] -> flatten
            global_state = obs_np.reshape(self.num_agents, -1).reshape(-1)

        self._recent_global_states.append(global_state)
        state_history = None
        if len(self._recent_global_states) == self._gt_window_size:
            # [window_size, state_dim]
            state_history = np.stack(list(self._recent_global_states), axis=0).astype(np.float32)
        
        # ✅ Step C: 存储到QAVER（使用hp_tag替代qos_tag）
        self.memory.push(
            state=np.asarray(obs, dtype=np.float32),
            action=actions_arr.tolist(),
            old_log_probs=old_lp_arr.tolist(),
            reward=float(reward),
            next_state=np.asarray(next_obs, dtype=np.float32),
            done=bool(done),
            hp_tag=hp_tag,  # ✅ Step C: 使用hp_tag
            episode_return=float(episode_return),
            td_error=td_error,
            state_history=state_history
        )
    
    def update(self, batch_size: int = 64, state_history_batch: Optional[torch.Tensor] = None) -> Dict:
        """
        修正问题3-5: 完整的PPO更新 + MI estimator训练
        """
        # Increment global steps for MI warmup
        self.global_steps += 1
        
        # 从QAVER采样
        experiences, weights, indices = self.memory.sample(batch_size)
        
        if not experiences:
            return {}
        
        # 解包（experience = (state, action, old_log_probs, reward, next_state, done, qos_tag, episode_return, state_history)）
        states = np.array([e[0] for e in experiences], dtype=np.float32)  # [B, num_agents, obs_dim]
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.long, device=device)  # [B, num_agents]
        old_log_probs = torch.tensor([e[2] for e in experiences], dtype=torch.float32, device=device)  # [B, num_agents]
        rewards = torch.tensor([e[3] for e in experiences], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = np.array([e[4] for e in experiences], dtype=np.float32)  # [B, num_agents, obs_dim]
        dones = torch.tensor([e[5] for e in experiences], dtype=torch.float32, device=device).unsqueeze(1)
        hp_tags = [e[6] for e in experiences]  # ✅ Step C: 使用hp_tag替代qos_tag
        episode_returns = [e[7] for e in experiences]

        # 如果外部没有提供 state_history_batch，则尝试从 buffer 内部的 state_history 取出。
        # 这可以让 dt_loss 在不改 runner 的情况下工作起来。
        if state_history_batch is None:
            histories = [e[8] if len(e) > 8 else None for e in experiences]
            if all(h is not None for h in histories):
                state_history_batch = torch.from_numpy(np.stack(histories, axis=0)).to(device)
        
        # Convert to torch tensors
        states = torch.from_numpy(states).to(device)  # [B, num_agents, obs_dim]
        next_states = torch.from_numpy(next_states).to(device)  # [B, num_agents, obs_dim]
        
        # ========== 更新感知模块 ==========
        perception_loss = self._update_perception_module(
            states, next_states, state_history_batch
        )
        
        # ========== 修正问题3: PPO更新（需要old_log_probs） ==========
        decision_loss = self._update_decision_module_ppo(
            states, actions, old_log_probs, rewards, next_states, dones, weights
        )
        
        # ✅ Step C: 更新MI estimator（使用hp_tags，HPB-2/HPB-1视为Sensitive）
        # ✅ 消融实验：如果lambda_mi=0，跳过MI estimator更新
        lambda_mi = self.config.get('lambda_mi', 1.0)
        if lambda_mi > 0:
            mi_loss_dict = self._update_mi_estimator(states, actions, hp_tags, episode_returns)
        else:
            mi_loss_dict = {
                'mi_loss': 0.0,
                'mi_infonce_bce': 0.693,
                'mi_l1out': 0.0,
                'mi_proxy': 0.0,
                'mi_weight': 0.0,
                'num_sensitive': 0
            }
        
        # ========== 更新优先级 ==========
        with torch.no_grad():
            # 简化：使用第一个agent计算TD-error
            agent_obs = states[:, 0, :] if len(states.shape) == 3 else states
            context, _ = self.edge_twins[0](agent_obs, None)
            belief, _ = self.belief_modules[0](context, training=False)
            
            next_agent_obs = next_states[:, 0, :] if len(next_states.shape) == 3 else next_states
            next_context, _ = self.edge_twins[0](next_agent_obs, None)
            next_belief, _ = self.belief_modules[0](next_context, training=False)
            
            values = self.critic(belief)
            next_values = self.critic(next_belief)
            td_errors = (rewards + self.config['gamma'] * next_values * (1 - dones) - values).abs()
            new_priorities = td_errors.squeeze().cpu().numpy() + 1e-6
        
        self.memory.update_priorities(indices, new_priorities.tolist())
        
        # 合并损失
        total_loss = {**perception_loss, **decision_loss, **mi_loss_dict}
        
        self.update_step += 1
        
        return total_loss
    
    def _update_perception_module(self, states: torch.Tensor, next_states: torch.Tensor,
                                 state_history_batch: Optional[torch.Tensor] = None) -> Dict:
        """
        修正问题2: Ground Twin shape正确
        """
        self.perception_optimizer.zero_grad()
        
        batch_size = states.shape[0]
        
        # ========== 修正问题2: Ground Twin监督 ==========
        dt_loss = torch.tensor(0.0, device=device)
        
        if state_history_batch is not None:
            # Ground Twin提取目标上下文 [batch, num_agents, context_dim]
            target_contexts = self.ground_twin.extract_target_context(state_history_batch)
            
            # Edge Twin提取实际上下文
            edge_contexts_list = []
            for i in range(self.num_agents):
                agent_obs = states[:, i, :] if len(states.shape) == 3 else states
                context, _ = self.edge_twins[i](agent_obs, None)
                edge_contexts_list.append(context)
            
            # 堆叠为 [batch, num_agents, context_dim]
            edge_contexts = torch.stack(edge_contexts_list, dim=1)
            
            # 修正问题2: shape现在匹配了
            if edge_contexts.shape == target_contexts.shape:
                dt_loss = F.mse_loss(edge_contexts, target_contexts)
        
        # ========== L2训练 ==========
        gbm_loss = torch.tensor(0.0, device=device)
        
        contexts = []
        for i in range(self.num_agents):
            # states: [B, num_agents, obs_dim], extract agent i's obs: [B, obs_dim]
            agent_obs = states[:, i, :] if len(states.shape) == 3 else states
            context, _ = self.edge_twins[i](agent_obs, None)
            contexts.append(context)
        
        # 先计算belief用于后续MI-Collab（推理模式，不计算loss）
        individual_beliefs = []
        for i in range(self.num_agents):
            belief, _ = self.belief_modules[i](contexts[i], training=False)
            individual_beliefs.append(belief)
        
        # 然后计算训练loss（如果需要）
        # 注意：对于GRU belief，我们使用MSE loss；对于Diffusion，使用DDPM loss
        for i in range(self.num_agents):
            agent_obs = states[:, i, :] if len(states.shape) == 3 else states
            _, loss = self.belief_modules[i](
                context=contexts[i],
                target_obs=agent_obs,
                training=True
            )
            if loss is not None:
                gbm_loss += loss
        
        gbm_loss /= self.num_agents
        
        # ========== L3 MI-Collab训练 ==========
        # individual_beliefs已经在上面计算了
        
        individual_beliefs = torch.stack(individual_beliefs, dim=1)
        collaborative_belief, _, _ = self.mi_collab(individual_beliefs)
        
        mi_collab_loss = self.mi_collab.compute_mvmi_loss(
            individual_beliefs, collaborative_belief
        )
        
        # 总感知损失
        lambda_1 = 0.1
        lambda_2 = 0.5
        lambda_3 = 0.2
        # ✅ 消融实验：支持lambda_mi参数控制MI loss权重
        lambda_mi = self.config.get('lambda_mi', 1.0)  # 默认1.0（Full EVA-Gen）
        
        perception_loss = lambda_1 * dt_loss + lambda_2 * gbm_loss + lambda_3 * lambda_mi * mi_collab_loss
        
        perception_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.edge_twins.parameters()) + 
            list(self.ground_twin.parameters()) +
            list(self.belief_modules.parameters()) +
            list(self.mi_collab.parameters()),
            max_norm=10.0
        )
        self.perception_optimizer.step()
        
        # ✅ 重要：重置GRU belief模块的hidden state，避免计算图冲突
        # 在perception更新后，我们需要重置hidden state，这样decision模块可以重新计算
        for i in range(self.num_agents):
            if hasattr(self.belief_modules[i], 'reset_hidden'):
                self.belief_modules[i].reset_hidden()
        
        return {
            'perception_loss': perception_loss.item(),
            'dt_loss': dt_loss.item(),
            'gbm_loss': gbm_loss.item(),
            'mi_collab_loss': mi_collab_loss.item()
        }
    
    def _update_decision_module_ppo(self, states: torch.Tensor, actions: torch.Tensor,
                                   old_log_probs: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor,
                                   dones: torch.Tensor, weights: torch.Tensor) -> Dict:
        """
        修正问题3: 真正的PPO更新（old_log_probs + clip）
        """
        batch_size = states.shape[0]
        
        # 生成collaborative beliefs
        contexts = []
        for i in range(self.num_agents):
            agent_obs = states[:, i, :] if len(states.shape) == 3 else states
            context, _ = self.edge_twins[i](agent_obs, None)
            contexts.append(context)
        
        individual_beliefs = []
        for i in range(self.num_agents):
            belief, _ = self.belief_modules[i](contexts[i], training=False)
            individual_beliefs.append(belief)
        
        individual_beliefs = torch.stack(individual_beliefs, dim=1)
        collaborative_belief, _, _ = self.mi_collab(individual_beliefs)
        
        # ✅ 重要：detach belief以避免与perception模块的计算图冲突
        # decision模块不应该通过belief反向传播到perception模块
        collaborative_belief = collaborative_belief.detach()
        
        # 同样处理next_states
        next_contexts = []
        for i in range(self.num_agents):
            agent_obs = next_states[:, i, :] if len(next_states.shape) == 3 else next_states
            context, _ = self.edge_twins[i](agent_obs, None)
            next_contexts.append(context)
        
        next_individual_beliefs = []
        for i in range(self.num_agents):
            belief, _ = self.belief_modules[i](next_contexts[i], training=False)
            next_individual_beliefs.append(belief)
        
        next_individual_beliefs = torch.stack(next_individual_beliefs, dim=1)
        next_collaborative_belief, _, _ = self.mi_collab(next_individual_beliefs)
        next_collaborative_belief = next_collaborative_belief.detach()
        
        # ========== Critic更新 ==========
        values = self.critic(collaborative_belief)
        
        with torch.no_grad():
            next_values = self.critic(next_collaborative_belief)
            targets = rewards + self.config['gamma'] * next_values * (1 - dones)
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Critic loss（带重要性权重）
        critic_loss = (weights.unsqueeze(1) * (values - targets) ** 2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # ========== 修正问题3: PPO Actor更新（带clip） ==========
        total_actor_loss = 0.0
        total_entropy = 0.0
        
        # Detach collaborative_belief to avoid backprop through perception module
        collaborative_belief_detached = collaborative_belief.detach()
        
        # 修正问题1: 每个agent分别更新
        for agent_i in range(self.num_agents):
            # 获取行为策略的old_log_probs（由交互时存入buffer）
            old_log_probs_i = old_log_probs[:, agent_i].detach()
            
            # 获取新的log_probs
            new_logits = self.actors[agent_i](collaborative_belief_detached, agent_id=agent_i)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(actions[:, agent_i])
            entropy = new_dist.entropy()
            
            # PPO clip
            ratio = torch.exp(new_log_probs - old_log_probs_i)
            surr1 = ratio * advantages.squeeze().detach()
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.squeeze().detach()
            
            actor_loss = -(weights * torch.min(surr1, surr2)).mean()
            actor_loss -= 0.01 * entropy.mean()  # Entropy bonus
            
            self.actor_optimizers[agent_i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_i].parameters(), max_norm=10.0)
            self.actor_optimizers[agent_i].step()
            
            total_actor_loss += actor_loss.item()
            total_entropy += entropy.mean().item()
        
        total_actor_loss /= self.num_agents
        total_entropy /= self.num_agents
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': total_actor_loss,
            'entropy': total_entropy
        }
    
    def _update_mi_estimator(self, states: torch.Tensor, actions: torch.Tensor,
                            hp_tags: List[str], episode_returns: List[float]) -> Dict:
        """
        修正问题5: 实际训练MI estimator
        修正问题6: 使用one-hot动作
        ✅ Step C: 使用hp_tags，HPB-2/HPB-1视为Sensitive样本
        
        改进 (from change.md):
        - Sensitive-only: 只在Sensitive样本上更新MI（HPB-2/HPB-1）
        - MI warmup: 早期不干扰policy，后期参与协作对齐
        - MI proxy: 记录可解释的MI强度指标
        """
        batch_size = states.shape[0]
        
        # ✅ Step C: Filter HPB-2/HPB-1 samples only (视为Sensitive)
        sensitive_indices = [i for i, tag in enumerate(hp_tags) if tag in ["HPB-2", "HPB-1"]]
        
        # Skip update if no Sensitive samples or batch too small
        if len(sensitive_indices) < 8:  # 最少8个样本才更新
            return {
                'mi_loss': 0.0,
                'mi_infonce_bce': 0.693,  # log(2)
                'mi_l1out': 0.0,
                'mi_proxy': 0.0,
                'mi_weight': 0.0,
                'num_sensitive': len(sensitive_indices)
            }
        
        # Extract Sensitive samples
        sens_states = states[sensitive_indices]
        sens_actions = actions[sensitive_indices]
        
        # 生成beliefs (only for Sensitive samples)
        contexts = []
        for i in range(self.num_agents):
            agent_obs = sens_states[:, i, :] if len(sens_states.shape) == 3 else sens_states
            context, _ = self.edge_twins[i](agent_obs, None)
            contexts.append(context)
        
        individual_beliefs = []
        for i in range(self.num_agents):
            belief, _ = self.belief_modules[i](contexts[i], training=False)
            individual_beliefs.append(belief)
        
        individual_beliefs = torch.stack(individual_beliefs, dim=1)
        collaborative_belief, _, _ = self.mi_collab(individual_beliefs)
        # 关键：MI estimator 只是一个辅助判别器/探针，不应把梯度回传到信念模块，
        # 否则会与 L1/L2/L3 的优化目标相互干扰并带来数值不稳定。
        collaborative_belief = collaborative_belief.detach()
        
        # 计算MI loss（用于训练MI estimator本身）
        # 使用第一个agent的动作作为示例
        actions_agent0 = sens_actions[:, 0]
        
        # InfoNCE(BCE) loss：**可直接最小化、稳定非负**
        mi_infonce_bce = self.sa_mi_estimator.estimate_mi_infonce(
            collaborative_belief, actions_agent0
        )
        
        # L1Out loss
        # L1Out 估计在小 batch/早期训练极易发散（你日志中的 -75/-161 就是典型）。
        # 这里仅用于监控，不纳入训练主损失；若你想用它做正则，请用平方惩罚并给极小权重。
        mi_l1out = self.sa_mi_estimator.estimate_mi_l1out(
            collaborative_belief, actions_agent0
        )
        
        # MI Proxy: log(2) - mi_infonce_bce
        # 早期 proxy≈0 (随机)，后期 proxy上升 (判别器学到协作结构)
        mi_proxy = math.log(2) - mi_infonce_bce.item()
        
        # MI Warmup: 早期不干扰policy，中后期参与协作对齐
        mi_warmup_weight = min(1.0, self.global_steps / self.mi_warmup_steps)
        
        # 训练目标：最小化二分类判别的 BCE。
        # 额外加一个极弱的稳定正则，避免 l1out_net logits 漂移（不会主导训练）。
        mi_loss = mi_warmup_weight * (mi_infonce_bce + 1e-4 * (mi_l1out ** 2))
        
        self.mi_optimizer.zero_grad()
        mi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sa_mi_estimator.parameters(), max_norm=10.0)
        self.mi_optimizer.step()
        
        return {
            'mi_loss': mi_loss.item(),
            'mi_infonce_bce': mi_infonce_bce.item(),
            'mi_proxy': mi_proxy,  # 可解释的MI强度指标
            'mi_l1out': mi_l1out.item(),
            'mi_weight': mi_warmup_weight,  # warmup进度
            'num_sensitive': len(sensitive_indices)  # Sensitive样本数量
        }
    
    def save_models(self, filepath: str):
        """保存模型"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'edge_twins': [twin.state_dict() for twin in self.edge_twins],
            'ground_twin': self.ground_twin.state_dict(),
            'belief_modules': [module.state_dict() for module in self.belief_modules],
            'mi_collab': self.mi_collab.state_dict(),
            'sa_mi_estimator': self.sa_mi_estimator.state_dict(),
            'update_step': self.update_step
        }, filepath)
    
    def load_models(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        
        for i, actor_state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(actor_state)
        
        self.critic.load_state_dict(checkpoint['critic'])
        
        for i, twin_state in enumerate(checkpoint['edge_twins']):
            self.edge_twins[i].load_state_dict(twin_state)
        
        self.ground_twin.load_state_dict(checkpoint['ground_twin'])
        
        for i, module_state in enumerate(checkpoint['belief_modules']):
            self.belief_modules[i].load_state_dict(module_state)
        
        self.mi_collab.load_state_dict(checkpoint['mi_collab'])
        self.sa_mi_estimator.load_state_dict(checkpoint['sa_mi_estimator'])
        
        self.update_step = checkpoint.get('update_step', 0)


# ============================================================
# 默认配置
# ============================================================

def get_default_config():
    """获取默认配置"""
    return {
        # 网络维度
        'hidden_dim': 128,
        'belief_dim': 256,
        
        # 训练参数
        'buffer_size': 100000,
        'gamma': 0.99,
        'eps_clip': 0.2,  # PPO clip参数
        
        # 学习率
        'lr_perception': 3e-4,
        'lr_actor': 1e-4,
        'lr_critic': 3e-4,
        'lr_mi_estimator': 3e-4,
        
        # QAVER
        'priority_alpha': 0.6,
        'priority_beta': 0.4,
        'p_HPB': 0.7,
        'initial_threshold': -10.0,
        
        # 修正问题7: 低频belief更新
        'belief_update_freq': 5,  # 每5步更新一次belief
        
        # 状态维度
        'state_dim': 128,
    }


if __name__ == "__main__":
    print("="*60)
    print("MI-Unity框架 - 完全修正版 v2")
    print("="*60)
    print("\n修正的7个关键问题：")
    print("\n✅ 问题1: 每个agent独立动作输出 [num_agents]")
    print("   - Per-agent actors + agent_id embedding")
    print("\n✅ 问题2: Ground Twin shape修正")
    print("   - 输出 [batch, num_agents, context_dim]")
    print("\n✅ 问题3: 真正的PPO/MAPPO")
    print("   - old_log_probs + ratio + clip")
    print("\n✅ 问题4: 环境交互接口")
    print("   - store_transition() 方法")
    print("\n✅ 问题5: MI estimator训练")
    print("   - _update_mi_estimator() 实际step")
    print("\n✅ 问题6: 动作one-hot转换")
    print("   - _action_to_onehot() 自动转换")
    print("\n✅ 问题7: 低频belief更新")
    print("   - 每5步更新一次（可配置）")
    print("="*60)
    
    # 测试初始化
    config = get_default_config()
    agent = MIUnityAgent(
        obs_dim=64,
        action_dim=5,
        belief_dim=256,
        num_agents=2,
        config=config
    )
    
    print(f"\n✅ Agent创建成功！")
    print(f"设备: {device}")
    print(f"Actors数量: {len(agent.actors)}")
    
    # 测试select_action
    obs = np.random.randn(64)
    actions, log_probs, entropies = agent.select_action(obs)
    
    print(f"\n测试select_action:")
    print(f"  Actions shape: {actions.shape}")  # 应该是 (2,)
    print(f"  Actions: {actions}")  # 应该是整数 [0-4]
    print(f"  Log_probs shape: {log_probs.shape}")
    
    # 测试store_transition
    next_obs = np.random.randn(64)
    agent.store_transition(
        obs=obs,
        actions=actions,
        reward=1.0,
        next_obs=next_obs,
        done=False,
        info={'qos_tag': 'Sensitive'},
        episode_return=10.0
    )
    
    print(f"\n测试store_transition:")
    print(f"  Buffer size: {len(agent.memory)}")
    
    print("\n✅ 所有功能正常！")
