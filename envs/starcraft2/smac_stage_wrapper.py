"""
SMAC Stage-1/Stage-2 Wrapper
根据 smac场景修改.md 文档实现的增强型 wrapper

Stage-1: 基础稀疏奖励
Stage-2: 添加遮挡、噪声、诱饵奖励和关键事件

支持三张地图的特定改造：
- 2m_vs_1z: VIP-Kite (VIP存活+风险敏感)
- 3m: Commander-First (高价值目标优先级)
- 8m: Two-VIP + Partial Blackout (大规模对齐)
"""

import numpy as np
from gym import Wrapper


class SMACStageWrapper(Wrapper):
    """
    SMAC Stage-1/Stage-2 Wrapper
    实现文档中定义的所有增强功能
    """
    
    def __init__(self, env, args):
        super().__init__(env)
        self.args = args
        self.map_name = args.map_name
        self.smac_stage = getattr(args, 'smac_stage', 1)
        
        # ========== Occlusion (遮挡/碎片化) ==========
        self.obs_dropout_p = getattr(args, 'obs_dropout_p', 0.0)  # 敌方观测dropout概率
        self.comm_dropout_p = getattr(args, 'comm_dropout_p', 0.0)  # 队友通信dropout概率
        
        # ========== Noise (观测噪声) ==========
        self.obs_noise_std = getattr(args, 'obs_noise_std', 0.0)  # 高斯噪声标准差
        
        # ========== Dummy Reward (诱饵梯度) ==========
        self.dummy_reward_scale = getattr(args, 'dummy_reward_scale', 0.0)  # 伤害奖励权重
        
        # ========== Map-specific Critical Events (关键事件) ==========
        # 2m_vs_1z: VIP-Kite
        if '2m_vs_1z' in self.map_name or '2m1z' in self.map_name:
            self.vip_ids = [getattr(args, 'vip_id', 0)]  # VIP单位ID
            self.vip_death_penalty = getattr(args, 'vip_death_penalty', 0.0)
            self.vip_alive_bonus = getattr(args, 'vip_alive_bonus', 0.0)
            self.critical_event_type = 'vip_kite'
        
        # 3m: Commander-First
        elif '3m' in self.map_name and '8m' not in self.map_name:
            self.commander_id = getattr(args, 'commander_id', 0)  # Commander敌方ID
            self.commander_first_bonus = getattr(args, 'commander_first_bonus', 0.0)
            self.commander_kill_bonus = getattr(args, 'commander_kill_bonus', 0.0)
            self.critical_event_type = 'commander_first'
        
        # 8m: Two-VIP + Partial Blackout
        elif '8m' in self.map_name:
            self.vip_ids = getattr(args, 'vip_ids', [0, 1])  # 两个VIP
            self.vip_death_penalty = getattr(args, 'vip_death_penalty', 0.0)
            self.vip_all_alive_bonus = getattr(args, 'vip_all_alive_bonus', 0.0)
            self.critical_event_type = 'two_vip'
        
        else:
            self.critical_event_type = 'none'
        
        # ========== Episode Tracking ==========
        self.episode_info = {}
        self.reset_episode_stats()
        
        # ========== Damage Tracking (for dummy reward) ==========
        self.prev_enemy_health = {}
        
        # ========== Commander Tracking ==========
        if self.critical_event_type == 'commander_first':
            self.commander_killed = False
            self.commander_killed_first = False
            self.other_enemies_killed_before_commander = False
        
        print(f"\n{'='*60}")
        print(f"SMACStageWrapper Initialized")
        print(f"{'='*60}")
        print(f"Map: {self.map_name}")
        print(f"Stage: {self.smac_stage}")
        print(f"Critical Event Type: {self.critical_event_type}")
        print(f"Obs Dropout: {self.obs_dropout_p}")
        print(f"Comm Dropout: {self.comm_dropout_p}")
        print(f"Obs Noise: {self.obs_noise_std}")
        print(f"Dummy Reward Scale: {self.dummy_reward_scale}")
        if self.critical_event_type == 'vip_kite':
            print(f"VIP IDs: {self.vip_ids}")
            print(f"VIP Death Penalty: {self.vip_death_penalty}")
            print(f"VIP Alive Bonus: {self.vip_alive_bonus}")
        elif self.critical_event_type == 'commander_first':
            print(f"Commander ID: {self.commander_id}")
            print(f"Commander First Bonus: {self.commander_first_bonus}")
        elif self.critical_event_type == 'two_vip':
            print(f"VIP IDs: {self.vip_ids}")
            print(f"VIP Death Penalty: {self.vip_death_penalty}")
            print(f"VIP All Alive Bonus: {self.vip_all_alive_bonus}")
        print(f"{'='*60}\n")
    
    def reset_episode_stats(self):
        """重置episode级别的统计"""
        self.episode_info = {
            # 基础指标
            'win': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            
            # Critical Event 指标
            'vip_dead': False,  # VIP是否死亡
            'vips_all_alive': True,  # 所有VIP是否存活
            'commander_killed': False,  # Commander是否被击杀
            'commander_killed_first': False,  # Commander是否首个被击杀
            'critical_success': False,  # 关键成功（win + 关键条件）
            'catastrophic_failure': False,  # 灾难性失败（VIP死亡）
            
            # Dummy reward 累计
            'total_damage_reward': 0.0,
            'total_critical_reward': 0.0,
        }
        
        # Reset commander tracking
        if self.critical_event_type == 'commander_first':
            self.commander_killed = False
            self.commander_killed_first = False
            self.other_enemies_killed_before_commander = False
    
    def reset(self):
        """重置环境"""
        obs, share_obs, available_actions = self.env.reset()
        self.reset_episode_stats()
        
        # 初始化敌方血量跟踪
        self.prev_enemy_health = {}
        try:
            for e_id in range(self.env.n_enemies):
                enemy = self.env.enemies[e_id]
                self.prev_enemy_health[e_id] = enemy.health if hasattr(enemy, 'health') else enemy.health_max
        except:
            pass
        
        # 应用观测修改（dropout + noise）
        obs_modified = self._apply_obs_modifications(obs)
        
        return obs_modified, share_obs, available_actions
    
    def step(self, actions):
        """执行一步并应用Stage-2增强"""
        obs, share_obs, reward, done, info, available_actions = self.env.step(actions)
        
        # ========== Stage-2 增强 ==========
        if self.smac_stage == 2:
            # 1. 计算dummy reward (诱饵梯度)
            dummy_reward = self._compute_dummy_reward()
            
            # 2. 计算critical event奖励/惩罚
            critical_reward, critical_penalty = self._compute_critical_rewards()
            
            # 3. 应用观测修改（dropout + noise）
            obs = self._apply_obs_modifications(obs)
            
            # 4. 修改总奖励
            # reward格式：[[r1], [r2], ...] - 每个智能体的reward是一个单元素列表
            # 根据StarCraft2_Env.py第642行，rewards = [[reward]]*self.n_agents
            total_bonus = dummy_reward + critical_reward - critical_penalty
            
            # 处理reward：对每个智能体的reward列表中的值添加bonus
            if isinstance(reward, list):
                # 列表格式：[[r1], [r2], ...] 或 [r1, r2, ...]
                new_reward = []
                for r in reward:
                    if isinstance(r, (list, np.ndarray)):
                        # 子列表：[[r1], [r2]] -> [[r1+bonus], [r2+bonus]]
                        new_reward.append([r_elem + total_bonus for r_elem in r])
                    else:
                        # 单个值：[r1, r2] -> [r1+bonus, r2+bonus]
                        new_reward.append(r + total_bonus)
                reward = new_reward
            elif isinstance(reward, np.ndarray):
                # numpy数组：直接添加
                reward = reward + total_bonus
            else:
                # 标量
                reward = reward + total_bonus
            
            # 5. 更新episode info
            self.episode_info['total_damage_reward'] += dummy_reward
            self.episode_info['total_critical_reward'] += (critical_reward - critical_penalty)
        
        # ========== 更新Episode统计 ==========
        self._update_episode_stats(info)
        
        # ========== 在done时添加Critical Event指标到info ==========
        if hasattr(done, '__iter__') and not isinstance(done, str):
            is_done = done.all() if hasattr(done, 'all') else all(done)
        else:
            is_done = done
        
        if is_done:
            # info是一个列表，我们需要将信息添加到第一个元素（通常用于全局info）
            if isinstance(info, list):
                if len(info) > 0 and isinstance(info[0], dict):
                    info_dict = info[0]
                else:
                    # 如果info[0]不是字典，创建一个新字典并添加到列表
                    info_dict = {}
                    if len(info) == 0:
                        info.append(info_dict)
                    else:
                        info[0] = info_dict
            else:
                info_dict = info
            
            info_dict['smac_stage'] = self.smac_stage
            info_dict['vip_dead'] = int(self.episode_info['vip_dead'])
            info_dict['critical_success'] = int(self.episode_info['critical_success'])
            info_dict['catastrophic_failure'] = int(self.episode_info['catastrophic_failure'])
            info_dict['total_damage_reward'] = self.episode_info['total_damage_reward']
            info_dict['total_critical_reward'] = self.episode_info['total_critical_reward']
            
            # Map-specific metrics
            if self.critical_event_type == 'vip_kite':
                info_dict['vips_all_alive'] = int(self.episode_info['vips_all_alive'])
            elif self.critical_event_type == 'commander_first':
                info_dict['commander_killed'] = int(self.episode_info['commander_killed'])
                info_dict['commander_killed_first'] = int(self.episode_info['commander_killed_first'])
            elif self.critical_event_type == 'two_vip':
                info_dict['vips_all_alive'] = int(self.episode_info['vips_all_alive'])
        
        return obs, share_obs, reward, done, info, available_actions
    
    def _apply_obs_modifications(self, obs):
        """应用观测修改：dropout + noise"""
        if self.smac_stage != 2:
            return obs
        
        obs_modified = []
        for agent_id, agent_obs in enumerate(obs):
            agent_obs = np.array(agent_obs, dtype=np.float32)
            
            # 1. Enemy observation dropout (敌方信息缺失)
            if self.obs_dropout_p > 0:
                agent_obs = self._apply_enemy_dropout(agent_obs, agent_id)
            
            # 2. Ally communication dropout (队友信息缺失)
            if self.comm_dropout_p > 0:
                agent_obs = self._apply_ally_dropout(agent_obs, agent_id)
            
            # 3. Observation noise (观测噪声)
            if self.obs_noise_std > 0:
                agent_obs = self._apply_obs_noise(agent_obs)
            
            obs_modified.append(agent_obs)
        
        return obs_modified
    
    def _apply_enemy_dropout(self, obs, agent_id):
        """
        对敌方特征进行dropout
        假设SMAC观测格式：[move_feats, enemy_feats, ally_feats, own_feats]
        """
        # 这里需要根据实际的SMAC观测维度进行调整
        # 简化实现：对所有特征以概率p进行dropout
        if np.random.random() < self.obs_dropout_p:
            # 估算敌方特征的位置（通常在观测的中间部分）
            # 这里采用保守策略：对观测的部分维度进行dropout
            n_features = len(obs)
            # 假设敌方特征占观测的30%-70%
            enemy_start = int(n_features * 0.3)
            enemy_end = int(n_features * 0.7)
            obs[enemy_start:enemy_end] = 0.0
        
        return obs
    
    def _apply_ally_dropout(self, obs, agent_id):
        """
        对队友特征进行dropout
        模拟通信中断
        """
        if np.random.random() < self.comm_dropout_p:
            # 估算队友特征的位置（通常在观测的后部）
            n_features = len(obs)
            # 假设队友特征占观测的70%-90%
            ally_start = int(n_features * 0.7)
            ally_end = int(n_features * 0.9)
            obs[ally_start:ally_end] = 0.0
        
        return obs
    
    def _apply_obs_noise(self, obs):
        """
        对观测添加高斯噪声
        只对连续特征添加噪声（相对位置、距离等）
        """
        # 添加噪声
        noise = np.random.randn(*obs.shape) * self.obs_noise_std
        obs_noisy = obs + noise
        
        # 裁剪到合理范围（假设观测已归一化到[0, 1]或[-1, 1]）
        obs_noisy = np.clip(obs_noisy, -1.0, 1.0)
        
        return obs_noisy
    
    def _compute_dummy_reward(self):
        """
        计算诱饵奖励（伤害奖励）
        对敌方造成伤害给予小额密集奖励
        """
        if self.dummy_reward_scale == 0:
            return 0.0
        
        total_damage = 0.0
        try:
            for e_id in range(self.env.n_enemies):
                enemy = self.env.enemies[e_id]
                if not enemy.health_max or enemy.health_max == 0:
                    continue
                
                current_health = enemy.health if hasattr(enemy, 'health') else 0
                prev_health = self.prev_enemy_health.get(e_id, current_health)
                
                damage = max(0, prev_health - current_health)
                total_damage += damage
                
                # 更新血量
                self.prev_enemy_health[e_id] = current_health
        except:
            pass
        
        # 诱饵奖励 = 伤害 * 权重
        dummy_reward = total_damage * self.dummy_reward_scale
        
        return dummy_reward
    
    def _compute_critical_rewards(self):
        """
        计算关键事件的奖励和惩罚
        返回: (critical_reward, critical_penalty)
        """
        critical_reward = 0.0
        critical_penalty = 0.0
        
        try:
            # 2m_vs_1z: VIP-Kite
            if self.critical_event_type == 'vip_kite':
                for vip_id in self.vip_ids:
                    if vip_id < len(self.env.agents):
                        vip = self.env.agents[vip_id]
                        if vip.health <= 0 and not self.episode_info['vip_dead']:
                            # VIP死亡 -> 灾难性惩罚
                            critical_penalty = self.vip_death_penalty
                            self.episode_info['vip_dead'] = True
                            self.episode_info['vips_all_alive'] = False
                            self.episode_info['catastrophic_failure'] = True
            
            # 3m: Commander-First
            elif self.critical_event_type == 'commander_first':
                # 检查commander是否被击杀
                if self.commander_id < len(self.env.enemies):
                    commander = self.env.enemies[self.commander_id]
                    if commander.health <= 0 and not self.commander_killed:
                        self.commander_killed = True
                        self.episode_info['commander_killed'] = True
                        
                        # 检查是否首个击杀
                        if not self.other_enemies_killed_before_commander:
                            self.commander_killed_first = True
                            self.episode_info['commander_killed_first'] = True
                            # 首杀commander -> 关键奖励
                            critical_reward = self.commander_first_bonus
                
                # 检查其他敌人是否被击杀
                for e_id in range(self.env.n_enemies):
                    if e_id != self.commander_id:
                        enemy = self.env.enemies[e_id]
                        if enemy.health <= 0 and not self.commander_killed:
                            self.other_enemies_killed_before_commander = True
            
            # 8m: Two-VIP
            elif self.critical_event_type == 'two_vip':
                all_vips_alive = True
                for vip_id in self.vip_ids:
                    if vip_id < len(self.env.agents):
                        vip = self.env.agents[vip_id]
                        if vip.health <= 0:
                            all_vips_alive = False
                            if not self.episode_info['vip_dead']:
                                # 任一VIP死亡 -> 惩罚
                                critical_penalty = self.vip_death_penalty
                                self.episode_info['vip_dead'] = True
                                self.episode_info['catastrophic_failure'] = True
                
                self.episode_info['vips_all_alive'] = all_vips_alive
        
        except Exception as e:
            # 如果访问单位属性出错，忽略（可能是环境未完全初始化）
            pass
        
        return critical_reward, critical_penalty
    
    def _update_episode_stats(self, info):
        """更新episode统计信息"""
        # 基础统计
        if 'battle_won' in info:
            self.episode_info['win'] = bool(info['battle_won'])
        
        if 'dead_allies' in info:
            self.episode_info['dead_allies'] = int(info['dead_allies'])
        
        if 'dead_enemies' in info:
            self.episode_info['dead_enemies'] = int(info['dead_enemies'])
        
        # 计算Critical Success
        win = self.episode_info['win']
        
        if self.critical_event_type == 'vip_kite':
            # Win + VIP存活
            self.episode_info['critical_success'] = win and not self.episode_info['vip_dead']
        
        elif self.critical_event_type == 'commander_first':
            # Win + Commander首杀
            self.episode_info['critical_success'] = win and self.episode_info['commander_killed_first']
        
        elif self.critical_event_type == 'two_vip':
            # Win + 所有VIP存活
            self.episode_info['critical_success'] = win and self.episode_info['vips_all_alive']
        
        else:
            # 无特殊条件，critical_success = win
            self.episode_info['critical_success'] = win
