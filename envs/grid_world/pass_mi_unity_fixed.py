"""
Pass environment with MI-Unity features (Stage-1/Stage-2 support)
Based on Pass environment, adding: ghost switches, coins, noise, unified metrics
"""
import copy
import gym
import numpy as np
from gym.spaces import Box, Discrete
from gym.utils import seeding


class Entity():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
class Agent(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.movable = True


class Landmark(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.movable = False
        

class Door(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.open = False


class PassMIUnity(gym.Env):
    """Pass environment with Stage-1/Stage-2 curriculum support"""
    
    def __init__(self, map_ind=0, max_timesteps=300, door_in_obs=False, full_obs=False, joint_count=False, 
                 activate_radius=None, grid_size=30,
                 # MI-Unity params
                 curriculum_stage=2,
                 obs_noise_level=0.3,
                 dummy_switch_ratio=0.5,
                 coin_density=50,
                 coin_respawn=False,
                 coin_reward_scale=0.1,
                 **kwargs):
        self.num_agents = 2
        self.max_timesteps = max_timesteps
        self.joint_count = joint_count
        self.grid_size = grid_size
        self.door_in_obs = door_in_obs
        self.full_obs = full_obs
        
        # MI-Unity parameters
        self.curriculum_stage = curriculum_stage
        self.obs_noise_level = obs_noise_level if curriculum_stage >= 2 else 0.0
        self.dummy_switch_ratio = dummy_switch_ratio if curriculum_stage >= 2 else 0.0
        self.coin_density = coin_density if curriculum_stage >= 2 else 0
        self.coin_respawn = coin_respawn
        self.coin_reward_scale = coin_reward_scale
        
        self._init_entity(map_ind, grid_size)
        
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, self.grid_size // 2] = 1
        self.time = 0
        self.agents, self.wall_map, self.door = None, None, None
        self.door_radius = self.grid_size // 10
        self.activate_radius = 1.5 * self.door_radius if activate_radius is None else activate_radius

        self._init_space()
        
        if self.joint_count:
            self.visit_counts = np.zeros(self.num_agents * [self.grid_size, self.grid_size])
        else:
            self.visit_counts = np.zeros((self.num_agents, self.grid_size, self.grid_size))
        
        # Initialize episode tracking (统一使用0/1，不用True/False)
        self.episode_has_door_open = 0
        self.episode_has_crossed = 0
        self.episode_has_any_target = 0
        self.episode_has_all_target = 0  # ✅ 添加all_target追踪
        self.t_open = -1
        self.t_cross = -1
        self.t_target = -1
        self.total_coin_reward = 0.0
        self.total_target_reward = 0.0
        
        # ✅ 添加缺失的统计字段（与SecretRoomMIUnity一致）
        self.switch_steps = 0  # 激活开关的累计步数
        self.door_steps = 0  # 门打开的累计步数
        self.max_y_reached_0 = 0.0  # agent 0 达到的最大y坐标
        self.max_y_reached_1 = 0.0  # agent 1 达到的最大y坐标
        self.time_first_reach_y_15_0 = -1  # agent 0 第一次到达y=15的时间
        self.time_first_reach_y_15_1 = -1  # agent 1 第一次到达y=15的时间
        
        # ✅ 调试模式（可通过环境变量控制）
        import os
        self.debug_mode = os.getenv('PASS_DEBUG', '0') == '1'
        
        self.reset()
    
    def _init_entity(self, map_ind, grid_size):
        # ✅ 统一坐标定义：使用同一套常量
        self.door_y_position = grid_size // 2  # 门的y坐标
        self.target_y_threshold = int(grid_size * 0.8)  # 目标区域的y阈值
        self.mid_y_threshold = grid_size // 2  # 中线阈值（用于crossed_door和max_y统计）
        
        if map_ind == 0:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_door = Door(grid_size // 2, self.door_y_position)
            # Real switches
            self.real_switches = [Landmark(grid_size // 10, int(grid_size * 0.8)),
                                  Landmark(int(grid_size * 0.8), grid_size // 10)]
            # Dummy switches (for Stage-2)
            self.dummy_switches = []
            if self.dummy_switch_ratio > 0:
                # Add mirrored dummy switches
                self.dummy_switches.append(Landmark(grid_size // 10, int(grid_size * 0.2)))
                self.dummy_switches.append(Landmark(int(grid_size * 0.2), grid_size // 10))
        else:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_door = Door(grid_size // 2, self.door_y_position)
            self.real_switches = [Landmark(grid_size // 10, int(grid_size * 0.8)),
                                  Landmark(int(grid_size * 0.8), grid_size // 10)]
            self.dummy_switches = []
            if self.dummy_switch_ratio > 0:
                self.dummy_switches.append(Landmark(grid_size // 10, int(grid_size * 0.2)))
                self.dummy_switches.append(Landmark(int(grid_size * 0.2), grid_size // 10))
        
        # Initialize coins for Stage-2 (will be generated in seed/reset)
        self.coins = []
        
    def _generate_coins(self):
        """Generate coins deterministically using self.np_random."""
        self.coins = []
        n_coins = int(getattr(self, "coin_density", 0) or 0)
        if n_coins <= 0:
            return
        for _ in range(n_coins):
            x = int(self.np_random.randint(1, self.grid_size-1))
            y = int(self.np_random.randint(1, max(2, self.grid_size//2)))
            self.coins.append(Landmark(x, y))
        
    def _init_space(self):
        if self.door_in_obs:
            if self.full_obs:
                self.observation_space = [Box(low=-1, high=1, shape=(5,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(5,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(10,), dtype=np.float32)]
            else:
                self.observation_space = [Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(3,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(6,), dtype=np.float32)]
        else:
            if self.full_obs:
                self.observation_space = [Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(4,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(8,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(8,), dtype=np.float32)]
            else:
                self.observation_space = [Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(2,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(4,), dtype=np.float32)]
        self.action_space = [Discrete(4), Discrete(4)]
    
    @staticmethod
    def ind2ndoor(ind):
        return 1
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed 
        
    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.door = copy.deepcopy(self.init_door)
        self.time = 0

        # Stage-1: door default open
        if self.curriculum_stage == 1:
            self.door.open = True
            # ✅ 修复：打开整列墙（让agents可以从任何x位置通过）
            self.wall_map[:, self.door.y] = 0
        
        # ✅ Reset episode tracking（必须重置所有flag和时间戳）
        self.episode_has_door_open = 1 if self.curriculum_stage == 1 else 0
        self.episode_has_crossed = 0
        self.episode_has_any_target = 0
        self.episode_has_all_target = 0
        self.t_open = 0 if self.curriculum_stage == 1 else -1
        self.t_cross = -1
        self.t_target = -1
        self.total_coin_reward = 0.0
        self.total_target_reward = 0.0
        
        # ✅ 重置统计字段
        self.switch_steps = 0
        self.door_steps = 0
        self.max_y_reached_0 = 0.0
        self.max_y_reached_1 = 0.0
        self.time_first_reach_y_15_0 = -1
        self.time_first_reach_y_15_1 = -1
        
        # Reset coins (使用统一的随机源)
        if self.coin_respawn or self.time == 0:
            if self.coin_density > 0:
                self._generate_coins()  # 使用_generate_coins方法
        
        if self.joint_count:
            visit_indices = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_indices] += 1
        else:
            for idx, agent in enumerate(self.agents):
                self.visit_counts[idx, agent.x, agent.y] += 1
        
        if self.full_obs:
            return self._get_full_obs()
        return self._get_obs()
    
    def step(self, actions):
        if not all(type(a) is int for a in actions):
            actions = [a.argmax() for a in actions]
        
        # ✅ 修复：update agents' position（修正坐标系定义）
        # Pass环境：agents从底部（低y）移动到顶部（高y）
        # 判断逻辑使用agent.y，所以UP/DOWN应该修改y坐标
        for idx, agent in enumerate(self.agents):
            action = actions[idx]
            
            x, y = agent.x, agent.y
            if action == 0: # UP (增加Y坐标，向目标移动)
                if y < self.grid_size - 1 and self.wall_map[x, y + 1] == 0:
                    agent.y += 1
            elif action == 1:   # DOWN (减少Y坐标，远离目标)
                if y > 0 and self.wall_map[x, y - 1] == 0:
                    agent.y -= 1
            elif action == 2:   # LEFT (减少X坐标)
                if x > 0 and self.wall_map[x - 1, y] == 0:
                    agent.x -= 1
            else:   # RIGHT (增加X坐标)
                if x < self.grid_size - 1 and self.wall_map[x + 1, y] == 0:
                    agent.x += 1
        
        # Check coin collection (支持Landmark对象和[x,y]列表两种格式)
        coin_reward = 0.0
        for coin in self.coins[:]:
            for agent in self.agents:
                # 兼容Landmark对象和[x,y]列表
                if isinstance(coin, Landmark):
                    dist = self._dist(agent, coin)
                else:
                    # coin是[x, y]列表
                    dist = np.sqrt((agent.x - coin[0]) ** 2 + (agent.y - coin[1]) ** 2)
                if dist <= 1.0:
                    coin_reward += self.coin_reward_scale
                    self.coins.remove(coin)
                    break
        self.total_coin_reward += coin_reward
        
        # update status of door (only real switches work)
        door_was_open = self.door.open
        open = False
        switch_activated = False
        for switch in self.real_switches:
            for agent in self.agents:
                if self._dist(agent, switch) <= self.activate_radius:
                    open = True
                    switch_activated = True
                    break
            if switch_activated:
                break
        
        # ✅ 统计switch_steps（激活开关的步数）
        if switch_activated:
            self.switch_steps += 1
        
        # ✅ 修复：Stage-1时门默认打开，应该保持打开状态（除非明确关闭）
        # Stage-2时门需要开关控制
        if self.curriculum_stage == 1:
            # Stage-1: 门保持打开
            self.door.open = True
            # ✅ 修复：打开整列墙
            self.wall_map[:, self.door.y] = 0
            if not door_was_open and self.t_open == -1:
                self.t_open = self.time
                self.episode_has_door_open = 1
        else:
            # Stage-2: 门由开关控制
            if open:
                self.door.open = True
                # ✅ 修复：打开整列墙
                self.wall_map[:, self.door.y] = 0
                if not door_was_open and self.t_open == -1:
                    self.t_open = self.time
                    self.episode_has_door_open = 1
            else:
                self.door.open = False
                # ✅ 修复：关闭整列墙
                self.wall_map[:, self.door.y] = 1
        
        # ✅ 统计door_steps（门打开的步数）- 无论门是否刚打开，只要门是开的就统计
        if self.door.open:
            self.door_steps += 1
        
        # ✅ Check if crossed door（使用统一的mid_y_threshold）
        if self.episode_has_crossed == 0:
            if all([agent.y > self.mid_y_threshold for agent in self.agents]):
                self.episode_has_crossed = 1
                if self.t_cross == -1:
                    self.t_cross = self.time
        
        # ✅ 更新max_y_reached统计
        if len(self.agents) > 0:
            self.max_y_reached_0 = max(self.max_y_reached_0, float(self.agents[0].y))
        if len(self.agents) > 1:
            self.max_y_reached_1 = max(self.max_y_reached_1, float(self.agents[1].y))
        
        # ✅ 更新time_first_reach_y_15统计（使用统一的mid_y_threshold）
        if len(self.agents) > 0 and self.agents[0].y >= self.mid_y_threshold and self.time_first_reach_y_15_0 < 0:
            self.time_first_reach_y_15_0 = self.time
        if len(self.agents) > 1 and self.agents[1].y >= self.mid_y_threshold and self.time_first_reach_y_15_1 < 0:
            self.time_first_reach_y_15_1 = self.time
        
        # update visit counts
        if self.joint_count:
            visit_indices = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_indices] += 1
        else:
            for idx, agent in enumerate(self.agents):
                self.visit_counts[idx, agent.x, agent.y] += 1
        
        self.time += 1
        reward, done = coin_reward, False
        
        # ✅ 修复：正确计算any_in_target和all_in_target（使用统一的target_y_threshold）
        any_in_target_now = any([agent.y > self.target_y_threshold for agent in self.agents])
        all_in_target_now = all([agent.y > self.target_y_threshold for agent in self.agents])
        
        # ✅ 修复：更新episode级别的标志位，并记录t_target
        # t_target应该在any或all首次触发时记录（根据需求选择）
        # 这里使用all_in_target作为触发条件（严格成功）
        if any_in_target_now and self.episode_has_any_target == 0:
            self.episode_has_any_target = 1
            # 如果需要any触发时也记录t_target，取消下面的注释
            # if self.t_target == -1:
            #     self.t_target = self.time
        
        if all_in_target_now and self.episode_has_all_target == 0:
            self.episode_has_all_target = 1
            # ✅ 修复：在首次all_in_target时记录t_target
            if self.t_target == -1:
                self.t_target = self.time
        
        # ✅ 修复：Target reward只在首次all_in_target时给予（使用flag防止重复）
        target_reward = 0.0
        if all_in_target_now and self.episode_has_all_target == 1 and self.t_target == self.time:
            # 只在刚刚首次到达时给奖励（t_target刚被设置为当前时间）
            target_reward = 100.0
            done = True
            
            # ✅ 调试输出：成功到达目标
            if self.debug_mode:
                print(f"[Step {self.time}] SUCCESS! All agents reached target (y > {self.target_y_threshold})")
                print(f"  Agent positions: {[(a.x, a.y) for a in self.agents]}")
                print(f"  Target reward: {target_reward}")
        
        self.total_target_reward += target_reward
        reward += target_reward
        
        if self.time >= self.max_timesteps:
            done = True
            if self.debug_mode and not all_in_target_now:
                print(f"[Step {self.time}] TIMEOUT! Episode ended without reaching target")
                print(f"  Agent positions: {[(a.x, a.y) for a in self.agents]}")
                print(f"  Max Y reached: agent0={self.max_y_reached_0}, agent1={self.max_y_reached_1}")
            if self.debug_mode and not all_in_target_now:
                print(f"[Step {self.time}] TIMEOUT! Episode ended without reaching target")
                print(f"  Agent positions: {[(a.x, a.y) for a in self.agents]}")
                print(f"  Max Y reached: agent0={self.max_y_reached_0}, agent1={self.max_y_reached_1}")
        
        # ✅ 计算min_dist_to_switch和min_dist_to_door_gap
        min_dist_to_switch = float('inf')
        if len(self.real_switches) > 0:
            for switch in self.real_switches:
                for agent in self.agents:
                    dist = self._dist(agent, switch)
                    min_dist_to_switch = min(min_dist_to_switch, dist)
        else:
            min_dist_to_switch = 0.0
        
        min_dist_to_door_gap = float('inf')
        for agent in self.agents:
            dist = self._dist(agent, self.door)
            min_dist_to_door_gap = min(min_dist_to_door_gap, dist)
        
        # ✅ 修复：Compute info with unified metrics（修正指标口径）
        info = {
            'episode_has_door_open': self.episode_has_door_open,
            'episode_has_crossed': self.episode_has_crossed,
            'crossed_door': self.episode_has_crossed,
            'episode_has_any_target': self.episode_has_any_target,
            # ✅ 修复：区分any_in_target和all_in_target
            'any_in_target': int(self.episode_has_any_target),
            'all_in_target': int(self.episode_has_all_target),
            # ✅ 修复：success定义为all_in_target（严格成功）或any_in_target（宽松成功）
            # 根据文档建议，这里使用all_in_target作为严格成功指标
            'success': int(self.episode_has_all_target),
            't_open': self.t_open,
            't_cross': self.t_cross,
            't_target': self.t_target,
            'total_coin': self.total_coin_reward,
            'total_target': self.total_target_reward,
            'coin_ratio': self.total_coin_reward / max(self.total_coin_reward + self.total_target_reward, 1e-6),
            # ✅ 添加缺失的统计字段
            'switch_steps': self.switch_steps,
            'door_steps': self.door_steps,
            'min_dist_to_switch': float(min_dist_to_switch),
            'min_dist_to_door_gap': float(min_dist_to_door_gap),
            'max_y_reached_0': self.max_y_reached_0,
            'max_y_reached_1': self.max_y_reached_1,
            'time_first_reach_y_15_0': self.time_first_reach_y_15_0 if self.time_first_reach_y_15_0 >= 0 else -1,
            'time_first_reach_y_15_1': self.time_first_reach_y_15_1 if self.time_first_reach_y_15_1 >= 0 else -1,
        }
            
        reward_n = [[reward]] * self.num_agents
        done_n = [done] * self.num_agents
        info_n = [info] * self.num_agents
        
        if self.full_obs:
            return self._get_full_obs(), reward_n, done_n, info_n
        return self._get_obs(), reward_n, done_n, info_n
        
    def _get_obs(self):
        obs_n = [np.array([self.agents[0].x, self.agents[0].y]) / self.grid_size,
                 np.array([self.agents[1].x, self.agents[1].y]) / self.grid_size]
        
        # Add observation noise for Stage-2
        if self.obs_noise_level > 0:
            for i in range(len(obs_n)):
                noise = self.np_random.standard_normal(obs_n[i].shape) * self.obs_noise_level * 0.1
                obs_n[i] = obs_n[i] + noise
                obs_n[i] = np.clip(obs_n[i], -1, 1)
        
        if self.door_in_obs:
            return [np.concatenate([obs_n[0], [self.door.open]]),
                    np.concatenate([obs_n[1], [self.door.open]])]
        return obs_n
    
    def _get_full_obs(self):
        obs_n = [np.array([self.agents[0].x, self.agents[0].y,
                           self.agents[1].x, self.agents[1].y]) / self.grid_size,
                 np.array([self.agents[1].x, self.agents[1].y,
                           self.agents[0].x, self.agents[0].y]) / self.grid_size]
        
        # Add observation noise for Stage-2
        if self.obs_noise_level > 0:
            for i in range(len(obs_n)):
                noise = self.np_random.standard_normal(obs_n[i].shape) * self.obs_noise_level * 0.1
                obs_n[i] = obs_n[i] + noise
                obs_n[i] = np.clip(obs_n[i], -1, 1)
        
        if self.door_in_obs:
            return [np.concatenate([obs_n[0], [self.door.open]]),
                    np.concatenate([obs_n[1], [self.door.open]])]
        return obs_n
    
    def _dist(self, e1, e2):
        return np.sqrt((e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2)    

    def get_visit_counts(self, agent_id=None):
        if agent_id is not None and not self.joint_count:
            return self.visit_counts[agent_id]
        return self.visit_counts
    
    def set_visit_counts(self, visit_counts, agent_id):
        if agent_id is not None and not self.joint_count:
            assert self.visit_counts[agent_id].shape == visit_counts.shape
            self.visit_counts[agent_id] = copy.deepcopy(visit_counts)
        else:
            assert self.visit_counts.shape == visit_counts.shape
            self.visit_counts = copy.deepcopy(visit_counts)
    
    def reset_visit_counts(self):
        self.visit_counts *= 0
        
    def visit_counts_decay(self, decay_coef):
        self.visit_counts *= decay_coef
