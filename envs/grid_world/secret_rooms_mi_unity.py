"""
MI-Unity Enhanced Secret Rooms Environment
==========================================
魔改重点:
1. 认知碎片化 (Epistemic Fragmentation):
   - 观测噪声: Switch位置返回多峰分布
   - Dummy Switch: 假开关迷惑智能体
   
2. 价值异质性 (Value Heterogeneity):
   - Coin (低价值): +0.1分, 80%数量, 快速刷新
   - Switch/Door (高价值): +100分, 20%数量, 关键任务
"""

import copy
import gym
import numpy as np
from gym.spaces import Box, Discrete
from gym.utils import seeding


class Entity(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Agent(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.movable = True
        self.coin_collected = 0  # 统计捡到的金币数


class Landmark(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.movable = False


class Switch(Landmark):
    """真开关"""
    def __init__(self, x, y, is_dummy=False):
        super().__init__(x, y)
        self.on = False
        self.is_dummy = is_dummy  # 是否是假开关
        
    def _dist(self, agent):
        return np.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
    
    def update(self, agents, activate_radius):
        if self.is_dummy:
            self.on = False  # 假开关永远无效
            return
        self.on = False
        for agent in agents:
            if self._dist(agent) <= activate_radius:
                self.on = True
                break


class DummySwitch(Switch):
    """假开关 - 植入认知碎片化"""
    def __init__(self, x, y):
        super().__init__(x, y, is_dummy=True)


class Coin(Landmark):
    """金币 - 植入价值异质性
    
    P0修复: 
    - update()返回本步是否被捡起的事件标志
    - reward只在picked=True时加一次，避免重复计数
    - 支持coin_respawn参数控制
    """
    def __init__(self, x, y):
        super().__init__(x, y)
        self.collected = False
        self.respawn_prob = 0.3  # 刷新概率（仅在respawn=True时生效）
        self.value = 0.1  # 低价值
    
    def _dist(self, agent):
        return np.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
    
    def update(self, agents, random_state, respawn=False):
        """
        检查是否被捡起，并可能重新刷新
        
        Returns:
            picked (bool): 本步是否被捡起（事件标志）
        """
        picked = False  # 本步是否被捡起的事件标志
        
        if not self.collected:
            # 尝试捡起
            for agent in agents:
                if self._dist(agent) <= 1.0:
                    self.collected = True
                    picked = True  # 标记为本步被捡起
                    agent.coin_collected += 1
                    break
        else:
            # 已被捡起，根据respawn参数决定是否刷新
            if respawn and random_state.random() < self.respawn_prob:
                self.collected = False
        
        return picked


class Door(Entity):
    def __init__(self, x, y, direction, switches_to_open):
        super().__init__(x, y)
        self.d = direction  # 0: vertical, 1: horizontal
        self.s = switches_to_open
        self.open = False
        # ✅ 步骤2：添加 latch 机制，一旦打开就保持打开
        self.ever_open = False  # 门是否曾经打开过（latch 标志）

    def update(self, switches):
        # ✅ 步骤2：实现 latch 逻辑：door_open = door_open_latched or switch_pressed
        # 检查开关是否被按下
        switch_pressed = False
        for indices in self.s:
            # 只检查真开关，忽略假开关
            valid_switches = [switches[i] for i in indices if not switches[i].is_dummy]
            if valid_switches and all([s.on for s in valid_switches]):
                switch_pressed = True
                break
        
        # Latch 逻辑：一旦打开过就保持打开，或者开关被按下
        if switch_pressed:
            self.ever_open = True  # 一旦打开，设置 latch
            self.open = True
        elif self.ever_open:
            # 已经打开过，保持打开（latch）
            self.open = True
        else:
            self.open = False


class SecretRoomsMIUnity(gym.Env):
    """
    MI-Unity增强版Secret Rooms
    
    参数:
    - obs_noise_level: 观测噪声强度 (0.0-0.5), 用于模拟认知碎片化
    - dummy_switch_ratio: 假开关比例 (0.0-1.0)
    - coin_density: 金币密度 (每个房间的金币数量)
    - coin_respawn: 金币是否重新刷新
    """
    def __init__(self, 
                 map_ind=20, 
                 max_timesteps=200,          # P0: 调整为200，控制episode长度
                 door_in_obs=False, 
                 full_obs=False, 
                 joint_count=False, 
                 activate_radius=None, 
                 grid_size=25,
                 # MI-Unity专属参数
                 obs_noise_level=0.0,        # 观测噪声 (认知碎片化)
                 dummy_switch_ratio=0.0,     # P1: 改为0.0，由固定策略控制
                 coin_density=50,            # P0: 增加到50个，确保N_coin * r_coin < R_target
                 coin_respawn=False,         # P0: 默认禁止刷新
                 # ✅ Curriculum 参数
                 curriculum_stage=2,         # 1=门默认开, 2=正常（需要开关）, 3=Stage-2+随机化
                 coin_reward_scale=1.0,      # ✅ Stage-1 可以设为 0 或很小，避免策略被 coin 带偏
                 # ✅ Stage-2/3 复杂场景参数
                 episode_limit=None,         # None 时使用 max_timesteps
                 random_spawn=False,         # 随机化起点
                 random_switch=False,        # 随机化开关位置
                 random_target=False,        # 随机化目标区
                 door_open_duration=None,     # 门限时打开时长（None=永久）
                 **kwargs):
        
        self.num_agents = map_ind // 10
        self.map_ind = map_ind
        self.max_timesteps = max_timesteps
        self.joint_count = joint_count
        self.grid_size = grid_size
        self.door_in_obs = door_in_obs
        self.full_obs = full_obs
        
        # MI-Unity参数
        self.obs_noise_level = obs_noise_level
        self.dummy_switch_ratio = dummy_switch_ratio
        self.coin_density = coin_density
        self.coin_respawn = coin_respawn
        
        # ✅ Curriculum 参数
        self.curriculum_stage = curriculum_stage  # 1=门默认开, 2=正常, 3=Stage-2+随机化
        self.coin_reward_scale = coin_reward_scale  # ✅ Coin 奖励缩放（Stage-1 建议设为 0）
        
        # ✅ Stage-2/3 复杂场景参数
        self.episode_limit = episode_limit if episode_limit is not None else max_timesteps
        self.random_spawn = random_spawn
        self.random_switch = random_switch
        self.random_target = random_target
        self.door_open_duration = door_open_duration  # None=永久打开
        
        # 如果设置了 episode_limit，更新 max_timesteps
        if episode_limit is not None:
            self.max_timesteps = episode_limit
        
        # P0: 上界断言 - 确保Lemma 1成立
        coin_value = 0.1
        target_value = 100.0
        max_coin_value = self.coin_density * coin_value
        assert max_coin_value < target_value, \
            f"Lemma 1 violation: N_coin * r_coin ({max_coin_value}) must < R_target ({target_value})"
        
        # 补丁2: 延迟初始化coins，等seed()设置后再生成
        self.init_coins = None  # 延迟初始化
        
        # 必须先初始化entity和wall，因为_init_coins()需要用到init_wall_map
        self._init_entity(map_ind, grid_size)
        self._init_wall(grid_size)
        
        # 初始化随机数生成器（会调用_init_coins()）
        self.seed()
        
        # 注意：_init_coins()已移到seed()中，确保使用正确的随机种子
        self._init_dummy_switches()  # 补丁1: 固定策略初始化假开关
        self._init_space()
        
        self.time = 0
        self.agents, self.wall_map, self.doors = None, None, None
        self.coins = None
        self.door_radius = 1
        self.activate_radius = activate_radius or 1.5 * self.door_radius
        
        # 统计信息
        self.total_coin_reward = 0.0
        self.total_target_reward = 0.0
        
        # 中间里程碑奖励追踪（用于提高学习信号密度）
        self.milestone_rewards_enabled = True  # 可通过参数控制
        # ✅ 添加奖励缩放参数，避免策略被带偏
        self.milestone_reward_scale = kwargs.get('milestone_reward_scale', 1.0)  # 默认不缩放
        self.agent_crossed_door = [False] * self.num_agents  # 追踪是否穿过Door0
        self.all_agents_right = False  # 追踪是否所有agent都在右侧
        
        if self.joint_count:
            self.visit_counts = np.zeros(self.num_agents * [self.grid_size, self.grid_size])
        else:
            self.visit_counts = np.zeros((self.num_agents, self.grid_size, self.grid_size))
        
        self.reset()
    
    def _init_entity(self, map_ind, grid_size):
        """初始化智能体、门、真开关"""
        ot_grid_size = grid_size // 3
        tt_grid_size = grid_size * 2 // 3
        
        if map_ind // 10 == 2:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_doors = [Door(ot_grid_size // 2, grid_size // 2, 0, [[0], [1]]),
                               Door((ot_grid_size + tt_grid_size) // 2, grid_size // 2, 0, [[0], [2]]),
                               Door((tt_grid_size + grid_size) // 2, grid_size // 2, 0, [[0], [3]])]
            
            # 真开关位置
            self.real_switches = [
                Switch(int(grid_size * 0.8), int(grid_size * 0.2)),  # main switch
                Switch(self.init_doors[0].x, int(self.grid_size * 0.8)),
                Switch(self.init_doors[1].x, int(self.grid_size * 0.8)),
                Switch(self.init_doors[2].x, int(self.grid_size * 0.8))
            ]
            
            if map_ind % 10 == 0:
                self.target_room = 1
            elif map_ind % 10 == 1:
                self.target_room = 2
            elif map_ind % 10 == 2:
                self.target_room = 3
            else:
                raise NotImplementedError(f"Not support map_ind {map_ind}.")
                
        elif map_ind // 10 == 3:
            self.init_agents = [Agent(grid_size // 10, grid_size // 10),
                                Agent(grid_size // 10 + 2, grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10 + 2)]
            self.init_doors = [Door(ot_grid_size // 2, grid_size // 2, 0, [[0], [1]]),
                               Door((ot_grid_size + tt_grid_size) // 2, grid_size // 2, 0, [[0], [2]]),
                               Door((tt_grid_size + grid_size) // 2, grid_size // 2, 0, [[0], [3]])]
            
            self.real_switches = [
                Switch(int(grid_size * 0.8), int(grid_size * 0.2)),
                Switch(self.init_doors[0].x, int(self.grid_size * 0.8)),
                Switch(self.init_doors[1].x, int(self.grid_size * 0.8)),
                Switch(self.init_doors[2].x, int(self.grid_size * 0.8))
            ]
            
            if map_ind % 10 == 0:
                self.init_doors[0].s = [[0], [3]]
                self.init_doors[1].s = [[0], [3]]
                self.init_doors[2].s = [[1, 2], [3]]
                self.target_room = 3
            elif map_ind % 10 in [1, 2, 3, 4]:
                # 简化为统一配置
                self.init_doors[0].s = [[0]]
                self.init_doors[1].s = [[0]]
                self.init_doors[2].s = [[1, 2]]
                self.target_room = 3
            else:
                raise NotImplementedError(f"Not support map_ind {map_ind}.")
        else:
            raise NotImplementedError(f"Not support map_ind {map_ind}.")
    
    def _init_dummy_switches(self):
        """
        补丁1修复: 离散语义的dummy_ratio
        
        严格定义（避免语义混乱）：
        - ratio = 0.0: 0个dummy（纯净环境）
        - ratio = 0.5: 1个dummy（仅主开关镜像，论文主实验配置）
        - ratio = 1.0: 全部dummy（所有4个开关都有镜像）
        
        任何其他值会报错，强制明确配置
        """
        self.dummy_switches = []
        
        # 离散判断，不用>=这种阈值逻辑
        if self.dummy_switch_ratio == 0.0:
            # 不加任何dummy
            pass
        
        elif self.dummy_switch_ratio == 0.5:
            # 只加主开关（index=0）的dummy
            if len(self.real_switches) > 0:
                main_switch = self.real_switches[0]
                mirror_x = self.grid_size - 1 - main_switch.x
                mirror_y = self.grid_size - 1 - main_switch.y
                
                if self._is_valid_dummy_position(mirror_x, mirror_y):
                    self.dummy_switches.append(DummySwitch(mirror_x, mirror_y))
        
        elif self.dummy_switch_ratio == 1.0:
            # 给所有真开关都加dummy
            for real_switch in self.real_switches:
                mirror_x = self.grid_size - 1 - real_switch.x
                mirror_y = self.grid_size - 1 - real_switch.y
                
                if self._is_valid_dummy_position(mirror_x, mirror_y):
                    self.dummy_switches.append(DummySwitch(mirror_x, mirror_y))
        
        else:
            # 不支持其他值，强制报错
            raise ValueError(
                f"dummy_switch_ratio must be exactly 0.0, 0.5, or 1.0 (got {self.dummy_switch_ratio}). "
                f"Use 0.0=no dummy, 0.5=main switch only, 1.0=all switches."
            )
        
        # 合并真假开关列表
        self.switches = self.real_switches + self.dummy_switches
    
    def _is_valid_dummy_position(self, x, y):
        """
        检查dummy switch位置是否合法
        - 不在墙上
        - 不与真开关重叠
        - 不与门重叠
        """
        # 检查边界
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        # 检查是否在墙上（需要先初始化墙）
        # 这里简化处理：只检查不与真开关重叠
        for real_switch in self.real_switches:
            if real_switch.x == x and real_switch.y == y:
                return False
        
        # 检查不与门重叠
        for door in self.init_doors:
            if door.x == x and door.y == y:
                return False
        
        return True
    
    def _is_valid_position(self, x, y):
        """检查位置是否合法（不在墙上，不在门位置）"""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        if self.init_wall_map[x, y] == 1:
            return False
        # 检查不在门位置
        for door in self.init_doors:
            if door.d == 0:  # vertical
                if abs(x - door.x) <= self.door_radius and y == door.y:
                    return False
            else:  # horizontal
                if x == door.x and abs(y - door.y) <= self.door_radius:
                    return False
        return True
    
    def _sample_spawn(self):
        """
        随机化起点位置（Stage-3）
        ✅ 硬约束：只能在下半区（门下方，y <= grid_size//2），否则一出生就在上半区就不需要门
        """
        mid_y = self.grid_size // 2
        for agent in self.agents:
            attempts = 0
            while attempts < 100:
                x = self.random.integers(0, self.grid_size // 3)  # 在左侧区域
                y = self.random.integers(0, mid_y)  # ✅ 硬约束：只能在下半区（y < mid_y）
                if self._is_valid_position(x, y):
                    # ✅ 检查不与已有 agent 重叠
                    overlap = False
                    for other_agent in self.agents:
                        if other_agent != agent and other_agent.x == x and other_agent.y == y:
                            overlap = True
                            break
                    if not overlap:
                        agent.x = x
                        agent.y = y
                        break
                attempts += 1
    
    def _sample_switch(self):
        """
        随机化开关位置（Stage-3）
        ✅ 硬约束：只能在下半区（门下方，y <= grid_size//2），否则开关在门后面变成死局
        """
        mid_y = self.grid_size // 2
        # 为每个真开关随机选择新位置（保持在左侧区域的下半区）
        for switch in self.real_switches:
            attempts = 0
            while attempts < 100:
                x = self.random.integers(0, self.grid_size // 3)  # 在左侧区域
                y = self.random.integers(0, mid_y)  # ✅ 硬约束：只能在下半区（y < mid_y）
                if self._is_valid_position(x, y):
                    # ✅ 检查不与其他开关重叠
                    overlap = False
                    for other_switch in self.real_switches:
                        if other_switch != switch and other_switch.x == x and other_switch.y == y:
                            overlap = True
                            break
                    # ✅ 检查不与 agent 重叠
                    if not overlap:
                        for agent in self.agents:
                            if agent.x == x and agent.y == y:
                                overlap = True
                                break
                    if not overlap:
                        switch.x = x
                        switch.y = y
                        break
                attempts += 1
    
    def _sample_target(self):
        """
        随机化目标区（Stage-3）
        ✅ 硬约束：只能在上半区（门上方，y > grid_size//2），否则不需要穿门
        """
        # 在右侧三个房间中随机选择一个作为目标（都在上半区）
        if self.map_ind // 10 == 2:  # 2 agents
            self.target_room = self.random.choice([1, 2, 3])
        elif self.map_ind // 10 == 3:  # 3 agents
            self.target_room = 3  # 对于3 agents，通常固定为room 3
        # ✅ 注意：target_room 1/2/3 都在上半区（y > grid_size//2），符合约束
    
    def _init_coins(self):
        """初始化金币 - 价值异质性核心"""
        self.init_coins = []
        if self.coin_density > 0:
            # 在每个可行走区域随机撒金币
            for _ in range(self.coin_density):
                attempts = 0
                while attempts < 100:  # 防止无限循环
                    x = self.random.integers(0, self.grid_size)
                    y = self.random.integers(0, self.grid_size)
                    
                    # 确保不在墙上，不在开关位置
                    if self.init_wall_map[x, y] == 0:
                        # 检查是否与现有开关重叠
                        too_close = False
                        for switch in self.real_switches:
                            if abs(switch.x - x) <= 1 and abs(switch.y - y) <= 1:
                                too_close = True
                                break
                        if not too_close:
                            self.init_coins.append(Coin(x, y))
                            break
                    attempts += 1
    
    def _init_wall(self, grid_size):
        ot_grid_size = grid_size // 3
        tt_grid_size = grid_size * 2 // 3
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, grid_size // 2] = 1
        self.init_wall_map[ot_grid_size, grid_size // 2:] = 1
        self.init_wall_map[tt_grid_size, grid_size // 2:] = 1
    
    def _init_space(self):
        """
        P2修复: Observation space保持与原环境一致
        
        基础维度: 2 (归一化位置)
        可选维度: 
          - door_in_obs=True: +len(switches) (解歧状态) + len(doors) (开关状态)
          - door_in_obs=False: 仅位置
        
        不包含coin/dummy的统计特征
        """
        # 基础观测: agent position
        obs_dim = 2
        
        # P3: 如果door_in_obs=True，添加switch解歧特征
        if self.door_in_obs:
            obs_dim += len(self.switches)  # switch状态 (Unknown/True/Dummy)
            obs_dim += len(self.init_doors)  # door开关状态
        
        L = self.num_agents * obs_dim if self.full_obs else obs_dim
        
        self.observation_space = [
            Box(low=-1, high=1, shape=(L,), dtype=np.float32) for _ in range(self.num_agents)]
        self.share_observation_space = [
            Box(low=-1, high=1, shape=(L * self.num_agents,), dtype=np.float32) 
            for _ in range(self.num_agents)]
        # ✅ 步骤5：添加 stay/no-op 动作，让 agent 可以在目标区等待队友
        self.action_space = [Discrete(5) for _ in range(self.num_agents)]  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
    
    def _in_target_room(self, agent, target_room):
        if target_room == 1:
            return (agent.y > self.grid_size // 2) and (agent.x < self.grid_size // 3)
        if target_room == 2:
            return (agent.y > self.grid_size // 2) and (agent.x > self.grid_size // 3) and \
                (agent.x < self.grid_size * 2 // 3)
        if target_room == 3:
            return (agent.y > self.grid_size // 2) and (agent.x > self.grid_size * 2 // 3)
        return False
    
    @staticmethod
    def ind2ndoor(ind):
        """
        P4: 添加此方法以接入MACE训练管线
        与原SecretRooms保持一致
        """
        if ind // 10 == 2:
            return 3
        if ind // 10 == 3:
            if ind % 10 in [0, 1]:
                return 3
            if ind % 10 in [2, 3, 4]:
                return 5
        raise ValueError(f"Unknown map index {ind}.")
    
    def seed(self, seed=None):
        """
        补丁2修复: 在seed设置后生成coins，确保可复现性
        
        工作流程：
        1. 设置随机种子 self.random
        2. 生成coins（使用已设置的self.random）
        3. 这样保证相同seed → 相同coins布局
        """
        self.random, seed = seeding.np_random(seed)
        
        # 补丁2: 在seed设置后立即生成coins（如果coin_density > 0）
        if self.coin_density > 0:
            self._init_coins()
        
        return seed
    
    def reset(self):
        """
        补丁2修复: 确保coins已初始化
        
        正常情况下seed()已经生成了init_coins，但为了安全：
        如果init_coins仍为None（例如直接reset未seed），则先初始化
        """
        # 补丁2: 安全检查 - 如果coins未初始化，先初始化
        if self.init_coins is None and self.coin_density > 0:
            self._init_coins()
        
        self.agents = copy.deepcopy(self.init_agents)
        # ✅ 关键修复：使用原始墙模板（deepcopy），确保每次 reset 都从干净状态开始
        # 这样即使上一局开门清过墙，reset 后也会完全恢复
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.doors = copy.deepcopy(self.init_doors)
        self.coins = copy.deepcopy(self.init_coins) if self.init_coins is not None else []
        self.time = 0
        self.total_coin_reward = 0.0
        self.total_target_reward = 0.0
        
        # ✅ Curriculum Stage-1: 门默认打开
        # ✅ 关键修复：必须复用 step() 中的清墙逻辑，确保墙被正确清除
        if self.curriculum_stage == 1:
            for door in self.doors:
                door.open = True
                door.ever_open = True  # 设置为已打开过
                # ✅ 复用 step() 中的清墙逻辑（确保一致性）
                if door.open:
                    if door.d == 0:  # vertical
                        self.wall_map[door.x - self.door_radius:
                                      door.x + self.door_radius + 1, door.y] = 0
                    else:  # horizontal
                        self.wall_map[door.x, door.y - self.door_radius:
                                      door.y + self.door_radius + 1] = 0
            # ✅ 设置 t_open = 0（表示门在初始时就打开）
            self.door0_ever_open = True
            self.door_open_steps = 0  # 初始时门已打开，步数为0
        else:
            # ✅ Stage-2/3: 门初始关
            # 注意：wall_map 已经从 init_wall_map deepcopy，所以门口墙已经存在
            # 不需要手动恢复墙，因为 reset 时已经用原始模板恢复了
            for door in self.doors:
                door.open = False
                door.ever_open = False  # 重置为未打开过
            # ✅ Stage-2: 门初始关
            self.door0_ever_open = False
            self.door_open_steps = -1  # -1 表示未开门
        
        # ✅ Stage-3: 随机化（spawn/switch/target）
        if self.curriculum_stage >= 3:
            if self.random_spawn:
                self._sample_spawn()
            if self.random_switch:
                self._sample_switch()
            if self.random_target:
                self._sample_target()
        
        # 重置中间里程碑追踪
        self.agent_crossed_door = [False] * self.num_agents
        self.all_agents_right = False
        
        # 进度信号追踪（用于诊断和HPB）
        self.hold_steps_switch0 = 0  # main switch连续保持on的步数
        # ✅ 注意：door0_ever_open 和 door_open_steps 已在上面根据 stage 设置
        # ✅ 添加开关追踪（用于日志）
        self.episode_has_switch = False  # 是否踩到过开关
        self.t_switch = -1  # 首次踩到开关的时间
        self.any_right_ever = False  # 是否有agent到过右侧
        self.any_target_ever = False  # 是否有agent进过target room
        self.all_right_ever = False  # 是否全员到过右侧
        self.progress_events = []  # 本episode发生的进度事件列表
        
        # ✅ 重置episode级别统计字段
        self.t_cross = -1  # 第一次通过门的时间
        self.t_target = -1  # 第一次到达目标的时间
        self.switch_steps = 0  # 激活开关的累计步数
        self.door_steps = 0  # 门打开的累计步数
        self.time_first_reach_y_15_0 = -1  # agent 0第一次到达y=15的时间
        self.time_first_reach_y_15_1 = -1  # agent 1第一次到达y=15的时间
        
        # 重置统计
        for agent in self.agents:
            agent.coin_collected = 0
        
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
        
        # 移动智能体
        for idx, agent in enumerate(self.agents):
            action = actions[idx]
            x, y = agent.x, agent.y
            
            if action == 0:  # UP
                if x > 0 and self.wall_map[x - 1, y] == 0:
                    agent.x -= 1
            elif action == 1:  # DOWN
                if x < self.grid_size - 1 and self.wall_map[x + 1, y] == 0:
                    agent.x += 1
            elif action == 2:  # LEFT
                if y > 0 and self.wall_map[x, y - 1] == 0:
                    agent.y -= 1
            elif action == 3:  # RIGHT
                if y < self.grid_size - 1 and self.wall_map[x, y + 1] == 0:
                    agent.y += 1
            # ✅ 步骤5：action == 4 是 STAY/no-op，agent 不移动
            # else: action == 4 (STAY) - 不执行任何移动
        
        # 更新开关状态
        self.wall_map = copy.deepcopy(self.init_wall_map)
        for switch in self.switches:
            switch.update(self.agents, self.activate_radius)
        
        # ✅ Stage-2: 检测是否踩到开关（用于日志）
        any_switch_on = any([s.on for s in self.real_switches])
        if any_switch_on and not self.episode_has_switch:
            self.episode_has_switch = True
            self.t_switch = self.time
        
        # 更新门状态（Stage-1 或已开门的门）
        door0_was_open = len(self.doors) > 0 and self.doors[0].open
        for door in self.doors:
            door.update(self.switches)
            if door.open:
                if door.d == 0:
                    self.wall_map[door.x - self.door_radius:
                                  door.x + self.door_radius + 1, door.y] = 0
                else:
                    self.wall_map[door.x, door.y - self.door_radius:
                                  door.y + self.door_radius + 1] = 0
        
        # ✅ Stage-2: 如果门还没开，检测是否应该开门（踩开关触发）
        # ✅ 关键：确保在同一步生效，t_open == t_switch
        door0_now_open = len(self.doors) > 0 and self.doors[0].open
        if not door0_now_open and self.curriculum_stage >= 2:
            # 检查是否有任何真开关被激活
            if any_switch_on:
                # 触发开门：找到第一个门（door0），设置为打开
                if len(self.doors) > 0:
                    door0 = self.doors[0]
                    door0.open = True
                    door0.ever_open = True
                    # 清墙
                    if door0.d == 0:  # vertical
                        self.wall_map[door0.x - self.door_radius:
                                      door0.x + self.door_radius + 1, door0.y] = 0
                    else:  # horizontal
                        self.wall_map[door0.x, door0.y - self.door_radius:
                                      door0.y + self.door_radius + 1] = 0
                    door0_now_open = True
                    # ✅ 关键：记录开门时间（与 t_switch 同一步）
                    if self.door_open_steps < 0:
                        self.door_open_steps = self.time  # 应该等于 t_switch
        
        # ✅ 限时开门逻辑（Stage-3 可选）
        # ✅ 关键修复：处理 agent 在门格上的情况，避免 agent 被墙吞进去
        if self.door_open_duration is not None and door0_now_open and self.door_open_steps >= 0:
            if self.time - self.door_open_steps > self.door_open_duration:
                # 门超时，需要恢复墙
                if len(self.doors) > 0:
                    door0 = self.doors[0]
                    # ✅ 安全检查：检查是否有 agent 在门格上
                    door_cells = []
                    if door0.d == 0:  # vertical
                        for x in range(max(0, door0.x - self.door_radius), 
                                      min(self.grid_size, door0.x + self.door_radius + 1)):
                            door_cells.append((x, door0.y))
                    else:  # horizontal
                        for y in range(max(0, door0.y - self.door_radius),
                                      min(self.grid_size, door0.y + self.door_radius + 1)):
                            door_cells.append((door0.x, y))
                    
                    # 检查是否有 agent 在门格上
                    agent_in_door = False
                    for agent in self.agents:
                        if (agent.x, agent.y) in door_cells:
                            agent_in_door = True
                            break
                    
                    if agent_in_door:
                        # ✅ 策略1：延迟关门（再等 1 步）
                        # 不关门，等待下一步再检查
                        pass
                    else:
                        # ✅ 安全：没有 agent 在门格上，可以关门
                        door0.open = False
                        door0_now_open = False
                        # 恢复墙
                        if door0.d == 0:  # vertical
                            self.wall_map[door0.x - self.door_radius:
                                          door0.x + self.door_radius + 1, door0.y] = 1
                        else:  # horizontal
                            self.wall_map[door0.x, door0.y - self.door_radius:
                                          door0.y + self.door_radius + 1] = 1
        
        # 追踪进度事件
        switch0_on = len(self.real_switches) > 0 and self.real_switches[0].on
        
        # 检测事件：门从关到开
        if door0_now_open and not self.door0_ever_open:
            self.progress_events.append("DOOR_OPEN")
            self.door0_ever_open = True
            # ✅ 记录开门时间（如果还没记录，应该已经在上面 Stage-2 逻辑中记录了）
            if self.door_open_steps < 0:
                self.door_open_steps = self.time
        
        # ✅ 注意：door_open_steps 表示首次开门的时间（不是累计步数）
        # 在 Stage-2 中，应该等于 t_switch（同一步生效）
        
        # 追踪main switch持续激活步数
        if switch0_on:
            self.hold_steps_switch0 += 1
        else:
            self.hold_steps_switch0 = 0
        
        # P0修复: 更新金币状态 - 只在被捡起的那一刻给reward
        coin_reward = 0.0
        for coin in self.coins:
            picked = coin.update(self.agents, self.random, respawn=self.coin_respawn)
            if picked:  # 只在本步被捡起时加reward，避免重复计数
                # ✅ 应用 coin_reward_scale（Stage-1 可以设为 0）
                coin_reward += coin.value * self.coin_reward_scale
        
        # 更新访问计数
        if self.joint_count:
            visit_indices = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_indices] += 1
        else:
            for idx, agent in enumerate(self.agents):
                self.visit_counts[idx, agent.x, agent.y] += 1
        
        self.time += 1
        
        # 计算奖励
        target_reward = 0.0
        done = False
        info = {}
        
        # ✅ 步骤0：修复统计口径陷阱 - 在 all_in_target=True 时立即终止
        all_in_target = all([self._in_target_room(agent, self.target_room) for agent in self.agents])
        any_in_target = any([self._in_target_room(agent, self.target_room) for agent in self.agents])
        
        # ✅ 修改A：拆分success指标
        # Coord-Success: 协作链条成立（踩开关→门开→有人到目标）
        coord_success = bool(self.door0_ever_open and any_in_target)
        # All-Success: 严格指标（所有人到目标）
        all_success = bool(all_in_target)
        
        if all_in_target:
            target_reward = 100.0  # 高价值任务
            done = True
            info["success"] = True  # 保留原有success（All-Success）
            info["coord_success"] = True  # ✅ 新增Coord-Success
            info["all_success"] = True  # ✅ 新增All-Success
            info["terminated"] = True  # 明确标记为正常终止（非时间限制）
        elif coord_success:
            # ✅ Coord-Success达成但All-Success未达成
            info["coord_success"] = True
            info["all_success"] = False
        else:
            info["coord_success"] = False
            info["all_success"] = False
        
        if self.time >= self.max_timesteps:
            done = True
            info["TimeLimit.truncated"] = True
        
        # ==================== 中间里程碑奖励（提高学习信号密度） ====================
        milestone_reward = 0.0
        mid_y = self.grid_size // 2
        
        # ✅ 修复：即使 done=True，也要更新 agent_crossed_door（用于统计和日志）
        # 这确保 crossed_door 事件能被正确记录，即使 episode 已经结束
        if self.milestone_rewards_enabled:
            # r_cross = +0.5: 任一agent第一次穿过Door0到右侧
            # ✅ 重要：这个更新必须在 done 检查之前，确保即使 done=True 也能记录
            for idx, agent in enumerate(self.agents):
                if not self.agent_crossed_door[idx] and agent.y > mid_y:
                    if not done:  # 只有在未结束时才给奖励
                        milestone_reward += 0.5 * self.milestone_reward_scale
                    # ✅ 无论 done 状态如何，都要更新 agent_crossed_door（用于统计）
                    self.agent_crossed_door[idx] = True
                    info[f"agent_{idx}_crossed_door"] = True
        
        # ✅ 其他里程碑奖励只在未结束时计算
        if self.milestone_rewards_enabled and not done:
            # r_hold = +0.02: Door0 open 且至少一个agent在main switch上
            # ✅ 建议：这个奖励很小（0.02），但如果使用 potential shaping，可以设为 0
            if len(self.doors) > 0 and self.doors[0].open:
                main_switch = self.real_switches[0]
                for agent in self.agents:
                    dist_to_main = ((agent.x - main_switch.x)**2 + (agent.y - main_switch.y)**2)**0.5
                    if dist_to_main <= self.activate_radius:
                        milestone_reward += 0.02 * self.milestone_reward_scale
                        break  # 只要有一个agent守门就给奖励
            
            # r_all_right = +1.0: 两个agent都在门右侧（只给一次）
            if not self.all_agents_right:
                if all(agent.y > mid_y for agent in self.agents):
                    milestone_reward += 1.0 * self.milestone_reward_scale
                    self.all_agents_right = True
                    info["all_agents_right"] = True
            
            # ✅ 步骤4：添加文档要求的协作奖励 shaping
            # r_any = +1：任意 agent 进入目标区（any_target=True，只给一次）
            any_in_target_now = any([self._in_target_room(agent, self.target_room) for agent in self.agents])
            if any_in_target_now and not self.any_target_ever:
                milestone_reward += 1.0 * self.milestone_reward_scale
                self.any_target_ever = True
                info["first_any_target"] = True
            
            # r_hold = +0.1：如果 agent 在目标区内则每步给一点点（鼓励等人）
            # ✅ 注意：如果使用 potential shaping，建议将这个设为 0 或很小
            for agent in self.agents:
                if self._in_target_room(agent, self.target_room):
                    milestone_reward += 0.1 * self.milestone_reward_scale
                    break  # 只要有一个agent在目标区就给奖励
        
        # 统计milestone奖励
        info["milestone_reward"] = milestone_reward
        
        # 总奖励 = 金币 + 目标 + 里程碑
        total_reward = coin_reward + target_reward + milestone_reward
        self.total_coin_reward += coin_reward
        self.total_target_reward += target_reward
        
        # 统计信息
        info["coin_reward"] = coin_reward
        info["target_reward"] = target_reward
        info["total_coin_reward"] = self.total_coin_reward
        info["total_target_reward"] = self.total_target_reward
        info["coin_ratio"] = self.total_coin_reward / max(self.total_coin_reward + self.total_target_reward, 1e-6)
        
        # ==================== 进度事件信息（用于HPB/QAVER和诊断） ====================
        # 按照文档要求：添加6个关键进度信号
        
        # 1. door0_open: Door0当前是否open
        info["door0_open"] = door0_now_open
        
        # 2. door_open_steps: 首次开门的时间（-1 表示未开门）
        info["door_open_steps"] = self.door_open_steps
        
        # 3. switch0_on: main switch当前是否on
        info["switch0_on"] = switch0_on
        
        # 4. hold_steps_switch0: 本episode内main switch连续保持on的步数
        info["hold_steps_switch0"] = self.hold_steps_switch0
        
        # 5. agent_on_right: 每个agent是否在门右侧区域
        mid_y = self.grid_size // 2
        info["agent_on_right"] = [agent.y > mid_y for agent in self.agents]
        
        # 6. agent_in_target: 每个agent是否在target room
        info["agent_in_target"] = [self._in_target_room(a, self.target_room) for a in self.agents]
        
        # 7. any_in_target: 是否有任一agent到过target（文档要求）
        info["any_in_target"] = int(self.any_target_ever)
        
        # 7.5. all_in_target: 是否所有agent都在target（文档要求 - 修复缺失）
        # ✅ 修复：添加 all_in_target 写入（当前步的状态）
        info["all_in_target"] = int(all_in_target)
        
        # 7.6. total_target: 目标奖励累计值（修复字段名）
        info["total_target"] = float(self.total_target_reward)
        
        # 8. all_on_right: 是否全员都到过门右侧（文档要求）
        info["all_on_right"] = self.all_right_ever
        
        # 9. ✅ Stage-2 新增：开关相关日志字段
        info["episode_has_switch"] = self.episode_has_switch
        info["t_switch"] = self.t_switch
        
        # 10. ✅ 添加episode级别统计字段（用于train_log.csv）
        info["episode_has_door_open"] = int(self.door0_ever_open)
        info["episode_has_crossed"] = int(any(self.agent_crossed_door))
        info["episode_has_any_target"] = int(self.any_target_ever)
        info["crossed_door"] = int(any(self.agent_crossed_door))
        
        # 11. ✅ 添加时间戳字段
        info["t_open"] = self.door_open_steps if self.door_open_steps >= 0 else -1
        # t_cross: 第一次通过门的时间（需要追踪）
        if not hasattr(self, 't_cross'):
            self.t_cross = -1
        if any(self.agent_crossed_door) and self.t_cross < 0:
            self.t_cross = self.time
        info["t_cross"] = self.t_cross if self.t_cross >= 0 else -1
        # t_target: 第一次到达目标的时间（需要追踪）
        if not hasattr(self, 't_target'):
            self.t_target = -1
        if self.any_target_ever and self.t_target < 0:
            self.t_target = self.time
        info["t_target"] = self.t_target if self.t_target >= 0 else -1
        
        # 12. ✅ 添加其他统计字段
        # switch_steps: 激活开关的步数（累计）
        if not hasattr(self, 'switch_steps'):
            self.switch_steps = 0
        if switch0_on:
            self.switch_steps += 1
        info["switch_steps"] = self.switch_steps
        # door_steps: 门打开的步数（累计）
        if not hasattr(self, 'door_steps'):
            self.door_steps = 0
        if door0_now_open:
            self.door_steps += 1
        info["door_steps"] = self.door_steps
        
        # 13. ✅ 添加距离统计字段
        if len(self.real_switches) > 0:
            main_switch = self.real_switches[0]
            min_dist_to_switch = min([np.sqrt((agent.x - main_switch.x)**2 + (agent.y - main_switch.y)**2) 
                                     for agent in self.agents])
            info["min_dist_to_switch"] = float(min_dist_to_switch)
        else:
            info["min_dist_to_switch"] = 0.0
        
        if len(self.doors) > 0:
            door0 = self.doors[0]
            min_dist_to_door_gap = min([np.sqrt((agent.x - door0.x)**2 + (agent.y - door0.y)**2) 
                                       for agent in self.agents])
            info["min_dist_to_door_gap"] = float(min_dist_to_door_gap)
        else:
            info["min_dist_to_door_gap"] = 0.0
        
        # 14. ✅ 添加y坐标统计字段
        mid_y = self.grid_size // 2
        max_y_reached_0 = max(0.0, float(self.agents[0].y)) if len(self.agents) > 0 else 0.0
        max_y_reached_1 = max(0.0, float(self.agents[1].y)) if len(self.agents) > 1 else 0.0
        info["max_y_reached_0"] = max_y_reached_0
        info["max_y_reached_1"] = max_y_reached_1
        
        # time_first_reach_y_15: 第一次到达y=15的时间
        if not hasattr(self, 'time_first_reach_y_15_0'):
            self.time_first_reach_y_15_0 = -1
            self.time_first_reach_y_15_1 = -1
        if len(self.agents) > 0 and self.agents[0].y >= 15 and self.time_first_reach_y_15_0 < 0:
            self.time_first_reach_y_15_0 = self.time
        if len(self.agents) > 1 and self.agents[1].y >= 15 and self.time_first_reach_y_15_1 < 0:
            self.time_first_reach_y_15_1 = self.time
        info["time_first_reach_y_15_0"] = self.time_first_reach_y_15_0 if self.time_first_reach_y_15_0 >= 0 else -1
        info["time_first_reach_y_15_1"] = self.time_first_reach_y_15_1 if self.time_first_reach_y_15_1 >= 0 else -1
        
        # 检测进度事件
        any_right_now = any(info["agent_on_right"])
        any_target_now = any(info["agent_in_target"])
        all_right_now = all(info["agent_on_right"])
        
        if any_right_now and not self.any_right_ever:
            self.progress_events.append("ANY_RIGHT")
            self.any_right_ever = True
        
        if any_target_now and not self.any_target_ever:
            self.progress_events.append("ANY_TARGET")
            self.any_target_ever = True
        
        if all_right_now and not self.all_right_ever:
            self.progress_events.append("ALL_RIGHT")
            self.all_right_ever = True
        
        if all(info["agent_in_target"]):
            self.progress_events.append("SUCCESS")
        
        # 6. progress_event: 本step发生的进度事件（字符串列表）
        info["progress_event"] = self.progress_events[-1] if self.progress_events else None
        
        # 兼容性：保留原有字段
        info["door_open"] = [d.open for d in self.doors]
        info["switch_on"] = [s.on for s in self.real_switches]
        
        # 计算progress_score（用于HPB判定）
        progress_score = 0
        if door0_now_open:  # 任意门打开
            progress_score += 1
        if switch0_on:  # 任意真开关激活
            progress_score += 1
        if any_target_now:  # 任意agent进入target room
            progress_score += 2
        if all(info["agent_in_target"]):  # 所有agent都在target room（成功）
            progress_score += 10
        
        info["progress_score"] = progress_score
        
        # QoS tag（基于进度事件）
        # Sensitive: 任何关键进度事件发生
        if progress_score >= 1:
            info["qos_tag"] = "Sensitive"
        else:
            info["qos_tag"] = "Tolerant"
        
        reward_n = [[total_reward]] * self.num_agents
        done_n = [done] * self.num_agents
        info_n = [info] * self.num_agents
        
        if self.full_obs:
            return self._get_full_obs(), reward_n, done_n, info_n
        return self._get_obs(), reward_n, done_n, info_n
    
    def _get_obs(self):
        """
        P2修复: 保持与原SecretRooms一致的observation维度
        P3实现: 添加观测层面的解歧机制
        
        Observation包含:
        - agent归一化位置 (带噪声)
        - (可选) door状态
        - P3: switch的Unknown/True/Dummy状态（仅在距离<1.0时解歧）
        
        注意: 不显式暴露coin/dummy的统计特征，避免泄露信息
        """
        obs_n = []
        for agent in self.agents:
            # 基础位置观测 + 噪声（认知碎片化）
            pos = np.array([agent.x, agent.y]) / self.grid_size
            if self.obs_noise_level > 0:
                noise = self.random.standard_normal(2) * self.obs_noise_level
                pos = np.clip(pos + noise, 0, 1)
            
            # P2: 保持与原环境一致，不添加coin/dummy统计特征
            # coins和dummy只通过环境dynamics影响reward，不作为observation
            obs = pos
            
            # P3: 添加switch解歧特征（仅在近距离时reveal）
            # 格式: [switch_0_state, switch_1_state, ...]
            # 0 = Unknown (距离>1.0), 1 = True (真开关且近), -1 = Dummy (假开关且近)
            if self.door_in_obs:  # 只在door_in_obs=True时添加switch解歧
                switch_states = []
                for switch in self.switches:
                    dist = np.sqrt((switch.x - agent.x)**2 + (switch.y - agent.y)**2)
                    if dist <= 1.0:  # 近距离解歧
                        if switch.is_dummy:
                            switch_states.append(-1.0)  # Dummy
                        else:
                            switch_states.append(1.0)   # True
                    else:
                        switch_states.append(0.0)       # Unknown
                
                obs = np.concatenate([obs, switch_states])
            
            obs_n.append(obs)
        
        if self.door_in_obs:
            # 添加门状态
            return [np.concatenate([obs, [d.open for d in self.doors]]) for obs in obs_n]
        return obs_n
    
    def _get_full_obs(self):
        obs = np.concatenate([o for o in self._get_obs()])
        if self.door_in_obs:
            obs = np.concatenate([obs, [d.open for d in self.doors]])
        return [obs for _ in range(self.num_agents)]
    
    def get_visit_counts(self, agent_id=None):
        if agent_id is not None:
            return self.visit_counts[agent_id]
        return self.visit_counts
    
    def set_visit_counts(self, visit_counts, agent_id):
        if agent_id is None:
            assert self.visit_counts.shape == visit_counts.shape
            self.visit_counts = copy.deepcopy(visit_counts)
        else:
            assert self.visit_counts[agent_id].shape == visit_counts.shape
            self.visit_counts[agent_id] = copy.deepcopy(visit_counts)
    
    def reset_visit_counts(self):
        self.visit_counts *= 0
    
    def visit_counts_decay(self, decay_coef):
        self.visit_counts *= decay_coef
    
    def render(self, **kwargs):
        """增强渲染：显示金币和假开关"""
        map_render = copy.deepcopy(self.wall_map)
        
        # 显示金币
        for coin in self.coins:
            if not coin.collected:
                map_render[coin.x, coin.y] = 6  # 金币
        
        # 显示真开关
        for switch in self.real_switches:
            if switch.on:
                map_render[switch.x, switch.y] = 2  # 激活的真开关
            else:
                map_render[switch.x, switch.y] = 7  # 未激活的真开关
        
        # 显示假开关
        for dummy in self.dummy_switches:
            map_render[dummy.x, dummy.y] = 8  # 假开关
        
        # 显示智能体
        for agent in self.agents:
            map_render[agent.x, agent.y] = 3
        
        # 显示门
        for door in self.doors:
            if door.d == 0:
                map_render[door.x - self.door_radius:door.x + self.door_radius + 1, door.y] = 4
            else:
                map_render[door.x, door.y - self.door_radius:door.y + self.door_radius + 1] = 5
        
        print('#' * (self.grid_size + 2))
        for r in range(self.grid_size):
            print('#', end='')
            for c in range(self.grid_size):
                val = map_render[r][c]
                if val == 0:
                    print(' ', end='')
                elif val == 1:
                    print('#', end='')
                elif val == 2:
                    print('□', end='')  # 真开关(激活)
                elif val == 3:
                    print('o', end='')  # 智能体
                elif val == 4:
                    print('|', end='')  # 竖门
                elif val == 5:
                    print('-', end='')  # 横门
                elif val == 6:
                    print('$', end='')  # 金币
                elif val == 7:
                    print('·', end='')  # 真开关(未激活)
                elif val == 8:
                    print('×', end='')  # 假开关
            print('#')
        print('#' * (self.grid_size + 2))
        print(f"Time: {self.time}/{self.max_timesteps} | Coin: {self.total_coin_reward:.1f} | Target: {self.total_target_reward:.1f}")


if __name__ == "__main__":
    print("=" * 50)
    print("MI-Unity Enhanced Secret Rooms Demo")
    print("=" * 50)
    
    env = SecretRoomsMIUnity(
        map_ind=20,
        grid_size=15,
        obs_noise_level=0.1,      # 10%观测噪声
        dummy_switch_ratio=0.5,   # 50%假开关
        coin_density=20,          # 20个金币
        coin_respawn=True
    )
    
    print("\n图例:")
    print("  o - 智能体")
    print("  □ - 真开关(激活)")
    print("  · - 真开关(未激活)")
    print("  × - 假开关 (Dummy Switch)")
    print("  $ - 金币 (Coin, +0.1)")
    print("  | - 竖门")
    print("  - - 横门")
    print("  # - 墙")
    print("\n控制: w/s/a/d = 上/下/左/右")
    print("目标: 踩真开关→开门→进入目标房间 (+100)")
    print("陷阱: 捡金币很容易，但会分散注意力!\n")
    
    done = [False] * env.num_agents
    a_dict = {'w': 0, 's': 1, 'a': 2, 'd': 3}
    
    while not any(done):
        env.render()
        actions = []
        for i in range(env.num_agents):
            a = None
            while a not in ['w', 's', 'a', 'd']:
                print(f"Agent {i} action: ", end='')
                a = input()
            actions.append(a_dict[a])
        
        obs, reward, done, info = env.step(actions)
        print(f"Reward: {reward[0][0]:.2f} | Info: {info[0]}")
    
    print("\n" + "=" * 50)
    print("Episode finished!")
    print(f"Total Coin Reward: {env.total_coin_reward:.1f}")
    print(f"Total Target Reward: {env.total_target_reward:.1f}")
    print(f"Coin Ratio: {env.total_coin_reward/(env.total_coin_reward+env.total_target_reward)*100:.1f}%")
    print("=" * 50)
