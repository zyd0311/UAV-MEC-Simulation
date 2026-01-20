import numpy as np


class HDDTable(object):
    def __init__(self, act_space, args):
        self.args = args
        self.act_space = act_space

        self.obs_range = np.array(args.obs_range)
        self.obs_scale = np.array(args.obs_scale)
        self.last_rew = args.hdd_last_rew
        self.reverse_ret = args.hdd_reverse_ret
        self.rew_nbins = args.discrete_nbins
        self.rew_bound = np.linspace(0, args.n_agents - 1, self.rew_nbins + 1)
        self.ret_nbins = args.hdd_ret_nbins
        self.ret_ub = args.hdd_ret_ub
        self.ret_bound = np.linspace(0, self.ret_ub, self.ret_nbins + 1)
        self.ret_bound[-1] = np.inf
        self.wsize = args.hdd_count_window
        self.decay = args.hdd_count_decay
        self.init1 = args.hdd_count_init1
        
        assert self.act_space.__class__.__name__ == "Discrete", "Count-based HDD only supports discrete action space."
        count_shape = (self.act_space.n,) + tuple(self.obs_range[self.obs_range > 0])
        assert not (self.last_rew and self.reverse_ret), "last_rew and reverse_ret cannot be both True."
        if self.last_rew:
            count_shape += (self.rew_nbins,)
        if self.reverse_ret:
            count_shape += (self.ret_nbins,)
        count_shape += (self.ret_nbins,)
        
        if self.wsize > 1:
            count_shape = (self.wsize,) + count_shape
            self.widx = 0
            
        self.hdd = np.zeros(count_shape, dtype=np.float32)
        if self.init1:
            self.hdd += 1

    def update(self, obs, actions, returns, last_rewards=None, reverse_returns=None):
        act_ind = actions.astype(np.int32)
        obs_ind = []
        # 修复：只处理obs_scale配置的维度，避免索引越界
        max_dim = min(obs.shape[-1], len(self.obs_scale))
        for index in range(max_dim):
            if index < len(self.obs_scale) and self.obs_scale[index] > 0:
                # 修复索引越界：clip到有效范围 [0, obs_range[index]-1]
                obs_val = obs[..., index] * self.obs_scale[index]
                obs_val = np.clip(obs_val, 0, self.obs_range[index] - 1)
                obs_ind.append(obs_val.astype(np.int32))
        obs_ind = np.stack(obs_ind, axis=-1)
        ret_ind = np.digitize(returns, self.ret_bound) - 1
        # 修复索引越界：clip到有效范围 [0, ret_nbins-1]
        ret_ind = np.clip(ret_ind, 0, self.ret_nbins - 1)
        indices = np.concatenate([act_ind, obs_ind, ret_ind], axis=-1)
        
        if self.last_rew:
            assert last_rewards is not None, "None for last_rewards."
            last_rew_ind = np.digitize(last_rewards, self.rew_bound) - 1
            indices = np.concatenate([act_ind, obs_ind, last_rew_ind, ret_ind], axis=-1)
        if self.reverse_ret:
            assert reverse_returns is not None, "None for reverse_returns."
            reverse_ret_ind = np.digitize(reverse_returns, self.ret_bound) - 1
            indices = np.concatenate([act_ind, obs_ind, reverse_ret_ind, ret_ind], axis=-1)
        
        indices = indices.reshape(-1, indices.shape[-1])
        if self.wsize > 1:
            self.hdd[self.widx][:] = 1 if self.init1 else 0
            for i in range(indices.shape[0]):
                self.hdd[self.widx][tuple(indices[i])] += 1
            self.widx = (self.widx + 1) % self.wsize
        else:
            self.hdd *= self.decay
            for i in range(indices.shape[0]):
                self.hdd[tuple(indices[i])] += 1
    
    def predict(self, obs, actions, returns, last_rewards=None, reverse_returns=None):
        act_ind = actions.astype(np.int32)
        obs_ind = []
        # 修复：只处理obs_scale配置的维度，避免索引越界
        max_dim = min(obs.shape[-1], len(self.obs_scale))
        for index in range(max_dim):
            if index < len(self.obs_scale) and self.obs_scale[index] > 0:
                # 修复索引越界：clip到有效范围 [0, obs_range[index]-1]
                obs_val = obs[..., index] * self.obs_scale[index]
                obs_val = np.clip(obs_val, 0, self.obs_range[index] - 1)
                obs_ind.append(obs_val.astype(np.int32))
        obs_ind = np.stack(obs_ind, axis=-1)
        ret_ind = np.digitize(returns, self.ret_bound) - 1
        # 修复索引越界：clip到有效范围 [0, ret_nbins-1]
        ret_ind = np.clip(ret_ind, 0, self.ret_nbins - 1)
        indices = np.concatenate([act_ind, obs_ind, ret_ind], axis=-1)
        
        if self.last_rew:
            assert last_rewards is not None, "None for last_rewards."
            last_rew_ind = np.digitize(last_rewards, self.rew_bound) - 1
            # 修复索引越界：clip到有效范围 [0, rew_nbins-1]
            last_rew_ind = np.clip(last_rew_ind, 0, self.rew_nbins - 1)
            indices = np.concatenate([act_ind, obs_ind, last_rew_ind, ret_ind], axis=-1)
        if self.reverse_ret:
            assert reverse_returns is not None, "None for reverse_returns."
            reverse_ret_ind = np.digitize(reverse_returns, self.ret_bound) - 1
            # 修复索引越界：clip到有效范围 [0, ret_nbins-1]
            reverse_ret_ind = np.clip(reverse_ret_ind, 0, self.ret_nbins - 1)
            indices = np.concatenate([act_ind, obs_ind, reverse_ret_ind, ret_ind], axis=-1)
        
        indices = tuple([indices[..., i:i+1] for i in range(indices.shape[-1])])
        if self.wsize > 1:
            return self.hdd.sum(axis=0)[indices] / self.hdd.sum(axis=(0, 1))[indices[1:]]
        return self.hdd[indices] / self.hdd.sum(axis=0)[indices[1:]]
    
    
class HDDTables(object):
    def __init__(self, act_space, args):
        self.reduce = args.hdd_reduce
        self.n_other_agent = args.n_agents - 1
        
        if self.reduce:
            self.hdd = HDDTable(act_space, args)
        else:
            self.hdd = [HDDTable(act_space, args) for _ in range(self.n_other_agent)]
    
    def update(self, obs, actions, returns, last_rewards=None, reverse_returns=None,
               available_actions=None, active_masks=None):
        if self.reduce:
            self.hdd.update(obs, actions, returns, last_rewards, reverse_returns)
        else:
            for i in range(self.n_other_agent):
                self.hdd[i].update(obs, actions, returns[i],
                                   None if last_rewards is None else last_rewards[i],
                                   None if reverse_returns is None else reverse_returns[i])
        return {}
    
    def evaluate_actions(self, obs, actions, returns, last_rewards=None, reverse_returns=None,
                         available_actions=None):
        if self.reduce:
            return self.hdd.predict(obs, actions, returns, last_rewards, reverse_returns)
        
        probs = []
        for i in range(self.n_other_agent):
            probs.append(self.hdd[i].predict(obs, actions, returns[i],
                                             None if last_rewards is None else last_rewards[i],
                                             None if reverse_returns is None else reverse_returns[i]))
        return np.stack(probs, axis=0)
    
    def save(self, save_dir, index):
        if self.reduce:
            np.save(str(save_dir) + f"/hdd_agent{index}.npy", self.hdd.hdd)
            return
        
        for i, h in enumerate(self.hdd):
            np.save(str(save_dir) + f"/hdd_agent{index}_{i}.npy", h.hdd)
            
    def restore(self, model_dir, index):
        if self.reduce:
            self.hdd.hdd = np.load(str(model_dir) + f"/hdd_agent{index}.npy")
            return
        
        for i, h in enumerate(self.hdd):
            h.hdd = np.load(str(model_dir) + f"/hdd_agent{index}_{i}.npy")