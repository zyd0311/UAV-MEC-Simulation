# 1) 放置 EVA-Gen agent 模块

在你的工程根目录（有 `src/` 的那个目录）下执行：

```
mkdir -p src/algo/eva_gen
# 把你现在的 EVA-Gen 文件（mi_unity_framework_complete_v2_patched.py）复制进去
cp /path/to/mi_unity_framework_complete_v2_patched.py src/algo/eva_gen/eva_gen_agent.py
```

再创建 `src/algo/eva_gen/__init__.py`：

```
from .eva_gen_agent import MIUnityAgent
```

------

# 2) 新增：EVA-Gen 训练脚本（train_eva_gen.py）

在工程根目录新建 `train_eva_gen.py`，内容如下（直接复制）：

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, time, datetime
from pathlib import Path
import numpy as np
import torch

from src.train import make_grid_env
from src.algo.eva_gen.eva_gen_agent import MIUnityAgent


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rollout_eval(env, agent: MIUnityAgent, episodes: int, seed: int):
    succ, crs, coins, targets, rets = [], [], [], [], []
    for ep in range(episodes):
        env.seed(seed * 50000 + ep * 1000)
        obs = env.reset()
        agent.reset_hidden_states()
        done = False
        ep_ret = 0.0
        last_info = {}
        steps = 0
        while not done and steps < env.max_timesteps:
            obs_arr = np.stack(obs, axis=0)  # [N, obs_dim]
            actions, _, _ = agent.select_action(obs_arr, deterministic=True)
            obs, r_n, d_n, info_n = env.step(actions.tolist())
            ep_ret += float(r_n[0][0])
            done = bool(d_n[0])
            last_info = info_n[0] if isinstance(info_n, list) else {}
            steps += 1

        ttr = float(last_info.get("total_target_reward", 0.0))
        tcr = float(last_info.get("total_coin_reward", 0.0))
        cr = float(last_info.get("coin_ratio", tcr / max(tcr + ttr, 1e-6)))
        succ.append(1 if ttr > 0 else 0)
        crs.append(cr)
        coins.append(tcr)
        targets.append(ttr)
        rets.append(ep_ret)

    return {
        "success_rate": float(np.mean(succ)),
        "coin_ratio": float(np.mean(crs)),
        "total_coin_reward_mean": float(np.mean(coins)),
        "total_target_reward_mean": float(np.mean(targets)),
        "episode_return_mean": float(np.mean(rets)),
    }


def main():
    ap = argparse.ArgumentParser()
    # Scene A defaults
    ap.add_argument("--env_name", default="SecretRoomMIUnity-v0")
    ap.add_argument("--map_ind", type=int, default=20)
    ap.add_argument("--grid_size", type=int, default=30)
    ap.add_argument("--n_agents", type=int, default=2)
    ap.add_argument("--max_timesteps", type=int, default=200)
    ap.add_argument("--episode_length", type=int, default=200)
    ap.add_argument("--activate_radius", type=float, default=1.5)
    ap.add_argument("--door_in_obs", action="store_true", default=False)
    ap.add_argument("--full_obs", action="store_true", default=False)
    ap.add_argument("--joint_count", action="store_true", default=False)

    # MIUnity params
    ap.add_argument("--obs_noise_level", type=float, default=0.3)
    ap.add_argument("--dummy_switch_ratio", type=float, default=0.5)
    ap.add_argument("--coin_density", type=int, default=50)
    ap.add_argument("--coin_respawn", action="store_true", default=False)

    # Budget / run
    ap.add_argument("--num_env_steps", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--experiment_name", default="EVA-Gen_MIUnity")

    # EVA-Gen hparams
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--belief_dim", type=int, default=64)
    ap.add_argument("--ddim_steps", type=int, default=10)
    ap.add_argument("--belief_update_freq", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--buffer_size", type=int, default=3000)
    ap.add_argument("--p_hpb", type=float, default=0.7)
    ap.add_argument("--hp_threshold", type=float, default=50.0)

    # intervals (episodes)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--eval_episodes", type=int, default=200)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = pick_device(args.device)
    print("[Device]", device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build env via your existing factory -> identical Scene-A config
    class A: ...
    all_args = A()
    for k, v in vars(args).items():
        setattr(all_args, k, v)
    all_args.num_agents = args.n_agents
    all_args.use_parallel = False
    env = make_grid_env(all_args)

    obs0 = env.reset()[0]
    obs_dim = int(np.asarray(obs0).shape[-1])
    action_dim = int(env.action_space[0].n)
    print(f"[Env] obs_dim={obs_dim}, action_dim={action_dim}, agents={args.n_agents}")

    agent = MIUnityAgent(
        num_agents=args.n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        belief_dim=args.belief_dim,
        ddim_steps=args.ddim_steps,
        belief_update_freq=args.belief_update_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        p_HPB=args.p_hpb,
        high_value_threshold=args.hp_threshold,
    )

    # Results dir (same style as repo)
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    base = Path(args.run_dir) / "results" if args.run_dir else (Path(__file__).resolve().parent / "results")
    run_dir = base / args.env_name / str(args.map_ind) / "eva_gen" / f"{args.experiment_name}-{ts}-seed{args.seed}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "params.json").write_text(json.dumps(vars(args), indent=2))

    train_csv = run_dir / "logs" / "train_log.csv"
    eval_csv = run_dir / "logs" / "eval_log.csv"
    if not train_csv.exists():
        train_csv.write_text(
            "episode,steps,episode_return,success,total_coin,total_target,coin_ratio,actor_loss,critic_loss,dt_loss,gbm_loss,mi_loss\n"
        )
    if not eval_csv.exists():
        eval_csv.write_text(
            "episode,steps,success_rate,coin_ratio,total_coin_reward_mean,total_target_reward_mean,episode_return_mean\n"
        )

    total_steps = 0
    ep_idx = 0
    start = time.time()

    while total_steps < args.num_env_steps:
        env.seed(args.seed + ep_idx * 1000)
        obs = env.reset()
        agent.reset_hidden_states()

        done = False
        ep_ret = 0.0
        last_info = {}
        episode_buf = []  # (obs_arr, actions, reward, next_obs_arr, done, log_probs)

        steps_in_ep = 0
        while not done and steps_in_ep < args.episode_length and total_steps < args.num_env_steps:
            obs_arr = np.stack(obs, axis=0)
            actions, log_probs, _ = agent.select_action(obs_arr, deterministic=False)

            next_obs, r_n, d_n, info_n = env.step(actions.tolist())
            r = float(r_n[0][0])
            done = bool(d_n[0])
            info = info_n[0] if isinstance(info_n, list) else {}

            ep_ret += r
            total_steps += 1
            steps_in_ep += 1
            last_info = info

            episode_buf.append((obs_arr, actions.copy(), r, np.stack(next_obs, axis=0), done, log_probs.copy()))
            obs = next_obs

            # light online updates (may be empty until buffer fills)
            agent.update()

        # Episode-level QoS tagging: success episode -> all transitions Sensitive
        success = 1 if float(last_info.get("total_target_reward", 0.0)) > 0 else 0
        qos_tag = "Sensitive" if success == 1 else "Tolerant"
        for (o, a, r, no, d, lp) in episode_buf:
            agent._last_log_probs = lp  # reuse your patched PPO log_prob storage
            agent.store_transition(o, a, r, no, d, {"qos_tag": qos_tag}, ep_ret)

        log = agent.update()

        tcr = float(last_info.get("total_coin_reward", 0.0))
        ttr = float(last_info.get("total_target_reward", 0.0))
        cr = float(last_info.get("coin_ratio", tcr / max(tcr + ttr, 1e-6)))
        with open(train_csv, "a") as f:
            f.write(
                f"{ep_idx},{total_steps},{ep_ret:.6f},{success},{tcr:.6f},{ttr:.6f},{cr:.6f},"
                f"{log.get('actor_loss', np.nan)},{log.get('critic_loss', np.nan)},"
                f"{log.get('dt_loss', np.nan)},{log.get('gbm_loss', np.nan)},{log.get('mi_loss', np.nan)}\n"
            )

        if ep_idx > 0 and ep_idx % args.save_interval == 0:
            cp = run_dir / "models" / f"cp_{ep_idx}"
            cp.mkdir(parents=True, exist_ok=True)
            agent.save_models(str(cp / "eva_gen.pt"))
            agent.save_models(str(run_dir / "models" / "eva_gen.pt"))
            print(f"[Save] ep={ep_idx} steps={total_steps}")

        if ep_idx > 0 and ep_idx % args.eval_interval == 0:
            m = rollout_eval(env, agent, args.eval_episodes, args.seed)
            with open(eval_csv, "a") as f:
                f.write(
                    f"{ep_idx},{total_steps},{m['success_rate']:.6f},{m['coin_ratio']:.6f},"
                    f"{m['total_coin_reward_mean']:.6f},{m['total_target_reward_mean']:.6f},{m['episode_return_mean']:.6f}\n"
                )
            print(
                f"[Eval] ep={ep_idx} steps={total_steps} "
                f"succ={m['success_rate']:.3f} coin_ratio={m['coin_ratio']:.3f} "
                f"ret={m['episode_return_mean']:.2f} t={(time.time()-start)/60:.1f}m"
            )

        ep_idx += 1

    agent.save_models(str(run_dir / "models" / "eva_gen.pt"))
    print("[Done] run_dir:", run_dir)


if __name__ == "__main__":
    main()
```

------

# 3) 运行 EVA-Gen（A 场景，推荐命令）

在工程根目录运行：

```
python train_eva_gen.py \
  --env_name SecretRoomMIUnity-v0 \
  --map_ind 20 \
  --max_timesteps 200 \
  --episode_length 200 \
  --obs_noise_level 0.3 \
  --dummy_switch_ratio 0.5 \
  --coin_density 50 \
  --coin_respawn False \
  --door_in_obs False \
  --num_env_steps 10000000 \
  --seed 0 \
  --device auto
```

它会在类似路径产出结果：

```
results/SecretRoomMIUnity-v0/20/eva_gen/EVA-Gen_MIUnity-<timestamp>-seed0/
  models/cp_50/eva_gen.pt
  models/eva_gen.pt
  logs/train_log.csv
  logs/eval_log.csv
  params.json
```

> 先建议你把 `--num_env_steps` 改成 `200000` 做一次冒烟测试，确认能跑、会保存、eval 会出。

------

# 4) 新增：离线评估脚本（analyze_eva_gen_run.py）

如果你希望像 MAPPO/MACE 一样生成 `checkpoint_eval.csv`，创建 `analyze_eva_gen_run.py`：

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re, csv
from pathlib import Path
import numpy as np
import torch

from src.train import make_grid_env
from src.algo.eva_gen.eva_gen_agent import MIUnityAgent


def list_checkpoints(models_dir: Path):
    cps = []
    for p in models_dir.glob("cp_*"):
        if p.is_dir():
            m = re.search(r"cp_(\d+)", p.name)
            if m:
                cps.append((int(m.group(1)), p))
    cps.sort(key=lambda x: x[0])
    return [p for _, p in cps]


def rollout_eval(env, agent, episodes, seed):
    succ, crs, coins, targets, rets = [], [], [], [], []
    for ep in range(episodes):
        env.seed(seed * 50000 + ep * 1000)
        obs = env.reset()
        agent.reset_hidden_states()
        done = False
        ep_ret = 0.0
        last_info = {}
        steps = 0
        while not done and steps < env.max_timesteps:
            obs_arr = np.stack(obs, axis=0)
            actions, _, _ = agent.select_action(obs_arr, deterministic=True)
            obs, r_n, d_n, info_n = env.step(actions.tolist())
            ep_ret += float(r_n[0][0])
            done = bool(d_n[0])
            last_info = info_n[0] if isinstance(info_n, list) else {}
            steps += 1

        ttr = float(last_info.get("total_target_reward", 0.0))
        tcr = float(last_info.get("total_coin_reward", 0.0))
        cr = float(last_info.get("coin_ratio", tcr / max(tcr + ttr, 1e-6)))
        succ.append(1 if ttr > 0 else 0)
        crs.append(cr)
        coins.append(tcr)
        targets.append(ttr)
        rets.append(ep_ret)

    return {
        "success_rate": float(np.mean(succ)),
        "coin_ratio": float(np.mean(crs)),
        "total_coin_reward_mean": float(np.mean(coins)),
        "total_target_reward_mean": float(np.mean(targets)),
        "episode_return_mean": float(np.mean(rets)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    params = json.loads((run_dir / "params.json").read_text())
    models_dir = run_dir / "models"

    class A: ...
    all_args = A()
    for k, v in params.items():
        setattr(all_args, k, v)
    all_args.num_agents = params.get("n_agents", 2)
    all_args.use_parallel = False

    env = make_grid_env(all_args)

    obs_dim = int(np.asarray(env.reset()[0]).shape[-1])
    action_dim = int(env.action_space[0].n)
    agent = MIUnityAgent(
        num_agents=all_args.num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=params.get("hidden_dim", 128),
        belief_dim=params.get("belief_dim", 64),
        ddim_steps=params.get("ddim_steps", 10),
        belief_update_freq=params.get("belief_update_freq", 10),
        buffer_size=params.get("buffer_size", 3000),
        batch_size=params.get("batch_size", 256),
        p_HPB=params.get("p_hpb", 0.7),
        high_value_threshold=params.get("hp_threshold", 50.0),
    )

    out_rows = []
    seed = params.get("seed", 0)

    # checkpoints
    for cp_dir in list_checkpoints(models_dir):
        ckpt = cp_dir / "eva_gen.pt"
        if not ckpt.exists():
            continue
        agent.load_models(str(ckpt))
        m = rollout_eval(env, agent, args.episodes, seed)
        out_rows.append({"checkpoint": cp_dir.name, **m})
        print(cp_dir.name, m)

    # final
    final_ckpt = models_dir / "eva_gen.pt"
    if final_ckpt.exists():
        agent.load_models(str(final_ckpt))
        m = rollout_eval(env, agent, args.episodes, seed)
        out_rows.append({"checkpoint": "final", **m})
        print("final", m)

    out_path = run_dir / "checkpoint_eval.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    print("[OK] wrote", out_path)


if __name__ == "__main__":
    main()
```

运行方式：

```
python analyze_eva_gen_run.py \
  --run_dir results/SecretRoomMIUnity-v0/20/eva_gen/EVA-Gen_MIUnity-XXXX-seed0 \
  --episodes 200
```

------

# 5) 两个你很可能会遇到的坑（提前告诉你）

### 坑 A：EVA-Gen 的 device 在 agent 文件内部有自己的 device 选择

你如果发现它打印 `使用设备: cpu`，说明它内部没选到 mps。
 处理方式：进 `src/algo/eva_gen/eva_gen_agent.py`，把 device 选择逻辑改成优先 mps（和你 MAPPO/MACE 一样）。

### 坑 B：成功 episode 极少导致 HPB 太稀

我在训练脚本里做了一个关键操作：**整条成功 episode 全部标为 Sensitive**（而不是仅最后一步）。
 这一步非常重要，否则你 HPB 永远没样本，EVA-Gen 的优势很难出现。