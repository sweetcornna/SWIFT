# SWIFT

SWIFT 是“低空闪翼”城市低空配送无人机三维避障项目的企业级主仓库。仓库负责项目配置、领域模型、仿真适配、PPO/HCA/APF 训练评估、实验产物和文档治理。

本仓库可以在没有 PyBullet 的电脑上先跑 repo-native 基线、训练 smoke 和测试；如果电脑上已有本地 PyBullet 底座，也可以继续运行真实 PyBullet 速度控制训练与 checkpoint 评估。

## 1. 你要先跑哪条路径

推荐按下面顺序启动：

| 目标 | 是否需要 PyBullet | 推荐命令入口 |
| --- | --- | --- |
| 验证项目能安装和测试 | 不需要 | `python -m pytest` |
| 跑最小基线 demo | 不需要 | `scripts\run_baseline_demo.py` |
| 跑 Torch PPO smoke | 不需要 | `scripts\run_ppo_mlp_smoke.py` |
| 检查本地 PyBullet 底座 | 需要 `D:\project\pybullet` 或等价路径 | `scripts\run_pybullet_smoke.py` |
| 跑 PyBullet PPO 训练 | 需要 PyBullet + Torch 同一解释器 | `scripts\run_pybullet_ppo_training.py` |

所有命令默认从仓库根目录执行。

## 2. 一键安装部署

Windows 新电脑推荐直接运行 PowerShell bootstrap：

```powershell
.\scripts\bootstrap.ps1 -Mode quick
```

这个命令会自动：

- 创建 `.venv`
- 升级 `pip`
- 安装 `.[dev]`
- 运行 `python -m pytest`
- 运行 `scripts\swift_healthcheck.py`
- 运行 3 episode baseline demo

需要 Torch 训练能力：

```powershell
.\scripts\bootstrap.ps1 -Mode train
```

这个命令会安装 `.[dev,train]`，并额外运行 PPO MLP smoke。

需要 PyBullet 联合部署：

```powershell
.\scripts\bootstrap.ps1 -Mode pybullet -PyBulletRoot D:\project\pybullet -PyBulletVenv D:\project\.venvs\swift-pybullet-pixi
```

这个命令会使用 PyBullet Pixi Python 创建联合 venv，安装 SWIFT、Torch、PyYAML，并运行 PyBullet check-only 与 runtime smoke。

跨平台或不想运行 PowerShell 时，用 Python 入口：

```powershell
python scripts/bootstrap.py --mode quick
python scripts/bootstrap.py --mode train
python scripts/bootstrap.py --mode pybullet --pybullet-root D:\project\pybullet --pybullet-venv D:\project\.venvs\swift-pybullet-pixi
```

先预览命令、不实际安装：

```powershell
.\scripts\bootstrap.ps1 -Mode quick -DryRun
python scripts/bootstrap.py --mode pybullet --pybullet-root D:\project\pybullet --dry-run
```

只安装、不跑验证：

```powershell
.\scripts\bootstrap.ps1 -Mode train -NoVerify
python scripts/bootstrap.py --mode train --no-verify
```

后续章节是手动分步说明；如果一键命令已经通过，可以直接跳到第 11 节查看产物位置。

## 3. 系统要求

基础运行环境：

- Python `>=3.11`
- Git
- PowerShell 7 或 Windows PowerShell
- 可联网的 `pip` 环境

推荐环境：

- Windows 10/11
- Python 3.11 或 3.12 用于 PyBullet 直接运行
- Python 3.14 可以运行本仓库测试和 Torch smoke，但直接安装 `pybullet` 可能没有现成 wheel，需要走 Pixi PyBullet venv 方案

可选环境：

- `D:\project\pybullet`：本机 PyBullet 仿真底座
- `D:\project\pybullet\.tools\pixi\pixi.exe`：PyBullet 底座自带 Pixi
- `D:\project\.venvs\swift-pybullet-pixi`：推荐的 PyBullet + Torch 联合训练解释器

如果你的 PyBullet 不在 `D:\project\pybullet`，修改 [configs/simulation.yaml](configs/simulation.yaml)：

```yaml
pybullet_root: D:\project\pybullet
pixi_executable: .tools\pixi\pixi.exe
required_tasks:
  - test
  - drone-demo
check_task: test
smoke_task: drone-demo
command_timeout_seconds: 120
```

`pixi_executable` 可以写相对路径。相对路径会以 `pybullet_root` 为基准解析。

## 4. 五分钟快速启动

### Windows PowerShell

```powershell
git clone https://github.com/sweetcornna/SWIFT.git
cd SWIFT

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest
python scripts\swift_healthcheck.py
python scripts\run_baseline_demo.py --episodes 3 --output outputs\baseline\baseline_metrics.json
```

如果激活虚拟环境时报 PowerShell execution policy 错误：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
```

### macOS 或 Linux

PyBullet 路径需要按本机实际位置改写。repo-native 路径可以这样启动：

```bash
git clone https://github.com/sweetcornna/SWIFT.git
cd SWIFT

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest
python scripts/swift_healthcheck.py
python scripts/run_baseline_demo.py --episodes 3 --output outputs/baseline/baseline_metrics.json
```

## 5. 安装选项

只跑测试和基础脚本：

```powershell
python -m pip install -e ".[dev]"
```

跑 Torch PPO/HCA 训练：

```powershell
python -m pip install -e ".[dev,train]"
```

尝试直接安装 PyBullet runtime 依赖：

```powershell
python -m pip install -e ".[dev,train,sim]"
```

注意：如果当前 Python 版本没有可用的 `pybullet` wheel，`.[sim]` 可能失败。Windows 上推荐使用第 8 节的 Pixi PyBullet venv 方案。

## 6. 项目健康检查

安装后先跑：

```powershell
python -m pytest
python scripts\swift_healthcheck.py
```

当前主线验证目标：

- `python -m pytest` 应通过全部测试
- `python scripts\swift_healthcheck.py` 应输出项目配置和本地路径检查结果
- `outputs/`、`checkpoints/`、`.pytest_cache/` 等运行产物不会进入 Git

## 7. 不依赖 PyBullet 的可运行流程

### 基线 demo

```powershell
python scripts\run_baseline_demo.py --episodes 3 --output outputs\baseline\baseline_metrics.json
```

输出：

- `outputs\baseline\baseline_metrics.json`
- 包含 episode 指标、成功/碰撞/超时状态和路径指标

### PPO MLP smoke

```powershell
python -m pip install -e ".[dev,train]"
python scripts\run_ppo_mlp_smoke.py --total-timesteps 128 --output outputs\training\ppo_smoke.json
```

输出：

- summary JSON
- training history JSONL
- Torch checkpoint
- artifact manifest

用 checkpoint 做确定性评估：

```powershell
$summary = Get-Content outputs\training\ppo_smoke.json | ConvertFrom-Json
$checkpoint = $summary.artifacts.checkpoint_path
python scripts\run_ppo_checkpoint_eval.py --checkpoint $checkpoint --episodes 3 --output outputs\evaluation\ppo_checkpoint_eval.json
```

### PPO HCA smoke

```powershell
python scripts\run_ppo_hca_smoke.py --total-timesteps 64 --output outputs\training\ppo_hca_smoke.json
```

### Stage 1 tuning report

```powershell
python scripts\run_stage1_tuning.py --output outputs\tuning\stage1_grid.json
python scripts\run_stage1_report.py --ppo-summary outputs\training\ppo_smoke.json --tuning-summary outputs\tuning\stage1_grid.json --output outputs\reports\stage1_training_tuning_report.json
```

## 8. 企业级长训练证据

下面命令不是五分钟 quick start，适合做完整训练证据：

```powershell
python scripts\run_multi_seed_training.py --total-timesteps 100000 --seeds 0 1 2 --evidence-level long_training_convergence --output outputs\training\enterprise_3x100k.json
python scripts\run_multi_seed_checkpoint_eval.py --input outputs\training\enterprise_3x100k.json --episodes 3 --holdout-seeds 10000 11000 12000 --output outputs\evaluation\enterprise_3x100k_holdout.json
python scripts\run_convergence_gate.py --input outputs\evaluation\enterprise_3x100k_holdout.json --output outputs\evaluation\enterprise_3x100k_gate_fail_on_reject.json --fail-on-reject
```

边界说明：

- `cpu_smoke_ablation` 只证明训练和产物链路能跑，不是收敛证据
- `deterministic_multi_scenario_tuning` 证明确定性场景覆盖，不是训练收敛证据
- 只有 `long_training_convergence` 且 gate 通过时，才支持 repo-native 收敛声明

## 9. PyBullet 底座接入

### 9.1 只检查 PyBullet 底座

如果本机有 `D:\project\pybullet`：

```powershell
python scripts\run_pybullet_smoke.py --check-only
python scripts\run_pybullet_smoke.py
```

第一条只检查路径和 Pixi 任务配置。第二条会调用 PyBullet 底座的 drone demo。

### 9.2 运行 PyBullet runtime smoke

先尝试直接 runtime：

```powershell
python scripts\run_pybullet_runtime_smoke.py --steps 1
python scripts\run_pybullet_runtime_smoke.py --steps 1 --enable-obstacles
```

如果当前 Python 不能 import `pybullet`，脚本会尝试 Pixi fallback。成功时会看到：

- `observation_dim=15`
- `raw_observation_dim=20`
- `runtime_contract=pybullet_velocity_training_compatibility`

### 9.3 创建 PyBullet + Torch 联合训练解释器

本仓库的 PyBullet 训练需要同一个解释器同时能 import：

- SWIFT
- Torch
- PyYAML
- PyBullet
- gym-pybullet-drones

在本机验证通过的 Windows 方案：

```powershell
D:\project\pybullet\.pixi\envs\default\python.exe -m venv --system-site-packages D:\project\.venvs\swift-pybullet-pixi
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe -m pip install --upgrade pip
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe -m pip install -e D:\project\SWIFT torch PyYAML
```

如果你的仓库路径不是 `D:\project\SWIFT`，把最后一行的路径换成本机 SWIFT 根目录。

验证解释器：

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe -c "import swift, torch, yaml, pybullet; print('ok')"
```

## 10. PyBullet PPO 训练与评估

### 10.1 32-step pilot

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --total-timesteps 32 --seed 0 --output outputs\training\pybullet_pilot_32.json
```

有显式 SWIFT obstacle 的 pilot：

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --total-timesteps 32 --seed 1 --enable-obstacles --output outputs\training\pybullet_obstacles_pilot_32.json
```

### 10.2 tuned no-obstacle 训练

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --training-config configs\training_pybullet_probe.yaml --total-timesteps 32768 --seed 6 --output outputs\training\pybullet_probe_speedprior_32768.json
```

checkpoint 评估：

```powershell
$summary = Get-Content outputs\training\pybullet_probe_speedprior_32768.json | ConvertFrom-Json
$checkpoint = $summary.artifacts.checkpoint_path
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_checkpoint_eval.py --training-config configs\training_pybullet_probe.yaml --checkpoint $checkpoint --episodes 5 --seed 1700 --output outputs\evaluation\pybullet_probe_speedprior_32768_eval.json
```

已验证指标：

- training `success_rate=1.0`
- evaluation `success_rate=1.0`
- `collision_rate=0.0`
- `timeout_rate=0.0`
- `runtime_contract=pybullet_velocity_training_compatibility`

These metrics are obsolete pre-accumulated-heading evidence. They were produced
when `heading_delta` was interpreted as an absolute heading, so they do not
demonstrate navigation under the current accumulated-heading action contract.

### 10.3 tuned explicit SWIFT-obstacle 训练

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --training-config configs\training_pybullet_obstacles.yaml --total-timesteps 32768 --seed 7 --enable-obstacles --output outputs\training\pybullet_obstacles_speedprior_32768.json
```

checkpoint 评估：

```powershell
$summary = Get-Content outputs\training\pybullet_obstacles_speedprior_32768.json | ConvertFrom-Json
$checkpoint = $summary.artifacts.checkpoint_path
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_checkpoint_eval.py --training-config configs\training_pybullet_obstacles.yaml --checkpoint $checkpoint --episodes 5 --seed 1800 --enable-obstacles --output outputs\evaluation\pybullet_obstacles_speedprior_32768_eval.json
```

已验证指标：

- training `success_rate=1.0`
- evaluation `success_rate=1.0`
- `collision_rate=0.0`
- `timeout_rate=0.0`
- `average_minimum_safety_distance=0.13011363294349707`

These metrics are also obsolete pre-accumulated-heading evidence. The obstacle
was outside the direct-path safety corridor, and the historical run does not
establish current-contract obstacle avoidance.

### 10.4 randomized PyBullet robustness training

Current randomized-final status: passed on 2026-07-12. The final implementation
uses a 27D observation with up to three sorted obstacle slots, a deterministic
visibility-graph waypoint prior, and a bounded PPO residual
(`policy_residual_scale=0.25`). The final 3 x 150,000-step run passed all ten
strict robustness checks on the reserved `2000000..2000099` holdout range.

随机训练使用四阶段课程：无障碍、单个可选障碍、单个路径阻挡障碍、最终 1–3 个随机障碍。CF2X 的 `0.061 m` 碰撞外廓会参与起点、目标点和路径阻挡判定。

seed 范围必须隔离：

- `500000..500099`：开发验证，可用于 pilot 和参数选择。
- `1000000..1000099`：历史实验已经消费，不得再次用于调参或最终声明。
- `2000000..2000099`：保留给最终 checkpoint，每个 checkpoint 只评估一次。

先验证 100 个最终阶段布局的真实 PyBullet 初始净空：

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_geometry_check.py --seed 500000 --scenarios 100 --output outputs\evaluation\pybullet_curriculum_geometry_validation.json --fail-on-invalid
```

几何门禁通过后依次运行 `2,048` smoke、`32,768` pilot 和 `100,000` candidate：

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 2048 --output outputs\training\pybullet_curriculum_smoke_1x2048.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 32768 --output outputs\training\pybullet_curriculum_pilot_1x32768.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_pilot_1x32768.json --episodes 20 --holdout-seed 500000 --output outputs\evaluation\pybullet_curriculum_pilot_validation.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 100000 --output outputs\training\pybullet_curriculum_candidate_1x100k.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_candidate_1x100k.json --episodes 100 --holdout-seed 500000 --output outputs\evaluation\pybullet_curriculum_candidate_validation.json
```

只有 candidate 达到成功率 `>=0.80`、碰撞率 `<=0.05`、超时率 `<=0.20` 才启动完整训练：

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 9 10 --total-timesteps 150000 --output outputs\training\pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k.json --episodes 100 --holdout-seed 2000000 --output outputs\evaluation\pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k_final_holdout.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_robustness_gate.py --input outputs\evaluation\pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k_final_holdout.json --output outputs\evaluation\pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k_gate.json --fail-on-reject
```

严格门槛保持不变：最差成功率不低于 `0.95`、最大碰撞率等于 `0`、最大超时率不高于 `0.05`、最差平均最小安全距离不低于 `0.10`。最终结果为最差成功率 `0.99`、最大碰撞率 `0.0`、最大超时率 `0.01`、最差平均最小安全距离 `0.12744620047273952`，并记录 `robustness_claim=true`、`passed_gates=10/10`。gate 失败仍会写报告，但不形成 robustness claim；该结论仅覆盖单无人机、headless、静态障碍 PyBullet 仿真，不等于真实飞行安全证据。

## 10.5 SWIFT-native robust hover 集成

此入口是独立的 stabilized-hover 能力，不修改第 10.4 节的 SWIFT
navigation、累计航向、障碍课程或 APF/HCA 行为。SWIFT 拥有配置、动作契约、
哈希绑定和发布逻辑；外部 PyBullet 仿真/训练底座不会被复制进本仓库。
默认 `configs/simulation.yaml` 假定它位于本仓库的相邻 `../pybullet` 目录；其他布局可通过
自定义 simulation config 指定。

默认科学配置位于 `configs/pybullet_hover.yaml`：

- `ct_att_yawrate_v1`：collective thrust + roll/pitch attitude + yaw rate；
- `(1,72)` observation / `(1,4)` action，30 Hz 控制、240 Hz physics；
- `robust_uniform_v1` reset、`hover_kin_safe_v1` reward；
- 只更新前 12 个物理 observation 的 `train_rms_physical12_v1`；
- 固定 100-case held-out bank，按 safety completion、p95 altitude MAE、return
  做 lexicographic robust checkpoint selection。

先做不启动训练的契约检查：

```powershell
python scripts\run_pybullet_hover_training.py --dry-run
python scripts\run_pybullet_hover_training.py --help
python scripts\run_pybullet_hover_eval.py --help
python scripts\run_pybullet_hover_viz.py --help
```

训练、只读评估和后处理可视化：

```powershell
python scripts\run_pybullet_hover_training.py --run-name swift-hover-seed1 --timeout 21600
python scripts\run_pybullet_hover_eval.py --model <run>\robust_best_model.zip --output <new-eval-dir> --evaluation-seed 101 --cases 100 --timeout 3600
python scripts\run_pybullet_hover_viz.py --input <eval-dir> --output <new-viz-dir> --html --timeout 600
```

训练 summary schema 5 将每个 `best`、`robust_best`、`final` ZIP 与精确
`.obsnorm.npz` sidecar 通过 SHA-256 绑定。评估拒绝未绑定或被修改的模型，并检查
输入训练目录在运行前后完全不变。

### 120k 三种子已训练模型

`artifacts/robust-hover/120k/` 发布 seeds 1–3 的 120,000 requested-step
`robust_best_model.zip` 和 `final_model.zip`、各自精确 normalization sidecar、
`action_profile.json`、`source_hashes.json`、`summary.json`、compact evaluation
JSON 和 `manifest.sha256.json`。不复制 `evaluation.npz`、训练 callback NPZ、
validation bank、trajectory bulk 或整个 PyBullet substrate。

三个 120k run 的实际 PPO timesteps 都是 120,832；robust-best 选择点分别是
seed 1: 110k、seed 2: 100k、seed 3: 110k。独立 100-case evaluations 均为
`accepted`，但证据范围仅为 single-drone、headless PyBullet stabilized hover，
不是导航能力或真实飞行安全证明。

可从本地已验证底座重新生成 curated bundle：

```powershell
python scripts\publish_pybullet_hover_models.py `
  --training-run ..\pybullet\results\ppo\ppo-hover-ct-att-yawrate-v1-120k-seed1-r1 `
  --training-run ..\pybullet\results\ppo\ppo-hover-ct-att-yawrate-v1-120k-seed2-r1 `
  --training-run ..\pybullet\results\ppo\ppo-hover-ct-att-yawrate-v1-120k-seed3-r1 `
  --evaluation ..\pybullet\results\ppo-evaluation\ct-att-yawrate-v1-120k-seed1-robust-best-r1 `
  --evaluation ..\pybullet\results\ppo-evaluation\ct-att-yawrate-v1-120k-seed2-robust-best-r1 `
  --evaluation ..\pybullet\results\ppo-evaluation\ct-att-yawrate-v1-120k-seed3-robust-best-r1
```

## 11. 产物在哪里

运行脚本会生成：

| 目录 | 内容 |
| --- | --- |
| `outputs\baseline\` | baseline demo 指标 |
| `outputs\training\` | 手动指定的训练 summary JSON |
| `outputs\evaluation\` | checkpoint evaluation JSON |
| `outputs\episodes\` | update-by-update JSONL history |
| `outputs\reports\` | run-id 自动生成的报告 |
| `checkpoints\` | Torch checkpoint |

这些目录已在 [.gitignore](.gitignore) 中忽略，不要提交训练产物和 checkpoint。

每个正式报告会包含：

- `record_type`
- `schema_version`
- `runtime_contract` 或训练后端信息
- `metrics`
- `artifacts`
- manifest sidecar

PyBullet 训练报告：

- `record_type=pybullet_ppo_training_report`
- `training_backend=torch_ppo_mlp_pybullet_velocity`

PyBullet checkpoint 评估报告：

- `record_type=pybullet_ppo_checkpoint_evaluation`
- manifest 会记录 checkpoint、配置文件和 summary 的 SHA-256

## 12. 重要配置文件

| 文件 | 用途 |
| --- | --- |
| [configs/simulation.yaml](configs/simulation.yaml) | 本地 PyBullet 底座路径和 Pixi 任务 |
| [configs/pybullet_hover.yaml](configs/pybullet_hover.yaml) | stabilized-hover 训练、normalization 和 robust selection 契约 |
| [configs/training.yaml](configs/training.yaml) | repo-native 默认训练配置 |
| [configs/training_pybullet_probe.yaml](configs/training_pybullet_probe.yaml) | PyBullet no-obstacle 可达速度控制训练 |
| [configs/training_pybullet_obstacles.yaml](configs/training_pybullet_obstacles.yaml) | PyBullet 显式 SWIFT obstacle 训练 |
| [configs/training_pybullet_randomized.yaml](configs/training_pybullet_randomized.yaml) | PyBullet 每回合随机静态障碍多种子训练 |
| [configs/evaluation.yaml](configs/evaluation.yaml) | evidence profile 和 convergence gate |

跨电脑运行时最常改的是 `configs/simulation.yaml` 的 `pybullet_root`。

## 13. 代码结构

| 路径 | 说明 |
| --- | --- |
| `src\swift\core\` | 类型、指标、领域基础结构 |
| `src\swift\envs\` | repo-native 和 PyBullet 训练环境 |
| `src\swift\rl\` | MLP、HCA、APF、Torch PPO |
| `src\swift\sim\` | PyBullet substrate 和 runtime adapter |
| `src\swift\experiments\` | 训练、评估、manifest、报告 |
| `scripts\` | 可直接运行的 CLI |
| `tests\` | 单元测试、脚本测试、文档测试 |
| `docs\` | 项目章程、架构、路线图、风险、材料追踪 |

## 14. 常见问题

### `python` 版本不对

先查：

```powershell
python --version
py -0p
```

推荐用 Python 3.11 或 3.12 创建 `.venv`。如果只跑 repo-native 测试，Python 3.14 也可用；如果要直接装 PyBullet，优先 3.11/3.12。

### PowerShell 不能激活虚拟环境

执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

然后重新激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

### `run_pybullet_smoke.py --check-only` 失败

检查：

```powershell
Test-Path D:\project\pybullet
Test-Path D:\project\pybullet\.tools\pixi\pixi.exe
Get-Content configs\simulation.yaml
```

如果 PyBullet 在别的目录，修改 `configs/simulation.yaml`。

### `import pybullet` 失败

先不要改 SWIFT 代码。优先使用 PyBullet 底座自己的 Pixi Python 创建联合 venv：

```powershell
D:\project\pybullet\.pixi\envs\default\python.exe -m venv --system-site-packages D:\project\.venvs\swift-pybullet-pixi
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe -m pip install -e D:\project\SWIFT torch PyYAML
```

### Torch 安装慢或失败

先确认 Python 版本，再单独安装：

```powershell
python -m pip install --upgrade pip
python -m pip install torch
python -c "import torch; print(torch.__version__)"
```

### 输出文件很多

正常。`outputs/` 和 `checkpoints/` 是实验产物目录，已经被 Git 忽略。需要清理时可以删除这些目录，不影响源码：

```powershell
Remove-Item -Recurse -Force outputs, checkpoints
```

只在确认不需要历史实验产物时执行。

## 15. 项目边界

- SWIFT 是主项目仓库，负责训练、评估、配置、报告和工程治理
- `D:\project\pybullet` 是外部本地仿真底座，不把其内部代码复制进 SWIFT
- 当前 PyBullet runtime 是单无人机、headless、velocity-action、无录制模式
- 仿真成功不是实飞安全证明
- GUI、多无人机、视频录制、复杂城市场景和真实飞行验证属于后续阶段

## 16. 新电脑验收清单

在一台新电脑上，最小验收：

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts\swift_healthcheck.py
python scripts\run_baseline_demo.py --episodes 3 --output outputs\baseline\baseline_metrics.json
```

有 Torch 时增加：

```powershell
python -m pip install -e ".[dev,train]"
python scripts\run_ppo_mlp_smoke.py --total-timesteps 128 --output outputs\training\ppo_smoke.json
```

有 PyBullet 底座时增加：

```powershell
python scripts\run_pybullet_smoke.py --check-only
python scripts\run_pybullet_runtime_smoke.py --steps 1
```

有 PyBullet + Torch 联合解释器时增加：

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --training-config configs\training_pybullet_probe.yaml --total-timesteps 32768 --seed 6 --output outputs\training\pybullet_probe_speedprior_32768.json
```
