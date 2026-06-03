<!-- D:\project\SWIFT\docs\risk-register.md -->
# Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sparse reward | Training may fail to discover useful behavior early. | Keep PPO+MLP baseline small, add shaped approach and safety terms, track convergence episodes. |
| PPO-HCA coupling | Attention layers may destabilize PPO updates. | Use dimension checks, normalization, gradient clipping, and compare against the MLP baseline. |
| APF local minima | APF may bias the policy toward dead ends in U-shaped or dense obstacles. | Keep HCA-only ablation, measure APF contribution, and let RL override APF features. |
| GPU availability | Long training may be blocked by local compute limits. | Keep CPU smoke tests, small deterministic scenarios, and prepare cloud GPU configuration later. |
| PyBullet substrate drift | `D:\project\pybullet` may change independently of SWIFT. | Validate through health checks and record exact adapter assumptions. |
| Premature PyBullet training coupling | Training code may become dependent on local PyBullet internals before SWIFT has a stable environment contract. | Keep Stage 1 baseline repo-native, limit PyBullet to command-level smoke, and add an environment adapter only after the contract stabilizes. |
| Demo reproducibility | GUI demos and videos can diverge from testable evidence. | Treat headless smoke tests and logged metrics as the source of truth. |
