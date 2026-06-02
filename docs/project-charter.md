<!-- D:\project\SWIFT\docs\project-charter.md -->
# Project Charter

## Mission

SWIFT builds an enterprise-grade main repository for an urban drone delivery
3D obstacle avoidance system. The system target is a demonstrable, verifiable,
and extensible simulation platform for safe low-altitude delivery navigation.

## Technical Line

- PPO is the decision baseline for continuous action control.
- HCA is the perception upgrade for layered target and threat attention.
- APF is the safety prior that provides interpretable attraction and repulsion
  vectors.
- PyBullet is the simulation substrate for 3D urban low-altitude validation.

## Bootstrap Boundary

The bootstrap creates repository structure, configuration, documentation,
health checks, and contracts. Full PPO training, HCA model execution, APF-HCA
fusion, and long-running GPU experiments are follow-on stages.

## Local Substrate

The local PyBullet substrate is `D:\project\pybullet`. SWIFT validates it
through a command-level adapter and does not copy it into this repository.
