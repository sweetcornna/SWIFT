import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pybullet_ppo_training_script_dry_run_validates_configs():
    result = subprocess.run(
        [sys.executable, "scripts/run_pybullet_ppo_training.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet PPO training config OK" in result.stdout
    assert "configs\\training.yaml" in result.stdout or "configs/training.yaml" in result.stdout
    assert "configs\\simulation.yaml" in result.stdout or "configs/simulation.yaml" in result.stdout


def test_pybullet_ppo_training_script_dry_run_avoids_runtime_imports():
    code = """
import builtins
import sys

blocked = {
    'swift.experiments.pybullet_training_runner',
    'swift.sim',
    'torch',
    'pybullet',
    'gym_pybullet_drones',
}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or any(name.startswith(prefix + '.') for prefix in blocked):
        raise RuntimeError(f'blocked import: {name}')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from scripts.run_pybullet_ppo_training import main
raise SystemExit(main(['--dry-run']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet PPO training config OK" in result.stdout


def test_pybullet_ppo_training_script_defaults_to_run_id_output_paths():
    code = """
from pathlib import Path
from scripts import run_pybullet_ppo_training as script

captured = {}

class FakeTrainingSettings:
    pass

class FakeSimulationSettings:
    pass

class FakeRunConfig:
    def __init__(self, **kwargs):
        captured['config_kwargs'] = kwargs

def fake_train(config):
    captured['config'] = config
    return {
        'artifacts': {'summary_json': 'outputs/reports/stage1/ppo_mlp/run-id.json'},
        'training': {'updates': 1, 'total_timesteps': 32},
        'runtime': {'runtime_contract': 'pybullet_velocity_training_compatibility'},
    }

script.load_training_settings = lambda path: FakeTrainingSettings()
script.load_simulation_settings = lambda path: FakeSimulationSettings()
script._load_runner = lambda: (FakeRunConfig, fake_train)
status = script.main(['--total-timesteps', '32'])
print('status=' + str(status))
print('output=' + str(captured['config_kwargs']['output']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "status=0" in result.stdout
    assert "output=None" in result.stdout
    assert "outputs/reports/stage1/ppo_mlp/run-id.json" in result.stdout
