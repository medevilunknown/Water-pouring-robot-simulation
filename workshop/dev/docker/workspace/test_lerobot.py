import torch
import os
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

print(f"Testing on Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 1. Test Dataset Creation (Disk I/O and Parquet)
print("1. Testing LeRobotDataset creation...")
dataset_path = "local/test_dummy_dataset"
dataset = LeRobotDataset.create(
    repo_id=dataset_path,
    fps=30,
    features={
        "observation.image": {"dtype": "video", "shape": (3, 224, 224)},
        "action": {"dtype": "float32", "shape": (6,)}
    }
)
print("PASS: Dataset schema created.")

# 2. Test ACT Policy memory allocation
print("2. Testing ACT Policy memory allocation...")

# The ACTPolicy requires a configuration specifying input/output feature schemas.
# For this test we create a minimal config that matches the dataset schema.
config = ACTConfig(
    input_features={
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    },
    n_action_steps=10,
    n_obs_steps=1,
    chunk_size=10,
)

policy = ACTPolicy(config=config)

# Move to GPU if available
if torch.cuda.is_available():
    policy.cuda()
    print("PASS: Policy successfully allocated to GPU memory.")
else:
    print("WARNING: Testing on CPU. GPU passthrough failed.")

print("All AI framework tests passed!")