import huggingface_hub as hf
import io
import textwrap
from pathlib import Path
from src.utils.arg_handler import get_hf_args
from src.shared.wandb_wrapper import WandbWrapper
from src.shared.environment_manager import EnvironmentManager
from src.ppo.agent_continuous import PPOAgentContinuous
from src.sac.agent import SACAgent
from td3.agent import TD3Agent


current_path = Path(__file__).resolve()
project_root = current_path.parent.parent


def get_all_configs(selection: str):
    """Get all files stored in root/config/ directory"""
    config_dir = project_root / "config"
    return [config_path for config_path in config_dir.glob(selection)]


def upload_to_hf(config_path: Path, username: str, collection: str = None):
    """Uploads model that corresponds to given config file to HuggingFace"""
    print("-----")
    print("Starting upload...")

    config_text = ""
    with open(config_path, "r") as f:
        config_text = "".join(f.readlines())

    wdb = WandbWrapper(str(config_path), mode="offline")
    model_name = wdb.get_hyperparameter("save_name")
    model_path = project_root / "models" / (model_name + ".pth")
    rms_path = project_root / "models" / (model_name + "_rms.npz")

    api = hf.HfApi()
    repo_name = model_name
    repo_id = f"{username}/{repo_name}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Skipping! Repository {repo_name} already exists.")

    print("Recording video...")
    reward, video_path = record_model_video(wdb, model_path)

    print("Building README...")
    readme = prepare_readme(
        wdb.get_hyperparameter("algorithm"),
        wdb.get_hyperparameter("environment"),
        config_text,
        reward
    )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"{model_name}.pth",
        repo_id=repo_id
    )
    api.upload_file(
        path_or_fileobj=rms_path,
        path_in_repo=f"{model_name}_rms.npz",
        repo_id=repo_id
    )
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.yaml",
        repo_id=repo_id
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id
    )

    while not video_path.exists():
        continue
    api.upload_file(
        path_or_fileobj=video_path,
        path_in_repo="replay.mp4",
        repo_id=repo_id
    )
    video_path.unlink()

    print(f"Done! The repository can be found at: https://huggingface.co/{repo_id}")

    if collection:
        collection_slug = f"{username}/{collection}"
        print(f"Adding to collection {collection_slug}...")
        try:
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="model",
                exists_ok=True
            )
        except Exception as e:
            print(f"Failed to add to collection {e}!")
    print("-----\n")
    wdb.finish()


def prepare_readme(alg: str, env: str, yaml: str, reward: float):
    """Prepares Readme text for HF page. The crazy indent is, sadly, needed."""
    readme_text = textwrap.dedent(f"""\
---
tags:
  - deep-reinforcement-learning
  - reinforcement-learning
library_name: pytorch
model-index:
  - name: {env}
    results:
      - task:
          type: reinforcement-learning
          name: reinforcement-learning
        dataset:
          name: {env}
          type: {env}
        metrics:
          - type: mean_reward
            value: {reward}
            name: mean_reward
---
# RoboDRL

## Model details
Algorithm: {alg}

Environment: {env}

Framework: PyTorch + custom implementation

## Used config
```yaml
{yaml.strip()}
""")
    return readme_text


def record_model_video(wdb: WandbWrapper, model_path: Path):
    """Records video of agent solving the environment for HF model page"""
    env_name = wdb.get_hyperparameter("environment")
    algorithm = wdb.get_hyperparameter("algorithm")
    video_folder = project_root / "outputs"

    env = EnvironmentManager(env_name, "rgb_array")
    env.build_continuous()
    env.build_video_recorder(video_folder=str(video_folder))

    if algorithm == "PPO Continuous":
        agent = PPOAgentContinuous(env, wdb)
    elif algorithm == "SAC":
        agent = SACAgent(env, wdb)
    elif algorithm == "TD3":
        agent = TD3Agent(env, wdb)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    try:
        agent.load_model(agent.actor, model_path)
    except:
        raise RuntimeError("Model not supported!")

    reward, _, _ = agent.play()
    return reward, project_root / "outputs" / "agent_video-episode-0.mp4"


if __name__ == "__main__":
    args = get_hf_args()
    configs = get_all_configs(args.selection)
    for config in configs:
        upload_to_hf(config, args.username, args.collection)
