import os
import warnings


def run_episode_task(config_path, model_path, rendering,):
    """Run the agent in a separate process to avoid OpenGL conflicts and issues."""
    os.environ["MUJOCO_GL"] = "glfw"
    warnings.filterwarnings("ignore")
    from src.shared.wandb_wrapper import WandbWrapper
    from src.shared.environment_manager import EnvironmentManager
    from src.ppo.agent_continuous import PPOAgentContinuous
    from src.sac.agent import SACAgent
    from src.td3.agent import TD3Agent

    try:
        wdb = WandbWrapper(config_path, mode="disabled")

        env_name = wdb.get_hyperparameter("environment")
        algorithm = wdb.get_hyperparameter("algorithm")
        if rendering == "Human Rendering":
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        env = EnvironmentManager(env_name, render_mode)
        env.build_continuous()
        if rendering == "Video Rendering":
            env.build_video_recorder()

        if algorithm == "PPO Continuous":
            agent = PPOAgentContinuous(env, wdb)
        elif algorithm == "SAC":
            agent = SACAgent(env, wdb)
        elif algorithm == "TD3":
            agent = TD3Agent(env, wdb)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if "RANDOM POLICY" not in model_path:
            try:
                agent.load_model(agent.actor, model_path)
            except RuntimeError:
                print("Model loading failed!")

        reward, steps, info = agent.play()
        if type(reward) is tuple:
            reward = reward[0]

        return {"reward": float(reward), "steps": int(steps), "done": True}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'env' in locals():
            env.close()
