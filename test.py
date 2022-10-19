from agents.ddpg import DDPG, MemoryBuffer, ActorFeedForward, CriticFeedForward

from unityagents import UnityEnvironment
import numpy as np

def test(env, agent, num_runs=5):
    # get the default brain
    brain_name = env.brain_names[0]

    for run_idx in range(1, num_runs+1):
        env_info = env.reset(train_mode=False)[brain_name]
        epoch_cum_rewards = None
        while True:
            states = env_info.vector_observations
            actions = agent.act(states)
            env_info = env.step(actions.detach().numpy())[brain_name]
            rewards = env_info.rewards
            dones = env_info.local_done

            if epoch_cum_rewards is None:
                epoch_cum_rewards = np.array(rewards)
            else:
                epoch_cum_rewards += np.array(rewards)

            if np.any(dones):
                break
        epoch_reward = np.mean(epoch_cum_rewards)
        print(f"Run {run_idx} avg performance: {epoch_reward}")

if __name__ == '__main__':
    actor = ActorFeedForward([33, 32, 16, 8, 4])
    critic = CriticFeedForward([33, 16], [4, 8], [24, 16, 16, 1])
    membuffer = MemoryBuffer()
    agent = DDPG(actor, critic, membuffer)
    env = UnityEnvironment(file_name='Reacher.app')
    agent.load("agent_checkpoints")

    test(env, agent)