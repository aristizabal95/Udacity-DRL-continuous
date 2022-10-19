from agents.ddpg import DDPG, MemoryBuffer, ActorFeedForward, CriticFeedForward

from unityagents import UnityEnvironment
from tqdm import tqdm
import numpy as np
import json

def train(env, agent, num_epochs=300, epsilon=0.9, epsilon_decay=1e-5):
    # get the default brain
    brain_name = env.brain_names[0]
    loop = tqdm(range(num_epochs))
    best_reward = float("-inf")
    rewards_history = []

    for epoch in loop:
        env_info = env.reset(train_mode=True)[brain_name]
        epoch_cum_rewards = None
        while True:
            states = env_info.vector_observations
            actions = agent.act(states, epsilon=epsilon)
            env_info = env.step(actions.detach().numpy())[brain_name]
            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done
            for exp in zip(states, actions, rewards, next_states, dones):
                state, action, reward, next_state, done = exp
                agent.step(state, action, reward, next_state, done)

            epsilon = epsilon * (1 - epsilon_decay)

            if epoch_cum_rewards is None:
                epoch_cum_rewards = np.array(rewards)
            else:
                epoch_cum_rewards += np.array(rewards)

            if np.any(dones):
                break
        print(epoch_cum_rewards)
        epoch_reward = np.mean(epoch_cum_rewards)
        if epoch_reward > best_reward:
            best_reward = epoch_reward
            agent.save(f"agent_checkpoints".replace(".", "_"))
        rewards_history.append(epoch_reward)
        loop.set_description(f"Avg Reward: {round(np.mean(rewards_history[-100:]), 4)} | Epsilon: {round(epsilon, 3)}")

    with open("reward_history.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == '__main__':
    actor = ActorFeedForward([33, 32, 16, 8, 4])
    critic = CriticFeedForward([33, 16], [4, 8], [24, 16, 16, 1])
    membuffer = MemoryBuffer()
    agent = DDPG(actor, critic, membuffer)
    env = UnityEnvironment(file_name='Reacher.app')

    train(env, agent)