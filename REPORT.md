[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: reward_history.png "Rewards Plot"

# Project Report

## Task: Reacher
This project demonstrates how to use Deep Reinforcement Learning to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Solution: Deep Deterministic Policy Gradient
To solve this environment, an actor-critic agent was created following the DDPG framework. This agent contains a total of 4 Feed Forward Neural Networks:
- **Actor Networks**: The actor is a normal FFNN with 3 hidden layers, and a Tanh activation at the end to restrict the range between -1 and 1. The actor uses 2 networks for learning stability.
- **Critic Networks**: The critic is composed of a FFNN with two separate inputs, one for the state and another for the action. These inputs are at first treated separately, but later concatenated to generate the final value prediction.

### Experience Replay
This implementation includes a memory buffer for storing experience. It is then used to sample batches of data to train the networks. The experience from all robotic arms is added to a single experience replay. By default, the memory buffer has a max capacity of `16384`.

### Exploration-Exploitation
At first, exploration was done through adding noise to the actions output. Following [OpenAI's Spinning Up explanation](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation), gaussian noise was added, and the magnitude of noise decreased during the learning procedure. This process did not yield great results, and learning did not converge. Because of this, I decided to go with an epsilon-greedy method. Noise magnitude is therefore fixed, and only added to the action vector with a probability of `epsilon`. 

Training begins with `epsilon=0.9` and gradually decreases after every training step.

### Results

![Rewards Plot][Image2]
The graph above demonstrates the agent's performance per episode, as well as the 100 Episode Average Reward. It can be seen that performance quickly increased after episode 100, going from around 4 to 30 Reward in a span of 50 episodes. After that, performance stayed between 30 and 40, with a dip to around 25 Reward near episode 250. Looking at the 100 episode average reward, we can see that the environment could be considered solved at around episode 225.

### Next Steps

There was little need for experimentation with the DDPG implementation used here. Both the architecture and hyperparameters were found without much exploration. A few experiments that could be interesting for the future are:

- **Explore other hyperparameters:** See how sensible the agent is against different hyperparameters. For example, different learning rates, alpha, gamma, or hidden layers/sizes.
- **Explore different architectures:** Try out PPO, A4C, or some of the other agents recommended for continuous-action tasks.
- **Use the same implementation for the Crawler Environment:** The Crawler Environment is a recommended optional exercise to work through after solving the Reacher Environment. It'd be interesting to see if this implementation (with neural networks IO adjusted accordingly) can solve this environment out-of-the-box.