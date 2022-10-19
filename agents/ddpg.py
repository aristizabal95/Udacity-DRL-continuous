import os
import yaml
import copy
from collections import namedtuple, deque
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class DDPG:
    def __init__(self, actor, critic, membuff, init_dict={}):
        self.set_params(init_dict)

        self.actor_local = actor
        self.critic_local = critic
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)

        self.loss_fn = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr, weight_decay=1.e-5)
        self.membuff = membuff
        self.update_step = 0

    def set_params(self, init_dict):
        self.batch_size = init_dict.get("batch_size", 512)
        self.update_rate = init_dict.get("update_rate", 128)
        self.discount = init_dict.get("discount", 0.99)
        self.alpha = init_dict.get("alpha", 0.001)
        self.device = init_dict.get("device", "cpu")
        self.actor_lr = init_dict.get("actor_lr", 0.001)
        self.critic_lr = init_dict.get("critic_lr", 0.001)

    def act(self, state, epsilon=0.0, noise_amount=1.0):
        state = torch.tensor(state, dtype=torch.float32)
        actions = self.actor_local(state)
        random_action = np.random.choice([1, 0], size=actions.shape, p=[epsilon, 1-epsilon])
        random_action = torch.tensor(random_action, requires_grad=False).float()
        noise = np.random.normal(scale=noise_amount/3, size=actions.shape) # / 3 so 99% of noise falls between -1 and 1
        noise = torch.tensor(noise, requires_grad=False).float()
        actions = (1 - random_action) * actions + random_action * noise
        return torch.clamp(actions, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.membuff.add(state, action, reward, next_state, done)

        should_update = self.update_step % self.update_rate == 0
        sufficient_experience = len(self.membuff) >= self.batch_size

        if sufficient_experience and should_update:
            batch = self.membuff.sample(self.batch_size)
            self.learn(batch)

        self.update_step += 1

    def learn(self, experience_batch):
        # Decompose the experience into tensors
        states, actions, rewards, next_states, dones = experience_batch
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        # Train the critic with the TD Estimate
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
        value_estimate = rewards + (1 - dones) * self.discount * q_next
        value_pred = self.critic_local(states, actions)
        critic_loss = self.loss_fn(value_pred, value_estimate)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train the actor by maximizing the critics estimate
        pred_actions = self.actor_local(states)
        actor_loss = -self.critic_target(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target, self.alpha)
        self.soft_update(self.actor_local, self.actor_target, self.alpha)

    def soft_update(self, local_model, target_model, alpha):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(alpha * local_param.data + (1.0-alpha) * target_param.data)

    def save(self, folder_path):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        critic_local_ckp = os.path.join(folder_path, "critic_local.pth")
        critic_target_ckp = os.path.join(folder_path, "critic_target.pth")
        actor_local_ckp = os.path.join(folder_path, "actor_local.pth")
        actor_target_ckp = os.path.join(folder_path, "actor_target.pth")

        torch.save(self.critic_local, critic_local_ckp)
        torch.save(self.critic_target, critic_target_ckp)
        torch.save(self.actor_local, actor_local_ckp)
        torch.save(self.actor_target, actor_target_ckp)

        params_dict = {
            "batch_size": self.batch_size,
            "update_rate": self.update_rate,
            "discount": self.discount,
            "alpha": self.alpha,
            "device": self.device,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
        }

        params_path = os.path.join(folder_path, "params.yaml")
        with open(params_path, "w") as f:
            yaml.dump(params_dict, f)

    def load(self, folder_path):
        critic_local_ckp = os.path.join(folder_path, "critic_local.pth")
        critic_target_ckp = os.path.join(folder_path, "critic_target.pth")
        actor_local_ckp = os.path.join(folder_path, "actor_local.pth")
        actor_target_ckp = os.path.join(folder_path, "actor_target.pth")
        
        self.critic_local = torch.load(critic_local_ckp)
        self.critic_target = torch.load(critic_target_ckp)
        self.actor_local = torch.load(actor_local_ckp)
        self.actor_target = torch.load(actor_target_ckp)

        params_path = os.path.join(folder_path, "params.yaml")
        with open(params_path, "r") as f:
            params_dict = yaml.safe_load(f)

        self.set_params(params_dict)



class MemoryBuffer:
    def __init__(self, max_size=16384):
        self.max_size = max_size
        self.buffer = deque([], maxlen=self.max_size)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        action = action.detach().numpy()
        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        exp_batch = random.sample(self.buffer, batch_size)
        # convert to tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for exp in exp_batch:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones).float()

        return states, actions, rewards, next_states, dones


class FeedForward(nn.Module):
    def __init__(self, layers_sizes, device="cpu"):
        super(FeedForward, self).__init__()
        self.device = device
        layers_io = list(zip(layers_sizes, layers_sizes[1:]))
        layers = []
        for i, o in layers_io[:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(*layers_io[-1]))
        self.layers = nn.ModuleList(layers).to(self.device)

    def forward(self, x):
        x = x.to(self.device).float()
        for layer in self.layers:
            x = layer(x)
        return x


class ActorFeedForward(nn.Module):
    def __init__(self, layers_sizes, device="cpu"):
        super(ActorFeedForward, self).__init__()
        self.model = FeedForward(layers_sizes, device)

    def forward(self, x):
        out = self.model(x)
        return F.tanh(out)


class CriticFeedForward(nn.Module):
    def __init__(self, state_layers, action_layers, head_layers, device="cpu"):
        super(CriticFeedForward, self).__init__()
        assert head_layers[0] == state_layers[-1] + action_layers[-1], "Layers mismatch"
        
        self.state_model = FeedForward(state_layers, device)
        self.action_model = FeedForward(action_layers, device)
        self.head_model = FeedForward(head_layers, device)

    def forward(self, state, action):
        state_emb = self.state_model(state)
        action_emb = self.action_model(action)
        head_input = torch.cat((state_emb, action_emb), dim=1)
        return self.head_model(head_input)