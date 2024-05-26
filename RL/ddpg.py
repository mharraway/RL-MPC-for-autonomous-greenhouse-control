import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import deque
from collections import namedtuple
from torch.autograd import Variable
import sys


#problem specifics
'''
u_min = [0,0,0]
u_max = [1.2, 7.5, 150]



action in [-1,1]^3
delta_u = 0.1*u_max
u(k) = clip (u(k-1), a(k).delta_u, u_min, u_max)

agent_state = [y1,y2,y3,y4,u1,u2,u3,k,d1,d2,d3,d4]]
'''



## ---------------------------------------------------Networks
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, agent_state,action, agent_next_state, reward, done):
        """Save a transition/ experience"""
        self.memory.append((agent_state, action, agent_next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        agent_states, actions, agent_next_states, rewards, terminals = zip(*batch)
        return agent_states, actions, agent_next_states, rewards, terminals
    
    def __len__(self):
        return len(self.memory)
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size,learning_rate):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)  #input layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  #input layer
        self.fc3 = nn.Linear(hidden_size, hidden_size) #hidden layer 1
        self.fc4 = nn.Linear(hidden_size, 1) #output layer
        self.fc5 = nn.Linear(action_dim,hidden_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,weight_decay=1e-5)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.cat([x, self.fc5(action)], dim=-1)
        # x = x + self.fc5(action)
        # x = torch.relu(x)
        # x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) #to get an action space between [-1,1]

        return x




## ---------------------------------------------------Agent
class DDPGagent:
    def __init__(self, state_dim = 12, action_dim = 3, critic_hidden_size=64, actor_hidden_size=64,learning_rate=1e-3, gamma=1, tau=1e-2, batch_size = 64, max_memory_size=100000):
        # Params
        self.num_actions = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Networks
        self.actor = Actor(state_dim, actor_hidden_size, action_dim, learning_rate)
        self.actor_target = Actor(state_dim, actor_hidden_size, action_dim, learning_rate)
        
        self.critic = Critic(state_dim, action_dim, critic_hidden_size, learning_rate)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden_size, learning_rate)
        
        # Training
        self.memory = ReplayMemory(max_memory_size)  
    
        self.update_target_networks(tau=1)
    
    
    # UnMapped action from policy network
    def get_action(self, obs:np.array, noise = 0.0) -> np.array:
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        action = self.actor(obs).detach().numpy()[0]
        return np.clip (action + noise * np.random.randn(3),-1,1)

    def store_experience(self, obs, action, next_obs, reward, done):
        self.memory.push(obs, action, next_obs, reward, done)
    
    def learn(self):
       
        obs, actions, next_obs, rewards, dones = self.memory.sample(self.batch_size)
        
        ## this might be very slow
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        
        #update critic
        self.critic.optimizer.zero_grad()
        
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q_values = self.critic_target(next_obs, next_actions)
            target_q_values = rewards + (1-dones) * self.gamma * target_q_values
  
        current_q_values = self.critic(obs, actions) # current Q values from current Q(Critic) network

        critic_loss = nn.MSELoss()(current_q_values, target_q_values)  #loss function
        critic_loss.backward() 
        self.critic.optimizer.step()

        #update critic
        self.actor.optimizer.zero_grad()
        # Actor loss
        policy_loss = -self.critic(obs, self.actor(obs)).mean()
        policy_loss.backward()
        self.actor.optimizer.step() # only optimizes the actors parameters

        self.update_target_networks()
        
        return critic_loss.item(), policy_loss.item()

        
        
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
            
        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))


 
