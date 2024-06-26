import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import casadi
import numpy as np
from typing import Union
from collections import deque
import random
from SHARED.params import *
from SHARED.model import *
from functools import partial
from tqdm import tqdm
from RL.environment import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from SHARED.display_trajectories import *
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import LinearLR
from numpy import savetxt
from stable_baselines3 import PPO, SAC,TD3
from torch.utils.data import Dataset, DataLoader

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, next_state, reward, done):
        """Save a transition/ experience"""
        self.memory.append((state, next_state,reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, next_states, rewards, dones = zip(*batch)
        return states, next_states, rewards, dones
    
    def __len__(self):
        return len(self.memory)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        now, next,reward,done = self.data[idx]
        return torch.tensor(now,dtype=torch.float32),torch.tensor(next,dtype=torch.float32),reward,done
    

class neural_net (nn.Module):
    def __init__(self,input_dim, hidden_dim) -> None:
        super(neural_net,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, 1)  
        # self.relu = nn.ReLU()  # ReLU activation function
        self.act_fn = nn.Tanh()
        
    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def weights_reset(self):
        for layer in self.children():
            if hasattr(layer, 'weight'):
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)
    
class value_function_TD():
    def __init__(self, input_dim, hidden_dim, learning_rate, batch_size, buffer_size = max_steps, tau = 0.001, train_episodes=1, model_path ="", env_path = ""):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        

        self.tau = tau
        
        self.neural_net = neural_net(input_dim,hidden_dim)
        # self.neural_net.weights_reset()
        self.target_neural_net = neural_net(input_dim,hidden_dim)
        self.train_episodes = train_episodes
        
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate)
        self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=train_episodes*max_steps)
        
        
        self.memory = ReplayMemory(buffer_size) 
        
        train_mem_size = (int) (0.8*buffer_size)
        val_mem_size = (int) (0.2*buffer_size)
        
        self.training_memory = ReplayMemory(train_mem_size) 
        self.validation_memory = ReplayMemory(val_mem_size) 
        
        i = 1
        log_path = "vf_logs_TD/"
        path = log_path+str(i)
        while os.path.exists(path):
            i+=1
            path = log_path+str(i)
            
        self.writer = SummaryWriter(path)  # Initialize TensorBoard writer
        self.model_path = model_path
        self.env_path = env_path
        
        self.update_target_networks(tau=1)
    
    def store_transition(self,state,next_state, reward, done):
        self.memory.push(state, next_state, reward, done)
    
    def learn(self, global_step = 0):
        
        for i,(states,next_states, rewards,dones) in enumerate(self.data_loader_train,0):
        
        #obtain transition tuples
            rewards         = rewards.to(torch.float32).unsqueeze(1)
            # print (dones)
            dones           = dones.int().to(torch.float32).unsqueeze(1)
            #optimizer
            self.optimizer.zero_grad()
            
            # Calculate TD error
            current_state_values = self.neural_net(states)
            next_state_values = self.target_neural_net(next_states).detach()        
            td_target = rewards + (1 - dones) * next_state_values
            
            
            #Calculating Loss
            mse_loss = nn.MSELoss()
            loss = mse_loss(current_state_values, td_target) 
            
            # Logging loss for visualization using TensorFlow
            log_step = (global_step) + (i)
            self.writer.add_scalar('Train/loss', loss.item(),global_step=log_step)
            self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'],global_step=global_step)
            loss.backward()
            self.optimizer.step()
            
            self.scheduler.step() 
            if log_step%10 == 0:
                self.validate(global_step=log_step)
    
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
                           
        for target_param, param in zip(self.target_neural_net.parameters(), self.neural_net.parameters()):
            target_param.data.copy_(param.data * tau  + target_param.data * (1.0 - tau))     
        
    
    def prepare_data(self):
        split_ratio = 1
        
        random.shuffle(self.memory.memory)
        split_point = int(len(self.memory) * split_ratio)
        self.training_memory.memory = deque(list(self.memory.memory )[:split_point])
        self.validation_memory.memory  = deque(list(self.memory.memory )[split_point:])
        
       
        self.train_dataset  = MyDataset(self.training_memory.memory )
        self.validate_dataset  = MyDataset(self.validation_memory.memory)
        
        self.data_loader_train = DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True)
        self.data_loader_validate = DataLoader(self.validate_dataset,batch_size=self.batch_size, shuffle=True)
        
    
    def train(self, epochs):

        self.prepare_data()

        total_iterations = epochs*len(self.data_loader_train)
        self.scheduler.total_iters = total_iterations
        
        print (f"Total training iterations:{total_iterations}")
        self.neural_net.train()
        
        for i in range(epochs):
            self.learn(global_step = i*len(self.data_loader_train))

            self.update_target_networks()
        self.neural_net.eval()         
        self.writer.close() 
    
    def sim_with_agent(self, extra_samples = 0):
        self.neural_net.eval()
        model_path = self.model_path
        env_path = self.env_path
        gamma = 1 #this does not really matter
        
        env = greenhouseEnv()
        env.random_starts = False
        env.stochastic = False
                
        #Creating trained normalized environment
        env_norm = greenhouseEnv()
        env_norm = DummyVecEnv([lambda: env_norm])
        env_norm = VecNormalize(env_norm, norm_obs = True, norm_reward = False, clip_obs = 10.,gamma=gamma)
        env_norm = env_norm.load(env_path,env_norm)
        env_norm.training = False
            
        #Create controller
        agent = SAC.load(model_path,env=env_norm)

        obs_log, obs_next_log, pairs= [],[],[]
        obs_now, info = env.reset()
        done = False
        nominal_traj = []
        
        for k in tqdm(range(max_steps)):
            nominal_traj.append(obs_now)
            for s in range(extra_samples):
            #Construct random observation
                env.freeze()
                obs_now_random,_,_ = env.sample_obs(0.05)
                obs_now_norm_random = env_norm.normalize_obs(obs_now_random)
                obs_log.append(obs_now_random)
                #get action
                random_action, _states = agent.predict(obs_now_norm_random, deterministic=True)
                #Step in environment
                obs_next_random, reward_random, done_random, _,info_random = env.step(random_action) #Returns the observation for RL agent  
                obs_next_norm_random = env_norm.normalize_obs(obs_next_random)
                self.store_transition(obs_now_norm_random, obs_next_norm_random,reward_random,done_random)
                
                obs_next_log.append(obs_next_random)
                pairs.append((obs_now_random,obs_next_random))
                env.unfreeze()
                
                            #Get action
            obs_now_norm = env_norm.normalize_obs(obs_now)
            action, _states = agent.predict(obs_now_norm, deterministic=True)            
            #Step in environment
            obs_next, reward, done, _,info = env.step(action)
            obs_next_norm = env_norm.normalize_obs(obs_next)
            
            obs_log.append(obs_now)
            
             
            self.store_transition(obs_now_norm, obs_next_norm,reward,done)

            
            #Next time step  
            obs_now = obs_next
        return obs_log, obs_next_log,pairs, np.vstack(nominal_traj)
        
    def test_with_agent(self):
        self.neural_net.eval()
        model_path = self.model_path
        env_path = self.env_path
        gamma = 1 #this does not really matter
        
        env = greenhouseEnv()
        env.random_starts = False
        env.stochastic = False
                
        #Creating trained normalized environment
        env_norm = greenhouseEnv()
        env_norm = DummyVecEnv([lambda: env_norm])
        env_norm = VecNormalize(env_norm, norm_obs = True, norm_reward = False, clip_obs = 10.,gamma=gamma)
        env_norm = env_norm.load(env_path,env_norm)
        env_norm.training = False
            
        #Create controller
        agent = SAC.load(model_path,env=env_norm)

        Y_log, D_log, U_log,value_log,cum_reward_log, X_log = [],[],[],[],[],[]
        total_reward = 0

        obs_now, info = env.reset()
        x_now = info['x']

        for k in tqdm(range(max_steps)):
            
            #Get action
            obs_now_norm = env_norm.normalize_obs(obs_now)
            action, _states = agent.predict(obs_now_norm, deterministic=True)            
            #Step in environment
            obs_next, reward, done, _,info = env.step(action)
            obs_next_norm = env_norm.normalize_obs(obs_next)
            
            #Evaluate value from value function
            value = self.evaluate_value(obs_next_norm) 
                        
            total_reward += reward
            Y_log.append(info['output'])
            X_log.append(info['x'])
            D_log.append(obs_next[8:])
            U_log.append(obs_next[4:7])
            cum_reward_log.append(total_reward)
            value_log.append(value)

            obs_now = obs_next
            # state_now = state_next

        U_log = np.array(U_log)
        Y_log = np.array(Y_log)
        D_log = np.array(D_log)
        to_show_reward = np.array(cum_reward_log)
        to_show_values = np.array(value_log)

        print_metrics(Y_log, U_log, D_log,vf = to_show_values,rewards=to_show_reward, day_range=(0,40))
        
        return Y_log,U_log, X_log
          
        
        
    def reset(self):
        self.neural_net.weights_reset()
        self.update_target_networks(tau = 1)
        
    def evaluate_value(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        value = self.neural_net(obs_tensor).detach().numpy()[0][0]
        return value    
    
    def validate(self, global_step = 0):
        
        self.neural_net.eval()
        
        for i,(states,next_states, rewards,dones) in enumerate(self.data_loader_validate,0):
            #obtain transition tuples
            # states, next_states, rewards, dones = self.validation_memory.sample(self.batch_size)
            # states          = torch.tensor(states, dtype=torch.float32)
            # next_states     = torch.tensor(next_states, dtype=torch.float32)
            rewards         = rewards.to(torch.float32).unsqueeze(1)
            # print (dones)
            dones           = dones.int().to(torch.float32).unsqueeze(1)
            
            #optimizer
                    
            # Calculate TD error
            current_state_values = self.neural_net(states).detach()
            next_state_values = self.target_neural_net(next_states).detach()        
            td_target = rewards + (1 - dones) * next_state_values
            
            
            #Calculating Loss
            mse_loss = nn.MSELoss()
            loss = mse_loss(current_state_values, td_target) 
            
            # Logging validation loss for visualization using TensorFlow
            self.writer.add_scalar('Train/validation', loss.item(),global_step=global_step)
            break
        self.neural_net.train()
              
    def sim_on_mpc(self,extra_samples = 0, spread = 0.1):
        self.neural_net.eval()
        
        #Env setup
        env = greenhouseEnv()
        env.random_starts = False
        env.stochastic = False
        
        #Controller Setup
        N = (3600*1)//dT
        MPC = casadi.Function.load('MPC/MPC_controller_1hr')
        casadi.DM.set_precision(15)
        env.using_mpc = True   
        get_d = partial(get_disturbance,weather_data = weather_data,start_time=start_time,Np=N, dt=dT)

        obs_log = []

        obs_now, info = env.reset()
        x_now = info["x"]
        u_opt = obs_now[4:7]       
        for k in tqdm(range(max_steps)):
            #Get weather prediction
            d_now = get_d(k)  
            
            #Get optimal action and trajectories
            u_opt,u_traj,x_traj = MPC(x_now,d_now,u_opt)
            u_opt  = np.array(u_opt)      
            u_traj = np.array(u_traj)
            x_traj = np.array(x_traj)
            u_opt = np.clip(u_opt,u_min.reshape(3,1),u_max.reshape(3,1))
            
            #Step in env
            obs_next, reward, done, _,info = env.step(u_opt)
            x_next = info["x"]
            
            obs_log.append(obs_next)
            obs_now_norm = normalizeObs(obs_now,obs_min,obs_max)
            obs_next_norm= normalizeObs(obs_next,obs_min,obs_max)
            self.store_transition(obs_now_norm,obs_next_norm,reward,done)
            
            for s in range(extra_samples):
            #Construct random observation
                env.freeze()
                obs_now_random,x_now_random,u_now_random = env.sample_obs(spread)
                obs_log.append(obs_now_random)
                #get action
                u_opt_random,_,_ = MPC(x_now_random,d_now,u_now_random)
                u_opt_random  = np.array(u_opt_random)  
                u_opt_random = np.clip(u_opt_random,u_min.reshape(3,1),u_max.reshape(3,1))
                #Step in environment
                obs_next_random, reward_random, done_random, _,info_random = env.step(u_opt_random) 

                obs_now_norm_random = normalizeObs(obs_now_random,obs_min,obs_max)
                obs_next_norm_random= normalizeObs(obs_next_random,obs_min,obs_max)
                self.store_transition(obs_now_norm_random, obs_next_norm_random,reward_random,done_random)
                env.unfreeze()

            obs_now = obs_next
            x_now = x_next
         
        return obs_log
    
    def test_with_mpc(self):
        self.neural_net.eval()
        #Env setup
        env = greenhouseEnv()
        env.random_starts = False
        env.stochastic = False

        #Controller Setup
        N = (3600*1)//dT
        MPC = casadi.Function.load('MPC/MPC_controller_1hr')
        casadi.DM.set_precision(15)
        env.using_mpc = True   
        get_d = partial(get_disturbance,weather_data = weather_data,start_time=start_time,Np=N, dt=dT)

        Y_log, D_log, U_log,value_log,cum_reward_log = [],[],[],[],[]
        total_reward = 0

        obs_now, info = env.reset()
        x_now = info["x"]
        u_opt = obs_now[4:7]

        for k in tqdm(range(max_steps)):
            #Get weather prediction
            d_now = get_d(k)  
            #Get optimal action and trajectories
            u_opt,u_traj,x_traj = MPC(x_now,d_now,u_opt)
            u_opt  = np.array(u_opt)      
            u_traj = np.array(u_traj)
            x_traj = np.array(x_traj)
            u_opt = np.clip(u_opt,u_min.reshape(3,1),u_max.reshape(3,1))
            
            #Step in env
            obs_next, reward, done, _,info = env.step(u_opt)
            x_next = info["x"]
            
            #Logging
            obs_next_norm= normalizeObs(obs_next,obs_min,obs_max)
            value = self.evaluate_value(obs_next_norm)
                        
            total_reward += reward
            Y_log.append(info['output'])
            D_log.append(obs_next[8:])
            U_log.append(obs_next[4:7])
            cum_reward_log.append(total_reward)
            value_log.append(value)

            obs_now = obs_next
            x_now = x_next
            # state_now = state_next

        U_log = np.array(U_log)
        Y_log = np.array(Y_log)
        D_log = np.array(D_log)
        to_show_reward = np.array(cum_reward_log)
        to_show_values = np.array(value_log)

        print_metrics(Y_log, U_log, D_log,vf = to_show_values,rewards=to_show_reward, day_range=(0,40))

        
        
       