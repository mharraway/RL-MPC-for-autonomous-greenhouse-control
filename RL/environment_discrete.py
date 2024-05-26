import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from SHARED.params import *
from SHARED.model import *
from SHARED.aux_functions import *
from functools import partial
from math import sin
from RL.helperFunctions import *

'''
              (0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8, 9,10,11)
agent_state = (y1,y2,y3,y4,u1,u2,u3,k, d1,d2,d3,d4)
'''

class greenhouseEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, normalize_obs = False):
        super(greenhouseEnv, self).__init__()
        
        self.action_space = spaces.Discrete(27)

        # min = np.concatenate([y_min,u_min,[0],d_min])
        # max = np.concatenate([y_max,u_max,[max_steps],d_max])
        min = np.zeros(12)
        max = np.ones(12)*10
        self.normalize_obs = normalize_obs
        
        self.observation_space = spaces.Box(low=-obs_max, high=obs_max, shape=(12,), dtype=np.float32)
        self.action_space.seed(GlobalSeed)
        
        
    def reset(self,seed=GlobalSeed,start_day=start_day):
        super().reset(seed=seed)
        
        
        #Model and Disturbances
        self.F, self.g = model_functions()
        start_T = (start_day-1)*86400 + 3600
        self.get_d = partial(get_disturbance,weather_data = weather_data,start_time=start_T,Np=1, dt=dT)
        
        #Internal States
        self.done = False
        self.k = 0
        self.x = np.copy(x0)
        self.y = self.g(x0).toarray().ravel()
        self.y_prev = np.zeros(4)
        
        d = self.get_d(0)
        self.temp_min , self.temp_max, self.temp_ref, self.c02_min, self.c02_max, self.c02_ref = Reference_function(d[0],MIN_TEMP_CONSTRAIN_NIGHT,MAX_TEMP_CONSTRAIN_NIGHT,MIN_C02_CONSTRAIN,MAX_C02_CONSTRAIN)

        
        #Observations
        y0 = self.y.copy()
        y0[0] = y0[0]           - self.y_prev[0]
        y0[1] = self.c02_ref    - y0[1]
        y0[2] = self.temp_ref   - y0[2]
        
        obs = np.concatenate([y0,u0.ravel(),[self.k],d.ravel()], dtype=np.float32)
        self.observation = obs

        return obs,{"output":self.y, "temp_min":self.temp_min,"temp_max":self.temp_max,"temp_ref":self.temp_ref,"c02_min":self.c02_min,"c02_max":self.c02_max,"c02_ref":self.c02_ref} 
        
        
    def step(self, action):
 
        #Current Observations
        obs = np.copy(self.observation)
        _y, u_prev, _k, d = deconstructObs(obs)
        
        #Get Control Input
        u = action2control(u_prev, action)

        
        #Step Internal State
        self.y_prev = self.y.copy()                                 #Store previous system outputs
        x_next = self.F(self.x,u,d)                                 #Obtain new system states
        x_next = np.clip(x_next.toarray().ravel(), x_min,x_max)     #Clip new system states to ensure stability
        self.x = x_next                                             #Store new system states
        self.y = self.g(x_next).toarray().ravel()                   #Obtain new system outputs
        self.k +=1                                                  #Increment time step
        d = self.get_d(self.k)                                      #Obtaine new disturbance
        
        
        #Obtain new min.max and reference traj
        y_next = self.y.copy()
        self.temp_min, self.temp_max, self.temp_ref, self.c02_min, self.c02_max, self.c02_ref = Reference_function(d[0],self.temp_min,self.temp_max,self.c02_min,self.c02_max)

        #Construct New observation
        y_next[0] = y_next[0]       - self.y_prev[0]
        y_next[1] = self.c02_ref    - y_next[1]
        y_next[2] = self.temp_ref   - y_next[2]
        
        obs_new = np.concatenate([y_next,u.ravel(),[self.k],d.ravel()], dtype=np.float32)
        self.observation = obs_new
        
        done = is_terminal(obs_new)
        reward = self.reward_function(obs,obs_new)
            
        if done:
            self.reset()
    
        return obs_new, reward, done, False, {"output":self.y, "temp_min":self.temp_min,"temp_max":self.temp_max,"temp_ref":self.temp_ref,"c02_min":self.c02_min,"c02_max":self.c02_max,"c02_ref":self.c02_ref} 
    

        
    def close(self):
        # ...
        pass
    
    def reward_function(self,old_obs,new_obs):
        y_old, _u_old, _k_old, d_old = deconstructObs(old_obs)
        y_new, u_new, _k_new, _d_new = deconstructObs(new_obs)
        
        reward = r_let*(y_new[0]) - (r_c02*u_new[0] + u_new[2]*r_q) #- u_old[1]*0.08
            
        penalties = 0
        
        
        y_c02 = self.c02_ref - y_new[1]
        # C02 constraints
        if (y_c02 < self.c02_min):
            penalties +=  0.1  * (self.c02_min - y_c02)**2
        elif y_c02 > self.c02_max:
            penalties +=  0.1  * (y_c02 - self.c02_max)**2
        else:
            penalties -=  pen_reward
            
        y_temp = self.temp_ref - y_new[2]            
        #Temp constraints           
        if (y_temp < self.temp_min):
            penalties +=  pen_temp  * (self.temp_min - y_temp)**2
        elif y_temp > self.temp_max:
            penalties +=  pen_temp  * (y_temp-self.temp_max)**2
        else:
            penalties -=  pen_reward     
        
        return reward - penalties

def action2control(u_prev:np.array, action):
    match action:
        case 0:
            action_3dim = np.array([0,0,0])
        case 1:
            action_3dim = np.array([0,1,0])
        case 2:
            action_3dim = np.array([0,1,1])
        case 3:
            action_3dim = np.array([0,-1,0])
        case 4:
            action_3dim = np.array([0,0,-1])
        case 5:
            action_3dim = np.array([0,-1,-1])
        case 6:
            action_3dim = np.array([0,1,-1])
        case 7:
            action_3dim = np.array([0,-1,1])
        case 8:
            action_3dim = np.array([0,0,1])
            
        case 9:
            action_3dim = np.array([-1,0,0])
        case 10:
            action_3dim = np.array([-1,1,0])
        case 11:
            action_3dim = np.array([-1,1,1])
        case 12:
            action_3dim = np.array([-1,-1,0])
        case 13:
            action_3dim = np.array([-1,0,-1])
        case 14:
            action_3dim = np.array([-1,-1,-1])
        case 15:
            action_3dim = np.array([-1,1,-1])
        case 16:
            action_3dim = np.array([-1,-1,1])
        case 17:
            action_3dim = np.array([-1,0,1])
            
        case 18:
            action_3dim = np.array([1,0,0])
        case 19:
            action_3dim = np.array([1,1,0])
        case 20:
            action_3dim = np.array([1,1,1])
        case 21:
            action_3dim = np.array([1,-1,0])
        case 22:
            action_3dim = np.array([1,0,-1])
        case 23:
            action_3dim = np.array([1,-1,-1])
        case 24:
            action_3dim = np.array([1,1,-1])
        case 25:
            action_3dim = np.array([1,-1,1])
        case 26:
            action_3dim = np.array([1,0,1])
            
        
    
    return np.clip(u_prev + action_3dim*delta_u, u_min, u_max) 




def is_terminal(observation)->bool:
    y, _u, k, _d= deconstructObs(observation)

    #np.any(y <= 0) or
    if ( k >= max_steps): 
        return True
    else:
        return False

def deconstructObs(observation):
    y = observation[0:4]
    u = observation[4:7]
    k = observation[7]
    d = observation[8:]
    return y, u, k, d

def normalizeObs(observation,min,max):
    obs_norm = 10*((observation - min)/(max-min))
    return np.array(obs_norm, dtype=np.float32)

def deNormalizeObs(obs_norm,min,max):
    obs = obs_norm*(max - min)/10 + min
    return obs

    