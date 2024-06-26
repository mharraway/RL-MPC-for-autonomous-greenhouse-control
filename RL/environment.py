import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from SHARED.params import *
from SHARED.model import *
from SHARED.aux_functions import *
from functools import partial
from math import sin
from RL.helperFunctions import *
from SHARED.setup import *
import copy

'''
    (0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8, 9,10,11,12)
S = (*y1_prev,y1,y2,y3,u1,u2,u3,k, d1,d2,d3,d4)
'''

class greenhouseEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, use_growth_dif  = False, stochastic = False, using_mpc = False, random_starts = False):
        super(greenhouseEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space.seed(GlobalSeed)
        self.k = 0
        self.random_starts = random_starts
        self.using_mpc = using_mpc
        self.stochastic = stochastic
        self.use_growth_dif = use_growth_dif
        
        
        
    def reset(self,seed=GlobalSeed,start_day=start_day):
        super().reset(seed=seed)
        
        
        #Model and Disturbances
        self.F, self.g = model_functions()
        self.rl_reward = partial(reward_function, return_type = "float")
        self.penalty_reward = partial(penalty_function)
        
        start_T = (start_day-1)*86400 + 3600
        self.get_d = partial(get_disturbance,weather_data = weather_data,start_time=start_T,Np=1, dt=dT)
            
        #Internal States
        self.done = False
        
        self.k = 0
        self.x = np.copy(x0)
        self.x_prev = np.copy(x0)
        
        
        self.u = np.copy(u0)
        self.y = self.g(np.copy(x0), nominal_params).toarray().ravel()
        self.y_prev = np.copy(self.y)           
        
        d = self.get_d(self.k)
        
        #Observations
        y0 = np.copy(self.y)
        
        #This is used for the value function training
        vf_obs =  np.concatenate([y0,u0.ravel(),[self.k],d.ravel()], dtype=np.float32)
        
        #Used for actual agent training
        if self.use_growth_dif:
            y0[0] = y0[0] - self.y_prev[0] ###----> growth difference in OBS
            
        # obs = np.concatenate([[self.y_prev[0]],y0,self.u.ravel(),[self.k],d.ravel()], dtype=np.float32)
        obs = np.concatenate([y0,self.u.ravel(),[self.k],d.ravel()], dtype=np.float32)
        self.observation = obs
        return obs,{"x":self.x, "output":self.y, "vf_obs": vf_obs, "vf_reward" : 0} 
        
        
    def step(self, action):
 
        #Current Observations
        obs = np.copy(self.observation)
        # _y1_prev,_y, u_prev, _k, d = deconstructObs(obs)
        _y, u_prev, _k, d = deconstructObs(obs)
        
        #Get Control Input
        if self.using_mpc:
            u = action  
        else:  
            u = action2control(u_prev, action)

        self.u = u
        #Step Internal State
        self.y_prev = np.copy(self.y)                                 #Store previous system outputs
        self.x_prev = np.copy(self.x)
        sys_params = noisy.parametric_uncertainty() if self.stochastic else nominal_params
        x_next = self.F(self.x,u,d,sys_params).toarray().ravel()                                 #Obtain new system states
        # self.x = noisy.add_noise(self.x, x_next) if self.stochastic else x_next
        self.x = x_next
        self.y = self.g(x_next, sys_params).toarray().ravel()                   #Obtain new system outputs
        self.k +=1                                                  #Increment time step
        d = self.get_d(self.k)                                      #Obtaine new disturbance
        
        
        #Obtain new min.max and reference traj
        y_next = np.copy(self.y)

        #Used for vf training
        vf_obs = np.concatenate([y_next,u.ravel(),[self.k],d.ravel()], dtype=np.float32)
        vf_reward = 0
        
        #Used for actual agent training
        growth = y_next[0]  - self.y_prev[0]
        if self.use_growth_dif:
            y_next[0] = growth

        # obs_new = np.concatenate([[self.y_prev[0]],y_next,u.ravel(),[self.k],d.ravel()], dtype=np.float32)
        obs_new = np.concatenate([y_next,u.ravel(),[self.k],d.ravel()], dtype=np.float32)
        self.observation = obs_new
        
        done = is_terminal(np.copy(obs_new))
        reward = self.rl_reward(delta_drymass = growth , control_inputs = u.ravel())
        reward -= self.penalty_reward(outputs2constrain = [y_next[1], y_next[2], y_next[3]],
                                constraint_mins = [C02_MIN_CONSTRAIN_MPC,TEMP_MIN_CONSTRAIN_MPC, HUM_MIN_CONSTRAIN],
                                constraint_maxs = [C02_MAX_CONSTRAIN_MPC, TEMP_MAX_CONSTRAIN_MPC, HUM_MAX_CONSTRAIN])
        
     
        return obs_new, reward, done, False, {"x":self.x,"output":self.y, "vf_obs": vf_obs,"vf_reward": vf_reward} 
    
    def freeze(self):
        self.freeze_k           = copy.deepcopy(self.k)
        self.freeze_x           = copy.deepcopy(self.x)
        self.freeze_observation = copy.deepcopy(self.observation)
        self.freeze_y           = copy.deepcopy(self.y)
        self.freeze_x_prev      = copy.deepcopy(self.x_prev)
        self.freeze_y_prev      = copy.deepcopy(self.y_prev)
        self.freeze_done        = copy.deepcopy(self.done)
        self.freeze_u           = copy.deepcopy(self.u)
        
    def unfreeze(self):
        self.k              = copy.deepcopy(self.freeze_k)
        self.x              = copy.deepcopy(self.freeze_x)
        self.x_prev         = copy.deepcopy(self.freeze_x_prev)
        self.observation    = copy.deepcopy(self.freeze_observation)
        self.y              = copy.deepcopy(self.freeze_y)
        self.y_prev         = copy.deepcopy(self.freeze_y_prev)
        self.done           = copy.deepcopy(self.freeze_done)
        self.u              = copy.deepcopy(self.freeze_u)
        
    def close(self):
        # ...
        pass
    
    def sample_obs(self,std=0.5):
        '''agent_state = (y1,y2,y3,y4,u1,u2,u3,k, d1,d2,d3,d4)'''
        d = self.get_d(self.k) 
        
        random_x = np.zeros(4,)    
        random_u = np.zeros(3,)   
        
        #Drymass   
        # random_x[0] =  np.random.normal(loc=self.x.ravel()[0], scale = std*self.x.ravel()[0])
        bounds_min = self.x.ravel()[0]*(1-0.8) - 0.01
        bounds_max = self.x.ravel()[0]*(1+0.7) + 0.01
        
        # bounds_min = self.x.ravel()[0]*(1-0.1) - 0.01
        # bounds_max = self.x.ravel()[0]*(1+0.1) + 0.01
        bounds_min = np.maximum(bounds_min,x_min[0])
        bounds_max = np.minimum(bounds_max,x_max[0])
        random_x[0] =  np.random.uniform(bounds_min,bounds_max)
    
        #Temp
        max_temp_traj = 30
        min_temp_traj = 7
        random_x[2] =  np.random.uniform(min_temp_traj,max_temp_traj)
        # bounds_min = self.x.ravel()[2]- (std)* (max_temp_traj-min_temp_traj)/2
        # bounds_max = self.x.ravel()[2]+ (std)* (max_temp_traj-min_temp_traj)/2
        # bounds_min = np.maximum(bounds_min,min_temp_traj)
        # bounds_max = np.minimum(bounds_max,max_temp_traj)
        # random_x[2] =  np.random.uniform(bounds_min,bounds_max)
        
        #C02
        
        # bounds_min = self.y.ravel()[1]- (std)*(1800-400)/2
        # bounds_max = self.y.ravel()[1]+ (std)*(1800-400)/2
        # bounds_min = np.maximum(bounds_min,400)
        # bounds_max = np.minimum(bounds_max,1800)        
        # random_x[1] =  np.random.uniform(co2ppm2dens(random_x[2],bounds_min),co2ppm2dens(random_x[2],bounds_max))
        random_x[1] =  np.random.uniform(co2ppm2dens(random_x[2],400),co2ppm2dens(random_x[2],1800))
        
        #Hum
        # bounds_min = self.y.ravel()[3]- (std)*(100-50)/2
        # bounds_max = self.y.ravel()[3]+ (std)*(100-50)/2
        # bounds_min = np.maximum(bounds_min, 50)
        # bounds_max = np.minimum(bounds_max, 100)  
        # random_x[3] =  np.random.uniform(rh2vaporDens(random_x[2],bounds_min),rh2vaporDens(random_x[2],bounds_max))
        random_x[3] =  np.random.uniform(rh2vaporDens(random_x[2],50),rh2vaporDens(random_x[2],100))
          
        random_x = np.clip(random_x,x_min,x_max)
        random_y = self.g(random_x, nominal_params).toarray().ravel() 
        
        
         
        bounds_min = self.u.ravel()[0]- (std)*(u_max[0]-u_min[0])/2
        bounds_max = self.u.ravel()[0]+ (std)*(u_max[0]-u_min[0])/2
        bounds_min = np.maximum(bounds_min, u_min[0])
        bounds_max = np.minimum(bounds_max, u_max[0])  
        random_u[0] = np.random.uniform(bounds_min,bounds_max)
        
        bounds_min = self.u.ravel()[1]- (std)*(u_max[1]-u_min[1])/2
        bounds_max = self.u.ravel()[1]+ (std)*(u_max[1]-u_min[1])/2
        bounds_min = np.maximum(bounds_min, u_min[1])
        bounds_max = np.minimum(bounds_max, u_max[1])  
        random_u[1] = np.random.uniform(bounds_min,bounds_max)
        
        # bounds_min = self.u.ravel()[2]- (std)*(u_max[2]-u_min[2])/2
        # bounds_max = self.u.ravel()[2]+ (std)*(u_max[2]-u_min[2])/2
        # bounds_min = np.maximum(bounds_min, u_min[2])
        # bounds_max = np.minimum(bounds_max, u_max[2])  
        # random_u[2] = np.random.uniform(bounds_min,bounds_max)
        
        random_u = np.random.uniform(u_min,u_max)
        # random_u = np.random.uniform(loc = self.u.ravel(), scale = std*self.u.ravel())
        random_u = np.clip(random_u, u_min, u_max)
        
        if self.use_growth_dif:
            random_y[0] = random_y[0]  - self.y_prev[0]  
        
        random_obs =  np.concatenate([random_y,random_u,[self.k],d.ravel()], dtype=np.float32)
        
        self.x = np.copy(random_x)
        self.y = self.g(random_x, nominal_params).toarray().ravel()  
        self.observation = np.copy(random_obs)
        self.u = np.copy(random_u)
        
        return random_obs,random_x,random_u
    
    def get_x(self):
        return self.x
    
    def get_obs(self):
        return self.observation
    
    def set_env_state(self,x,x_prev,u,k):
        self.x = np.copy(x)
        self.x_prev = np.copy(x_prev)
        self.y = self.g(x, nominal_params).toarray().ravel()
        self.y_prev = self.g(x_prev, nominal_params).toarray().ravel()
        y = np.copy(self.y)
        
        if self.use_growth_dif:
            y[0] = y[0]  - self.y_prev[0] 
        
        self.k = k
        self.u = np.copy(u).ravel()
        d = self.get_d(k).ravel()     
        self.observation = np.concatenate([y,self.u,[self.k],d], dtype=np.float32)
        
    def step_with_agent(self,agent,env_norm):
        obs_norm = env_norm.normalize_obs(self.observation)
        action, _states = agent.predict(obs_norm, deterministic=True)
        obs_next, reward, done, _,info = self.step(action)
        
        return self.x,obs_next[8:]
        

def action2control(u_prev:np.array, action:np.array):
    return np.clip(u_prev + action*delta_u, u_min, u_max) 

def is_terminal(observation)->bool:
    # _y1_prev,_y, u_prev, k, d = deconstructObs(observation)
    _y, u_prev, k, d = deconstructObs(observation)
    if ( k >= max_steps): 
        return True
    else:
        return False
    
'''    (0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8, 9,10,11,12)
S = (*y1_prev,y1,y2,y3,y4,u1,u2,u3,k, d1,d2,d3,d4)'''
def deconstructObs(observation):
    # y1_prev = observation[0]
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

def normalizeState(xx,xx_min,xx_max):
    state_norm = ((xx - xx_min)/(xx_max-xx_min))*(10) - 5
    return np.array(state_norm, dtype=np.float32)
