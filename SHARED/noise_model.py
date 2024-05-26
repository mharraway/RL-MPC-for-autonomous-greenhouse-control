import numpy as np
import casadi
from SHARED.params import *

class noise_model():
    def __init__(self,scale = 0) -> None:
        
        #Data ranges of weather
        self.I_range = 635
        self.C02_range = 0.00041
        self.T_range = 17.4
        self.H_range = 0.005
        
        # self.std = scale * np.array([self.I_range,self.C02_range,self.T_range,self.H_range])
        self.scale = scale
        self.e = 0
        
    def add_noise(self,x,x_next):
        curr_x = np.copy(np.array(x))
        next_x = np.copy(np.array(x_next))
        
        curr_growth = next_x[0] - curr_x[0]
        w = np.random.uniform(low=1-self.scale, high=1+self.scale)
        # w = np.random.normal(loc=0,scale=self.scale*( np.abs(next_x[0]-curr_x[0]) ))
        
        
        next_x[0] = curr_x[0]  + curr_growth*(w)
        next_x[1] = curr_x[1]  + (next_x[1] - curr_x[1])*(w)
        next_x[2] = curr_x[2]  + (next_x[2] - curr_x[2])*(w)
        next_x[3] = curr_x[3]  + (next_x[3] - curr_x[3])*(w)
        # w = np.random.normal(loc=0,scale=self.scale*( np.abs(next_x[0]-curr_x[0]) ))
        
        # new_dry_mass
        # new_dry_mass = curr_x[0] + stochastic_growth
        
        # #Dont let dry mass go below 1 gram for stability reasons in model
        next_x[0] = np.maximum(0.0025,next_x[0])
        
        # next_x[0] = new_dry_mass  
        
        return next_x
    
    def parametric_uncertainty(self):
        range_width = self.scale * nominal_params
        
        lower_bound = nominal_params - range_width/2
        upper_bound = nominal_params + range_width/2
        
        sample = np.random.uniform(lower_bound, upper_bound)
        
        return sample
            
            