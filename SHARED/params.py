import numpy as np
import casadi
import torch

#Model Parameters
c_1_1       =       0.544                               #c_alpha_beta  
c_1_2       =       2.65e-7                             #c_resp_d
c_1_3       =       53                                  #c_LAI_d 
c_1_4       =       3.55e-9                             #c_phot_Io
c_1_5       =       5.11e-6                             #c_phot_C02_1
c_1_6       =       2.3e-4                              #c_phot_C02_2
c_1_7       =       6.29e-4                             #c_phot_C02_3
c_1_8       =       5.2e-5                              #c_phot


c_2_1       =       4.1                                 #c_cap_c
c_2_2       =       4.87e-7                             #c_resp_c
c_2_3       =       7.5e-6                              #c_leak   0.75e-4
c_2_4       =       8.3144598                           #R
c_2_5       =       273.15                              #c_T 
c_2_6       =       101325                              #P
c_2_7       =       0.04401                             #M_C02  
c_2_8       =       10998                               #---

c_3_1       =       3e4                                 #c_cap_q
c_3_2       =       1290                                #c_cap_v
c_3_3       =       6.1                                 #c_go
c_3_4       =       0.2                                 #c_rad_og

c_4_1       =       4.1                                 #c_cap_h
c_4_2       =       3.6e-3                              #c_evap_ca
c_4_3       =       9348                                #c_sat_H20_1
c_4_4       =       8314                                #c_R
c_4_5       =       273.15                              #c_T 
c_4_6       =       17.4                                #c_sat_H20_2
c_4_7       =       239                                 #c_sat_H20_3
c_4_8       =       17.2694                              #c_sat_H20_5  17.4
c_4_9       =       238.3                               #c_sat_H20_6  239

nominal_params = np.array([
    c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7, c_1_8,
    c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7,
    c_3_1, c_3_2, c_3_3, c_3_4,
    c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9
])

p_0         =       610.78
Mw          =       18.01528e-3 




# Model Configuration
dT          =       1800                                 #time scale of my system




noise_scale = 0
GlobalSeed = 4

#Time Period and settings
start_day       = 30                          #start of growth period (days) ------>Change This
tf              = 40                          #period of growth
eps             = 100#300                         #Number of episodes to train
warm_up_eps     = 9                          #Number of warm up episodes

start_time      = (start_day-1)*86400 + 3600  #start of growth period (seconds)
max_steps       = (tf  * 86400)//dT           #number of steps in the simulation


#Initial Condtions
x0 = np.array([0.0035,0.001,15,0.008])
u0 = np.array([0,0,50])


#----Min and Max Ranges for system internal states and INPUTS----
DRYMASS_MAX = 0.6
DRYMASS_MIN = 0.002

C02_MAX = 0.004
C02_MIN = 0

# TEMP_MAX = 60
# TEMP_MIN = -20

TEMP_MAX = 40
TEMP_MIN = 5


HUM_MAX = 0.051
HUM_MIN = 0

C02_MAX_INPUT   = 1.2
VENT_MAX_INPUT  = 7.5
HEAT_MAX_INPUT  = 150

u_min   = np.array([0, 0, 0])                                       
u_max   = np.array([C02_MAX_INPUT, VENT_MAX_INPUT, HEAT_MAX_INPUT]) 
delta_u = 0.1*u_max     

x_min = np.array([DRYMASS_MIN,C02_MIN,TEMP_MIN,HUM_MIN])
x_max = np.array([DRYMASS_MAX,C02_MAX,TEMP_MAX,HUM_MAX])

# obs_mon = np.concatenate([[DRYMASS_MAX, 2000, TEMP_MAX, 100],u_max,[600,0.0012,20,0.01]])
# obs_min = np.array([DRYMASS_MIN, C02_MIN, TEMP_MIN, HUM_MIN,u_min,0,0.0006,0,0.004])
# obs_max = np.array([DRYMASS_MAX, 2000, TEMP_MAX, 100],u_max,[600,0.0012,20,0.01])
obs_min = np.concatenate([[DRYMASS_MIN, C02_MIN, TEMP_MIN, HUM_MIN],u_min,[0],[0,0.0006,0,0.004]])
obs_max = np.concatenate([[DRYMASS_MAX, 2000, TEMP_MAX, 100],u_max,[max_steps],[600,0.0012,20,0.01]])
#----Constraints on system internal outputs----

C02_MAX_CONSTRAIN_MPC       = 1600 #1250
C02_MIN_CONSTRAIN_MPC       = 500#700

TEMP_MAX_CONSTRAIN_MPC       = 20 #1250
TEMP_MIN_CONSTRAIN_MPC       = 10#700

C02_MAX_CONSTRAIN_DAY       = 1600
C02_MIN_CONSTRAIN_DAY       = 500#800
C02_MAX_CONSTRAIN_NIGHT     = 1600#1200
C02_MIN_CONSTRAIN_NIGHT     = 500

TEMP_MAX_CONSTRAIN_DAY      = 20
TEMP_MIN_CONSTRAIN_DAY      = 10
TEMP_MAX_CONSTRAIN_NIGHT    = 20
TEMP_MIN_CONSTRAIN_NIGHT    = 10

HUM_MAX_CONSTRAIN           = 90
HUM_MIN_CONSTRAIN           = 0

y_max_day   = np.array([np.inf, C02_MAX_CONSTRAIN_DAY, TEMP_MAX_CONSTRAIN_DAY, HUM_MAX_CONSTRAIN])         
y_min_day   = np.array([0, C02_MIN_CONSTRAIN_DAY, TEMP_MIN_CONSTRAIN_DAY, HUM_MIN_CONSTRAIN]) 

y_max_night  = np.array([np.inf, C02_MAX_CONSTRAIN_NIGHT, TEMP_MAX_CONSTRAIN_NIGHT, HUM_MAX_CONSTRAIN])      
y_min_night  = np.array([0, C02_MIN_CONSTRAIN_NIGHT, TEMP_MIN_CONSTRAIN_NIGHT, HUM_MIN_CONSTRAIN])                          

#Rewards and Prices This may change!
c_hw        = (1-0.07)*22.5                             #Fresh weight to dry weight ratio, as found in van henten optimal control
c_price_let = (0.8+1.33)/2                              #The retail price range in Euro for lettuce is between EUR 0.80 and EUR 1.33 per kilogram

c_price_C02 = 0.1906 #0.08  #0.14              #0.08                        #Some papers say 0.14, others say 0.1906 EUR/Kg
c_price_q   = 0.1281 #0.06                        #electricity peak of 0.168 and offf-peak of 0.088 EUR/kWh. Gas Prices however are 0.0134 EUR/MJ. Which is cheaper

#Penalties for state constraint violation
pen_c02     = (1/20) * (1e-3)#0.1
pen_temp_ub = (1/200)#0.001
pen_temp_lb = (1/300)#0.001
pen_hum     = (1/50)#0.02
pen_reward  = 0.000


#Conversion Factors
j_to_kwh = 1/3.6e6
mg_to_kg = 1e-6


#Economic Rewards
r_let = c_price_let*c_hw
r_c02 = dT*mg_to_kg*c_price_C02
r_q   = dT*j_to_kwh*c_price_q


 


