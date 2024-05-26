from SHARED.params import *


def model_functions(dt = 1800):
    
    #params
    system_params = casadi.MX.sym("params", 28)
    
    #State Variables
    x1 = casadi.MX.sym("x1")   #->Drymass              (kg/m2)
    x2 = casadi.MX.sym("x2")   #->Indoor C02           (kg/m3)
    x3 = casadi.MX.sym("x3")   #->Indoor Temp          (deg C)
    x4 = casadi.MX.sym("x4")   #->Indoor Humidity      (kg/m3)
    X  = casadi.vertcat(x1,x2,x3,x4)

    #Disturbance Variables
    d1 = casadi.MX.sym("d1")   #->Irradiance           (W/m2)
    d2 = casadi.MX.sym("d2")   #->Outdoor C02          (kg/m3)
    d3 = casadi.MX.sym("d3")   #->Outdoor Temp         (deg C)
    d4 = casadi.MX.sym("d4")   #->Outdoor Humidity     (kg/m3)
    D  = casadi.vertcat(d1,d2,d3,d4)

    #Control Variables
    u1 = casadi.MX.sym("u1")   #->C02 Injection        (mg/(m2.s))
    u2 = casadi.MX.sym("u2")   #->Ventilation          (mm/s)
    u3 = casadi.MX.sym("u3")   #->Heating              (W/m2)
    U  = casadi.vertcat(u1,u2,u3)


   # Intermediate Processes
    phi_phot_c      = (1-np.exp(-system_params[2]*x1)) * (system_params[3] * d1 * (-system_params[4] * (x3**2) + system_params[5]*x3 - system_params[6])*(x2 - system_params[7]))/(system_params[3] * d1 + (-system_params[4] * (x3**2) + system_params[5]*x3 - system_params[6])*(x2 - system_params[7]))
    phi_vent_c      = (u2*(1e-3) + system_params[10]) * (x2-d2)
    Q_vent_q        = (system_params[16]*u2*(1e-3) + system_params[17]) * (x3-d3)
    Q_rad_q         =  system_params[18]*d1
    phi_transp_h    = system_params[20] * (1-np.exp(-system_params[2]*x1)) * ((system_params[21]/(system_params[22]*(x3 + system_params[23]))) * np.exp((system_params[24]*x3)/(x3+system_params[25])) - x4)
    phi_vent_h      = (u2*(1e-3) + system_params[10])*(x4-d4)

    # Model Dynamics
    dx1dt =    system_params[0] *  phi_phot_c  -  system_params[1] * x1 * 2**(0.1*x3 - 2.5)                                # change in Drymass
    dx2dt = (1/system_params[8])*(-phi_phot_c  +  system_params[9] * x1 * 2**(0.1*x3 - 2.5) + u1*(1e-6) - phi_vent_c)      # change in C02 Density
    dx3dt = (1/system_params[15])*(u3           - Q_vent_q                        + Q_rad_q)                                # Change in Temperature
    dx4dt = (1/system_params[19])*(phi_transp_h - phi_vent_h)                                                             # Change in Humidity

    #Model Outputs
    y1 = x1
    y2 = co2dens2ppm(x3,x2, system_params)
    y3 = x3
    y4 = casadi.mmin(casadi.horzcat(100,vaporDens2rh(x3,x4,system_params)))
    

    #for numerical stability of the model
    x1 = casadi.mmin(casadi.horzcat(x_max[0],x1))
    x2 = casadi.mmin(casadi.horzcat(x_max[1],x2))
    x3 = casadi.mmin(casadi.horzcat(x_max[2],x3))
    x4 = casadi.mmin(casadi.horzcat(x_max[3],x4))
    
    x1 = casadi.mmax(casadi.horzcat(x_min[0],x1))
    x2 = casadi.mmax(casadi.horzcat(x_min[1],x2))
    x3 = casadi.mmax(casadi.horzcat(x_min[2],x3))
    x4 = casadi.mmax(casadi.horzcat(x_min[3],x4))


    #Functions
    output = casadi.vertcat(y1,y2,y3,y4)
    ode = casadi.vertcat(dx1dt, dx2dt, dx3dt, dx4dt)

    g = casadi.Function('g',[X,system_params],[output],['x','params'],['output'])
    f = casadi.Function('f',[X,U,D,system_params],[ode],['x','u','d','params'],['ode'])
    f = f.expand()
    g = g.expand()
    
    
    #Integration Options
    intg_options = {}
    intg_options["simplify"] = True
    intg_options["number_of_finite_elements"] = 4

    fixed_params = casadi.vertcat(D, system_params)
    #Dynamical System
    dyn = {}
    dyn["x"] = X
    dyn["u"] = U                                                            #parameters that are fixed during integration horizon
    dyn["p"] = fixed_params                                                        #parameters that are fixed during integration horizon
    dyn["ode"] = f(X,U,D,system_params)

    
    #Integrator
    intg = casadi.integrator('intg','rk',dyn,0,dt,intg_options)             #Runge Kutta 4th Order Method
    res = intg(x0 = X,p = fixed_params,u = U)
    F = casadi.Function('F',[X,U,D,system_params],[res["xf"]],['x','u','d','p'],['xnext'])    #Discretized Function
    
    return F,g


def vaporDens2rh(temp, vaporDens,params):
    satP = p_0*np.exp(params[26]*temp/(temp+params[27]))
    rh = (100*params[11]*(temp+params[12])/(Mw*satP))*vaporDens
    return rh

def rh2vaporDens(temp, rh):
    satP = p_0*np.exp(c_4_8*temp/(temp+c_4_9))
    pascals=(rh/100)*satP
    vaporDens = pascals*Mw/(c_2_4*(temp+c_2_5))
    return vaporDens

def co2ppm2dens(temp, ppm):
    co2Dens = c_2_6*(1e-6)*ppm*c_2_7/(c_2_4*(temp+c_2_5))
    return co2Dens
def co2dens2ppm(temp, dens, params):
    ppm = (1e6)*params[11]*(temp+params[12])*dens/(params[13]*params[14])
    return ppm


def reward_function(delta_drymass = 0, control_inputs = (0,0,0), return_type = "DM"):

    reward = 0
    #Control Inputs
    u_c02 = control_inputs[0]
    u_v   = control_inputs[1]
    u_q   = control_inputs[2]
    
    #Reward Recieved
    reward = r_let*(delta_drymass) - (u_c02*r_c02 + u_q*r_q + u_v*0)
    
    reward = float(reward) if return_type == "float" else reward
    return reward


def penalty_function(outputs2constrain = None, constraint_mins = None,constraint_maxs = None):

    penalties = 0
    # C02 constraints
    c02_level   = outputs2constrain[0]
    c02_min     = constraint_mins[0]
    c02_max     = constraint_maxs[0]
    
    if c02_level < c02_min:
        penalties += pen_c02 * (c02_min - c02_level)
    elif c02_level > c02_max:
        penalties += pen_c02 * (c02_level - c02_max)
    else:
        penalties -= pen_reward
    
    #Temp constraints      
    temp_min    = constraint_mins[1]
    temp_max    = constraint_maxs[1]   
    temp_level  = outputs2constrain[1]  
    
    if temp_level < temp_min:
        penalties += pen_temp_lb * (temp_min - temp_level)
    elif temp_level > temp_max:
        penalties += pen_temp_ub * (temp_level - temp_max)
    else:
        penalties -= pen_reward
    
    return penalties
        
#Might be Deprecated
def reward_evaluation(delta_drymass = 0, control_inputs = None, outputs2constrain = None,constraint_mins = None,constraint_maxs = None):
    reward = 0
    penalties = 0
    
    reward = reward_function(delta_drymass,control_inputs, return_type="float")
    penalties = penalty_function (outputs2constrain=outputs2constrain,constraint_mins=constraint_mins, constraint_maxs=constraint_maxs)
    
    return reward,penalties
    
 