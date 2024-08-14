import matplotlib.pyplot as plt
import numpy as np
from SHARED.params import *
from IPython.display import display
from tabulate import tabulate
from pprint import pprint
from tabulate import tabulate
from numpy import loadtxt
import os

def average_metrics(path, sims = 30):
    reward_logs = []
    Y_logs = []
    U_logs = []
    comp_times = []
    vf_logs = []
    
    for sim in range(sims):
        sim_path = "/Sim_" + str(sim) + '/'
        reward_logs.append( loadtxt(path + sim_path + 'rewards_log.csv' , delimiter=',') )
        Y_logs.append( loadtxt(path + sim_path + 'Y_log.csv' , delimiter=',') )
        U_logs.append( loadtxt(path + sim_path + 'U_log.csv' , delimiter=',') )
        comp_times.append( loadtxt(path + sim_path + 'comp_time_log.csv' , delimiter=',') )
        if os.path.exists(path + sim_path + 'vf_log.csv'):
            vf_logs.append( loadtxt(path + sim_path + 'vf_log.csv', delimiter=',') )
            
    
    #Average Outputs
    avg_outputs = np.stack (Y_logs)
    avg_outputs = np.mean(avg_outputs,axis=0)
    
    min_outputs = np.min(np.stack (Y_logs),axis=0)
    max_outputs = np.max(np.stack (Y_logs),axis=0)
    
    
    #Average Outputs
    avg_controls = np.stack (U_logs)
    avg_controls = np.mean(avg_controls,axis=0)   
    
    min_inputs = np.min(np.stack (U_logs),axis=0)
    max_inputs = np.max(np.stack (U_logs),axis=0)
    
    #Average computational times
    avg_comp_times = np.stack (comp_times)
    avg_comp_times = np.mean(avg_comp_times,axis=0)      
            
    avg_reward_logs = np.stack(reward_logs)
    avg_reward_logs= np.mean(avg_reward_logs, axis=0)
    
    
    #Average Rewards Accumulated
    stacked_rewards = np.stack(reward_logs)
    avg_rewards = np.mean(stacked_rewards, axis=0)
    min_rewards = np.full(len(reward_logs[0]), np.inf)  # Start with positive infinity for min comparison
    max_rewards = np.full(len(reward_logs[0]), -np.inf) 
    
    variance = np.zeros(len(reward_logs[0])) 
    plt.figure()
    
    for i in range(len(reward_logs[0])):
        variance[i] = np.var(stacked_rewards[:,i])
        min_rewards[i] = np.min(stacked_rewards[:,i])
        max_rewards[i] = np.max(stacked_rewards[:,i])

    
    # plt.fill_between(x, max_rewards, min_rewards, color="crimson", alpha=0.4)
    # plt.fill_between(x, Y2max, Y2min, color="#9a0bad", alpha=0.4)  
    plt.plot(min_rewards,linestyle = '--',alpha = 0.5)
    plt.plot(max_rewards,linestyle = '--',alpha = 0.5)
    

    
    plt.plot(avg_reward_logs,label = "Reward")
    plt.grid(True)
    plt.xlabel("Time Step (k)")
    plt.ylabel("$Euro \cdot m^2%")
    plt.title("Cumulative Reward")
    plt.show()
    
    plt.figure()
    plt.plot (variance)
    plt.grid()
    plt.show()
    
    
    #Average Values recorded
    # if vf_logs:
    #     avg_values = np.stack(vf_logs)
    #     avg_values = np.mean(avg_values, axis=0)
    #     plt.plot(avg_values,label = "Values")
    #     plt.plot(avg_values+avg_rewards[:-1],label = "Difference")
    # plt.legend()
    
    
    fig, axs_y = plt.subplots(2, 2, sharex=True, sharey=False, layout='constrained', figsize=(10, 4))
    # plt.figure()
    # for realization in Y_logs:
    #     realization = np.stack(realization)
    axs_y[0,0].plot(min_outputs[:,0],alpha = 0.6)
    axs_y[0,1].plot(min_outputs[:,1],alpha = 0.6)
    axs_y[1,0].plot(min_outputs[:,2],alpha = 0.6)
    axs_y[1,1].plot(min_outputs[:,3],alpha = 0.6)
    
    axs_y[0,0].plot(max_outputs[:,0],alpha = 0.6)
    axs_y[0,1].plot(max_outputs[:,1],alpha = 0.6)
    axs_y[1,0].plot(max_outputs[:,2],alpha = 0.6)
    axs_y[1,1].plot(max_outputs[:,3],alpha = 0.6)
    
    axs_y[0,0].plot(avg_outputs[:,0],label = "Drymass")
    axs_y[0,1].plot(avg_outputs[:,1],label = "C02")
    axs_y[1,0].plot(avg_outputs[:,2],label = "Temperature")
    axs_y[1,1].plot(avg_outputs[:,3],label = "Humidity")
    axs_y[0,0].grid(True)
    axs_y[0,1].grid(True)
    axs_y[1,0].grid(True)
    axs_y[1,1].grid(True)
    
    

    
    
    #Metrics
    delta_drymass = avg_outputs[-1,0] - avg_outputs[0,0]
    total_c02 = sum(avg_controls[:, 0])
    total_q = sum(avg_controls[:, 2])
    epi= (delta_drymass)*(r_let) - total_c02*r_c02  - total_q*r_q
    
    avg_comp_time = np.mean(avg_comp_times)
    Final_Reward = avg_rewards[-1]
    
    metrics = {
        "EPI                (EURO/m2)":epi,
        "Total growth       (kg/m2)":delta_drymass,
        "Total C02 usage    (kg/m2)":total_c02*dT*mg_to_kg,
        "Total Heating      (kWh)": total_q*dT*j_to_kwh,   
        "Computational Time (s)":avg_comp_time,
        "FINAL PERFORMANCE     ":Final_Reward     ,
        "Variance"              :    variance[-1],
    }
    
    print(tabulate(metrics.items()))
    


def print_metrics(Y_log= np.empty((0,)), U_log= np.empty((0,)), D_log= np.empty((0,)), vf= np.empty((0,)),rewards= np.empty((0,)), day_range = (0,tf), time_log = np.empty((0,)), show_charts=True):
    
    Y_log = Y_log[:, :][(day_range[0] * 96):(day_range[1]) * 96]
    U_log = U_log[:, :][(day_range[0] * 96):(day_range[1]) * 96]
    D_log = D_log[:, :][(day_range[0] * 96):(day_range[1]) * 96]
    
    if show_charts:
        #Disturbances
        fig, axs_d = plt.subplots(2, 2, sharex=True, sharey=False, layout='constrained', figsize=(15, 3))
        fig.suptitle("Disturbances")
        for i, (title, ylabel) in enumerate([("Irradiance", "$W \cdot m^{-2}$"), ("Outdoor C02", "$kg \cdot m^{-3}$"),
                                            ("Outdoor Temperature", "$^{\circ}C$"), ("Outdoor Humidity", "$kg \cdot m^{-3}$")]):
            ax = axs_d[i // 2, i % 2]
            ax.set_title(title)
            ax.plot(D_log[:,i], color = 'lightCoral',linestyle='-', marker='.', markerfacecolor='g', markeredgecolor='g',markersize = 3)
            ax.grid(True)
            ax.set_ylabel(ylabel)


        #Outputs
        fig, axs_y = plt.subplots(2, 2, sharex=True, sharey=False, layout='constrained', figsize=(15, 3))
        fig.suptitle("Outputs")
        for i, (title, ylabel) in enumerate([("Drymass", "$kg \cdot m^{-2}$"), ("Indoor C02", "ppm"),
                                            ("Indoor Temperature", "$^{\circ}C$"), ("Indoor Humidity","%")]):
            ax = axs_y[i // 2, i % 2]
            ax.set_title(title)
            ax.plot(Y_log[:,i], color = 'lightCoral', linestyle='-', marker='.', markerfacecolor='g', markeredgecolor='g', markersize = 3)
            ax.grid(True)
            ax.set_ylabel(ylabel)


        #Control Actions
        fig, axs_u = plt.subplots(3,1,sharex=True, sharey=False, layout='constrained',figsize=(7, 5))
        fig.suptitle("Control Actions")
        for i, (title, ylabel) in enumerate([("C02 Injection Rate", "$mg/(m^{2} \cdot s)$"), ("Ventilation Rate", "$mm/s$"),
                                            ("Heating Supply", "$W \cdot m^{-2}$")]):
            ax = axs_u[i]
            ax.set_title(title)
            ax.plot(U_log[:,i], color = 'lightCoral',linestyle='-', marker='.', markerfacecolor='g', markeredgecolor='g', markersize = 3)
            ax.grid(True)
            ax.set_ylabel(ylabel)


        #Constraints
        
        
        # axs_y[0,1].plot(c02_mins[(day_range[0] * 96):(day_range[1]) * 96],color = 'k' ,linestyle = '--')
        # axs_y[0,1].plot(c02_maxs[(day_range[0] * 96):(day_range[1]) * 96],color = 'k' ,linestyle = '--')

        #CO2 Constrains
        axs_y[0,1].axhline(y=C02_MAX_CONSTRAIN_MPC,color = 'k' ,linestyle = '--')
        axs_y[0,1].axhline(y=C02_MIN_CONSTRAIN_MPC,color = 'k' ,linestyle = '--')

        #Humidity Constrains
        axs_y[1,1].axhline(y=HUM_MAX_CONSTRAIN,color = 'k' ,linestyle = '--')
        
        #Temperature Constrains
        axs_y[1,0].axhline(y=TEMP_MAX_CONSTRAIN_MPC,color = 'k' ,linestyle = '--')
        axs_y[1,0].axhline(y=TEMP_MIN_CONSTRAIN_MPC,color = 'k' ,linestyle = '--')
        
        # axs_y[0,1].plot(temp_mins[(day_range[0] * 96):(day_range[1]) * 96],color = 'k' ,linestyle = '--')
        # axs_y[0,1].plot(temp_maxs[(day_range[0] * 96):(day_range[1]) * 96],color = 'k' ,linestyle = '--')



    
    
        #Value function and cumulative rewards
        if rewards.shape != (0,)  or vf.shape != (0,):
            fig, axs_vf = plt.subplots(1,1,sharex=True, sharey=False, layout='constrained',figsize=(7, 3))    
            axs_vf.set_title("Value Function")

            if vf.shape != (0,):
                axs_vf.plot(vf[(day_range[0] * 96):(day_range[1]) * 96], color = 'g',linestyle='-', marker='.', markerfacecolor='g', markeredgecolor='g', markersize = 3,label = "Value Function")

            if rewards.shape != (0,):    
                axs_vf.plot(rewards[(day_range[0] * 96):(day_range[1]) * 96], color = 'Cyan',linestyle='--', marker='.', markerfacecolor='Cyan', markeredgecolor='Cyan', markersize = 3, label = "Cumulative Rewards")
        
            if rewards.shape != (0,) and vf.shape != (0,):
                axs_vf.plot((vf+rewards)[(day_range[0] * 96):(day_range[1]) * 96], color = 'Pink',linestyle='--', marker='.', markerfacecolor='Pink', markeredgecolor='Pink', markersize = 3, label = "Difference")
            axs_vf.legend()
            axs_vf.grid(True)
            axs_vf.set_ylabel('value/reward')
        
    #Calculating Rewards
    delta_drymass = Y_log[-1,0] - Y_log[0,0]
    total_c02 = sum(U_log[:, 0])
    total_q = sum(U_log[:, 2])
    epi= (delta_drymass)*(r_let) - total_c02*r_c02  - total_q*r_q

    
    
    avg_comp_time = None if time_log.size == 0 else np.mean(time_log)
    Final_Reward = rewards[-1] if np.any(rewards != None) else 0
    
    violations_temp = 0
    violations_c02 = 0
    #Constraint violations
    temp_violations = 0
    for temp in Y_log[:,2]:
        if temp > TEMP_MAX_CONSTRAIN_MPC:
            temp_violations += temp - TEMP_MAX_CONSTRAIN_MPC
            violations_temp += (temp - TEMP_MAX_CONSTRAIN_MPC)*pen_temp_ub
        elif temp < TEMP_MIN_CONSTRAIN_MPC:
            temp_violations += TEMP_MIN_CONSTRAIN_MPC - temp
            violations_temp += (TEMP_MIN_CONSTRAIN_MPC - temp)*pen_temp_lb
            
    c02_violations = 0
    for c02 in Y_log[:,1]:
        if c02 > C02_MAX_CONSTRAIN_DAY:
            c02_violations += c02 - C02_MAX_CONSTRAIN_DAY
            violations_c02 += (c02 - C02_MAX_CONSTRAIN_DAY)*pen_c02
        elif c02 < C02_MIN_CONSTRAIN_DAY:
            c02_violations += C02_MIN_CONSTRAIN_DAY - c02
            violations_c02 += (C02_MIN_CONSTRAIN_DAY - c02)*pen_c02
    

    calc_reward =epi- violations_c02 - violations_temp
    
    metrics = {
        "EPI                (EURO/m2)":epi,
        "Total growth       (kg/m2)":delta_drymass,
        "Total C02 usage    (kg/m2)":total_c02*dT*mg_to_kg,
        "Total Heating      (kWh)": total_q*dT*j_to_kwh,   
        "Computational Time (s)":avg_comp_time,
        "Temp violations    (deg)":temp_violations,
        "C02 violations     (ppm)":c02_violations,
        "FINAL PERFORMANCE     ":Final_Reward  ,
        "Calculated Performance":calc_reward,    
        "Temp violations (ERO/m2)":violations_temp,
        "C02 violations (ERO/m2)": violations_c02
    }
    
    # print(tabulate(metrics.items()))
    return metrics
    

def compare_metrics(Y_logs = {}, U_logs = {}, D_logs = {}, reward_logs = {}, comp_time_logs = {}):
    
    '''Time Series'''
    #Outputs
    delta_drymass = 0
    y_rl,y_mpc,y_rl_mpc = [],[],[]
    
    y_rl     = Y_logs["rl"]
    y_mpc    = Y_logs["mpc"]
    y_rl_mpc = Y_logs["rl_mpc"]
    delta_drymass = np.array([y_rl[-1,0] - y_rl[0,0] ,y_mpc[-1,0] - y_mpc[0,0] ,y_rl_mpc[-1,0] - y_rl_mpc[0,0]])
    
    fig, axs_y = plt.subplots(4,1, sharex=True, sharey=False, layout='constrained', figsize=(15, 10))
    fig.suptitle("Outputs")
        
    for i, (title, ylabel) in enumerate([("Drymass", "$kg \cdot m^{-2}$"), ("Indoor C02", "ppm"),
                                        ("Indoor Temperature", "$^{\circ}C$"), ("Indoor Humidity","%")]):
        ax = axs_y[i]
        ax.set_title(title)
        ax.plot(y_rl[:,i],alpha = 0.7,     label = "RL")
        ax.plot(y_mpc[:,i],alpha = 0.7,   label = "MPC")
        ax.plot(y_rl_mpc[:,i],alpha = 0.7,label = "RL_MPC")
        ax.grid(True)
        ax.set_ylabel(ylabel)
        ax.legend()
    
    #Control actions 
    total_c02,total_q,total_v=0,0,0
    u_rl,u_mpc,u_rl_mpc = [],[],[]

    u_rl        = U_logs["rl"]
    u_mpc       = U_logs["mpc"]
    u_rl_mpc    = U_logs["rl_mpc"]
    
    total_c02 = np.array([sum(u_rl[:, 0]), sum(u_mpc[:, 0]),sum( u_rl_mpc[:, 0])])
    total_q   = np.array([sum(u_rl[:, 2]), sum(u_mpc[:, 2]),sum( u_rl_mpc[:, 2])])
    total_v   = np.array([sum(u_rl[:, 1]), sum(u_mpc[:, 1]),sum( u_rl_mpc[:, 1])])
        
    # Time series      
    fig, axs_u = plt.subplots(3,1,sharex=True, sharey=False, layout='constrained',figsize=(15, 10))
    fig.suptitle("Control Actions")
    for i, (title, ylabel) in enumerate([("C02 Injection Rate", "$mg/(m^{2} \cdot s)$"), ("Ventilation Rate", "$mm/s$"),
                                        ("Heating Supply", "$W \cdot m^{-2}$")]):
        ax = axs_u[i]
        ax.set_title(title)
        ax.plot(u_rl[:,i],alpha = 0.7,label = 'RL')
        ax.plot(u_mpc[:,i],alpha = 0.7,label = 'MPC')
        ax.plot(u_rl_mpc[:,i],alpha = 0.7,label = 'RL_MPC')
        ax.grid(True)
        ax.set_ylabel(ylabel)
        ax.legend()
            
    #Rewards
    final_reward = [0,0,0]
    reward_rl        = reward_logs["rl"]
    reward_mpc       = reward_logs["mpc"]
    reward_rl_mpc    = reward_logs["rl_mpc"]   

    final_reward     = [reward_rl[-1],reward_mpc[-1],reward_rl_mpc[-1]]
    
    
    fig, axs_r = plt.subplots(1,1,sharex=True, sharey=False, layout='constrained',figsize=(10, 5))
    axs_r.set_title("Cumulative Reward")
    axs_r.plot(reward_rl[:],alpha = 0.7,label = 'RL')
    axs_r.plot(reward_mpc[:],alpha = 0.7, label = 'MPC')
    axs_r.plot(reward_rl_mpc[:],alpha = 0.7,label = 'RL_MPC')
    axs_r.grid(True)
    axs_r.set_ylabel("Reward")
    axs_r.legend()   
     
    comp_times = [0,0,0]
    comp_times = np.array([np.mean(comp_time_logs["rl"]), np.mean(comp_time_logs["mpc"]),np.mean(comp_time_logs["rl_mpc"])])
        
     
    final_epi = (delta_drymass)*(r_let) - total_c02*r_c02  - total_q*r_q


    '''Bar graphs'''
    fig, axs_bars = plt.subplots(2,3,sharex=True, sharey=False, layout='constrained',figsize=(10, 5))
    fig.suptitle("Metric Comparisons")

    # Create growth bar plot
    delta_drymass  = delta_drymass*1000
    colors = ['skyblue', 'salmon','lightgreen']
    ax_growth = axs_bars[0][0]
    ax_growth.set_title("Drymass Growth")
    bars = ax_growth.bar(["RL","MPC","RL_MPC"],delta_drymass, color =  colors)   
    ax_growth.set_ylabel("$g \cdot m^{-2}$")
    # Add text annotations
    for i, value in enumerate(delta_drymass):
        ax_growth.text(i, value, str(round(value,1)),
                    ha='center', va='top')
    
    #EPI bar graphs
    ax_epi =axs_bars[0][1]
    ax_epi.set_title("Economic Benefit")
    ax_epi.bar(["RL","MPC","RL_MPC"],final_epi, color =  colors)
    ax_epi.set_ylabel("Euro $\cdot m^{-2}$")
    # Add text annotations
    for i, value in enumerate(final_epi):
        ax_epi.text(i, value, str(round(value,1)),
                    ha='center', va='top')
        
    #Final Performance Bar graph
    ax_perf = axs_bars[0][2]
    ax_perf.set_title("Final Performance")
    ax_perf.bar(["RL","MPC","RL_MPC"],final_reward, color =  colors)
    ax_perf.set_ylabel("Reward")
    # Add text annotations
    for i, value in enumerate(final_reward):
        ax_perf.text(i, value, str(round(value,3)),
                    ha='center', va='top')
    
    #Computational Time
    ax_time = axs_bars[1][0]
    ax_time.set_title("Comp Time")
    ax_time.bar(["RL","MPC","RL_MPC"],comp_times, color =  colors)
    ax_time.set_ylabel("Time (s)")
    # Add text annotations
    for i, value in enumerate(comp_times):
        ax_time.text(i, value, str(round(value,3)),
                    ha='center', va='top')

    #C02 Usage
    ax_c02 = axs_bars[1][1]
    ax_c02.set_title("C02")
    ax_c02.bar(["RL","MPC","RL_MPC"],total_c02*dT/1e6, color =  colors)
    ax_c02.set_ylabel("$kg/(m^{2} \cdot s)$")
    # Add text annotations
    for i, value in enumerate(total_c02*dT/1e6):
        ax_c02.text(i, value, str(round(value,3)),
                    ha='center', va='top')
    
    #Heating Usage
    ax_q = axs_bars[1][2]
    ax_q.set_title("Heating")
    ax_q.bar(["RL","MPC","RL_MPC"],total_q*dT/1e6, color =  colors)
    ax_q.set_ylabel("MJ")
    # Add text annotations
    for i, value in enumerate(total_q*dT/1e6):
        ax_q.text(i, value, str(round(value,3)),
                    ha='center', va='top')
    
    
    metrics_1 = {
        "RL --> MPC"        :None,
        "EPI            (%)":100*(final_epi[1] - final_epi[0])/final_epi[0],
        "Growth         (%)":100*(delta_drymass[1] - delta_drymass[0])/delta_drymass[0],
        "C02 usage      (%)":100*(total_c02[1] - total_c02[0])/total_c02[0], 
        "Heating        (%)":100*(total_q[1] - total_q[0])/total_q[0],    
        "Ventilation    (%)":100*(total_v[1] - total_v[0])/total_v[0],  
        "Time           (%)":100*(comp_times[1] - comp_times[0])/comp_times[0],
        "Perf           (%)":100*(final_reward[1] - final_reward[0])/final_reward[0],      
    }
    
    
    
    metrics_2 = {
        "RL --> RL_MPC"     :None,
        "EPI            (%)":100*(final_epi[2] - final_epi[0])/final_epi[0],
        "Growth         (%)":100*(delta_drymass[2] - delta_drymass[0])/delta_drymass[0],
        "C02 usage      (%)":100*(total_c02[2] - total_c02[0])/total_c02[0], 
        "Heating        (%)":100*(total_q[2] - total_q[0])/total_q[0],    
        "Ventilation    (%)":100*(total_v[2] - total_v[0])/total_v[0],  
        "Time           (%)":100*(comp_times[2] - comp_times[0])/comp_times[0],
        "Perf           (%)":100*(final_reward[2] - final_reward[0])/final_reward[0],      
    }
    
    
    
    metrics_3 = {
        "MPC --> RL_MPC"    :None,
        "EPI            (%)":100*(final_epi[2] - final_epi[1])/final_epi[1],
        "Growth         (%)":100*(delta_drymass[2] - delta_drymass[1])/delta_drymass[1],
        "C02 usage      (%)":100*(total_c02[2] - total_c02[1])/total_c02[1], 
        "Heating        (%)":100*(total_q[2] - total_q[1])/total_q[1],    
        "Ventilation    (%)":100*(total_v[2] - total_v[1])/total_v[1],  
        "Time           (%)":100*(comp_times[2] - comp_times[1])/comp_times[1],
        "Perf           (%)":100*(final_reward[2] - final_reward[1])/final_reward[1],      
    }
    
    print(tabulate(metrics_1.items(),numalign=['left', 'right']))
    print(tabulate(metrics_2.items(),numalign=['left', 'right']))
    print(tabulate(metrics_3.items(),numalign=['left', 'right']))