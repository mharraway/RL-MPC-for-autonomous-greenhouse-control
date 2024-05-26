
import pandas as pd
import numpy as np
import random
import os
import torch
from SHARED.model import *

def get_disturbance(k:int = 0, weather_data:pd.DataFrame = None ,start_time:int = 0, Np = 1, dt = 900, stochastic = False)->np.array:
    time = k*dt
    total_seconds = 31536000 ##this might need updating
    d = np.zeros((4,Np))
    time = (start_time + time) # total_seconds
    fields = ['Io','C02', 'To','Hum']
    for i in range(Np):
        d[:,i] = weather_data[weather_data['time'] == (time + i*dt)][fields].to_numpy()[0].copy()
    return d


    
def load_weather_data(path)->pd.DataFrame:
    
    if "seljaar" in path:
        column_names = ['time', 'Io', 'Vo', 'To','Tsky', 'To_n', 'C02ppm', 'dayNumber', 'RH']
    elif "energy_plus" in path:
        column_names = ['time', 'Io', 'To','Hum','C02', 'Vo',"Tsky","Tsoil","Inight","elevation"]
    elif "outdoorWeatherWurGlas2014" in path:
        column_names = ['time', 'Io', 'To','RH','Vo','C02ppm']
    else:
        return None
    
    weather_data = pd.read_csv(path, header=None,names=column_names)
    
    if "seljaar" or "outdoorWeatherWurGlas2014" in path:
        #Change Relative Humidity to Absolute Humidity
        weather_data = weather_data.assign(Hum=weather_data.apply(lambda row: rh2vaporDens(row['To'],row['RH']), axis=1))
        #Change C02 ppm to kg.m^-3
        weather_data = weather_data.assign(C02=weather_data.apply(lambda row: co2ppm2dens(row['To'],row['C02ppm']), axis=1))
        
        
    total_seconds = weather_data.loc[weather_data.index[-1]]["time"]
        
    return weather_data,total_seconds




