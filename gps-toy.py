# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:31:36 2022

GPX parser 

@author: 578507
"""
import sys
sys.path.append("C:/GPX/gpxpy-1.5.0/") # look for user defined functions in this path
sys.path.append("C:/GPX/filterpy-master") # look for user defined functions in this path


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import regex as re
from datetime import datetime, timedelta
#import dateutil.parser as dparser 
#import matplotlib.dates as mdates
#from matplotlib.dates import DateFormatter
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
#import math as m
import gpxpy, gpxpy.gpx 
from mpl_toolkits import mplot3d
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


meter2mile = 1./1609
meter2ft = 3.28084
ms2mph = 2.23694
nm2meter = 1852

# returns distance d in meters calculation on a great circle
def gc_dist(df):
    diff_df = df.diff().drop([0])
    shift_df = df.shift(periods=-1).drop([len(df)-1])
    # Distance between points on great circle less subject to rounding error for short distances in radians
    d = 2*np.arcsin(np.sqrt((np.sin((diff_df['lat_rad'])/2))**2 + 
        np.cos(df['lat_rad'])*np.cos(shift_df['lat_rad'])*(np.sin((diff_df['lon_rad'])/2))**2))   
    d = round(d*180*60*nm2meter/np.pi,6)
    return d


# returns distance d in meters using flat earth approx
# def fe_dist(df):
#     diff_df = df.diff().drop([0])
#     shift_df = df.shift(periods=-1).drop([len(df)-1])
    
    
#     distance_North=R1*dlat
#     distance_East=R2*cos(lat0)*dlon
  
#     d = round(d*180*60*nm2meter/np.pi,6)
#     return d


# calculate the instantaneous grade in degrees
def grade_calc(df):
    g = df.delta_z * 3.28084 / df.delta_dist
    g = g.fillna(0.0)
    df['grade'] = np.arctan(g)

def stat_printer(df):
    dist_total = round(df['delta_dist'].sum() * meter2mile, 1)
    ele_gain = round(df.query('delta_z > 0')['delta_z'].sum() * meter2ft,1)
    max_speed = round(df['gc_vel_inst'].quantile(.9999) * ms2mph,1)
    elapsed_time = df.timestamp[len(df)-1] - df.timestamp[0] # total elapsed time 
    stopped_time = df.query('gc_vel_inst > -0.5 & gc_vel_inst < 0.5')['delta_time'].sum()
    mean_speed = round(df['gc_vel_inst'].mean() * ms2mph,1)
    moving_time = elapsed_time-stopped_time
    calories=(df.delta_time.dt.seconds*df.power).sum()/(4184*.24) 
    
    print('Total distance traveled: ', dist_total, ' miles' )
    print('Total Elevation Gain: ', ele_gain, ' feet')
    #print('Average grade: ', round(df['grade'].mean(),2), ' deg')
    print('Max speed: ', max_speed, ' mph')
    print('Mean speed: ', mean_speed, ' mph')
    print('Time elapsed: ', elapsed_time)
    print('Moving time: ', moving_time)
    print('Avergae Power: ', round(df.power.mean(),2), 'W')
    print('Energy produced: ', round(calories,1), ' Calories')
    #return elapsed_time, stopped_time   
        
def power_calc(df):
    my_mass = 72.6 #kg
    grav = 9.81 #m/s^2  
    bike_mass = 13.6 #kg
    Crr = 0.0063 # Coefficient of rolling resistance
    Cd_area = .307 # frontal area * drag Coeff for dropbars 
    wind_vel = 4.4 # m/s
    rho = 1.225 * np.exp(-0.00011856 * df.ele)
    Fg = grav * np.sin(df.grade) * (my_mass + bike_mass)
    Fr = grav * np.cos(df.grade) * (my_mass + bike_mass) * Crr
    Fa = 0.5 * Cd_area * rho * (df.gc_vel_inst + wind_vel)**2
    loss = 0.04
    power = (Fg + Fr + Fa) * df.gc_vel_inst / (1-loss)
    df['power'] = power
    
#%% dialog to grab file and parse as gpx object

root = tk.Tk()
root.withdraw()
src_file = filedialog.askopenfilename()

gpx_file = open(src_file, 'r')

gpx = gpxpy.parse(gpx_file)


# parses gpx file and creates dataframe of pertinent info
lst = list()

for track in gpx.tracks:
    for segment in track.segments:
        for i,point in enumerate(segment.points):
            # create a list of important data in gpx file
            lst.append([point.latitude, point.longitude, point.elevation, point.time])
            
df = pd.DataFrame(lst, columns = ['lat','lon','ele','timestamp'])            

del(lst)
#%%
# latitude converted to radians
df = df.assign(lat_rad = lambda df:df.lat*np.pi/180)

# longitude converted to radians  
df = df.assign(lon_rad = lambda df:df.lon*np.pi/180)
           
df = df.drop([len(df)-1])
df['delta_dist'] = gc_dist(df)
df['delta_time'] = df.diff()['timestamp']
df['delta_z'] = df.diff()['ele']




diff_df = df.diff().drop([0])
shift_df = df.shift(periods=-1).drop([len(df)-1])

a = 6378.137000 #km

r1 = a*(1-np.e**2)/(1-np.e**2*(np.sin(df['lat'][0]))**2)**(3/2)
r2 = a/np.sqrt(1-np.e**2*(np.sin(df['lat'][0]))**2)    


distance_north = r1*diff_df['lat']
distance_east = r2*np.cos(df['lat'][0])*diff_df['lon']

distance = np.sqrt(distance_north**2 + distance_east**2)

print('Flat Earth Approx distance: ', distance.sum(), 'm')
#%%    


# fn to calculate instantaneous grade in degrees
grade_calc(df)

# create a column of instantaneous velocities in mph
df = df.assign(gc_vel_inst = lambda df: (df.delta_dist/df.delta_time.dt.seconds)) 

# fn to calculate power in Watts
power_calc(df)

# need to fill NaN's with 0 value for maths later
df.at[0,'delta_dist'] = 0.0 
df.at[0,'delta_z'] = 0.0
df.at[0,'delta_time'] = timedelta(0)    
df.at[0,'gc_vel_inst'] = 0.0
 

# mask = (df['power'] > df.power.quantile(.01)) & (df['power'] <= df.power.quantile(.99))
# power = df.power.loc[mask]

stat_printer(df)
#%%

#%% 3-d plotter
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(df.ele.min(), df.ele.min())
xline = np.linspace(df.lat.min(), df.lat.min())
yline = np.linspace(df.lon.min(), df.lon.min())
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = df.ele
xdata = df.lat 
ydata = df.lon
ax.scatter3D(xdata, ydata, zdata, c=zdata)
ax.plot(xdata,ydata)

#%%

my_filter = KalmanFilter(dim_x=2, dim_z=1)

# initialize the filter matrices

my_filter.x = np.array([[2.],
                [0.]])       # initial state (location and velocity)

my_filter.F = np.array([[1.,1.],
                [0.,1.]])    # state transition matrix

my_filter.H = np.array([[1.,0.]])    # Measurement function
my_filter.P *= 1000.                 # covariance matrix
my_filter.R = 5                      # state uncertainty
my_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.1) # process uncertainty









