# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:40:48 2021

@author: Vega Briones, Jorge

This code was made to read csv data from CAMELS dataset, which provides 
attributes and Gauged measurements as discharge, temperature, precipitation,etc.

The Analysis devolopen on this code are:
- Application of categorical variables in CAMEL's dataset.
- Determination of drought periods with Fixed and Variable thresholds.
- Determinarion of discharge curves.

.....

"""

#%% Load packages

import os
import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import matplotlib.collections as collections

# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import sys

from datetime import datetime, timedelta
import math

#%% Load and clean data

# In which folder I am working?
pathName = os.getcwd()

## Import Attribute Information
df_At = pd.read_csv('h:\\SIG\\Hidrología\\CAMELScl\\CAMELS_cactchmentboundaries_attributes_V00.csv', sep=',', encoding='cp1252', header = 0, index_col = 0)
print(df_At.head(5))
index_gauge = df_At.index

## Import Discharge (m3/s)
df_Q = pd.read_csv('h:\\SIG\\Hidrología\\CAMELScl\\2_CAMELScl_streamflow_m3s\\2_CAMELScl_streamflow_m3s.csv',header = 0, index_col = 0)
print(df_Q.head(5))
#df_Q_m3s = df_Q.astype(np.float)
df_Q = df_Q.transpose()
print(df_Q.head(5))
# Define index
df_Q.index=index_gauge
#df_Q.set_index('gauge_id',inplace=True)
print(df_Q.head(5))

#%% MERGE ATTRIBUTES with DISCHARGE

# Merge
df_Q_At = pd.merge(df_At, df_Q, left_index=True, right_index=True)
print(df_Q_At.head(5)) 
#Drop 
df_Q_At.columns.values #['gauge_name', 'area_km2', 'record_ini', 'record_end', 'bigdam', 'gauge_lat', 'gauge_lon', 'n_obs', 'gauge_elev', 'elev_mean', 'location, 'snow_frac', 'lc_glacier', 'region_cl']
df_Q_At = df_Q_At.drop(columns = ['gauge_name', 'area_km2', 'record_ini', 'record_end', 'bigdam', 'gauge_lat', 'gauge_lon', 'n_obs', 'gauge_elev', 'elev_mean', 'location', 'snow_frac', 'lc_glacier', 'region_cl'])
print(df_Q_At.head(5)) 
df_Q_At = df_Q_At.transpose()
print(df_Q_At.head(5)) #38374 rows x 7 columns
# See types
print(df_Q_At.dtypes)

# Change to numeric
df_Q_At = df_Q_At.apply(pd.to_numeric, errors = 'coerce')
print(df_Q_At.dtypes)
print(df_Q_At.head(5))

# Transform index to datetime (from 1913/02/15 to 2018/03/09)
Arange = pd.date_range('1913-02-15', '2018-03-09', freq='1d')
df_Q_At['Time']=pd.to_datetime(Arange) # Carefull with the order of date: YYYY-MM-DD
df_Q_At = df_Q_At.set_index('Time')
print(df_Q_At.head(5))


#%% COUNT AVAILABLE DATA

#count = df_Q_At.dropna(axis=0,how='any') #change its not ANY


#%% Time Selection

# Select last 30 years
df_Nan30 = df_Q_At['1988-03-09':'2018-03-09']

# Select last 60 years
df_Nan60 = df_Q_At['1958-03-09':'2018-03-09']

# Select last 90 years
df_Nan90 = df_Q_At['1928-03-09':'2018-03-09']

#%% SELECT NON NULL DATA SERIES

# Missing data tolerance = 90%
# 10958*0.9 = 9.862 missing data

# Average of non-missing data for all dataset
df_NanALL = df_Q_At.notnull().mean().round(4) * 100
df_NanALL = df_NanALL[df_NanALL.iloc[:]>= 90]
df_NanALL = df_NanALL.rename('Qp', inplace=True)
print(df_NanALL)#2

# Average of non-missing data for last 90 years
df_Nan90 = df_Nan90.notnull().mean().round(4) * 100
df_Nan90 = df_Nan90[df_Nan90.iloc[:]>= 90]
df_Nan90 = df_Nan90.rename('Qp', inplace=True)
print(df_Nan90)#6

# Average of non-missing data for last 60 years
df_Nan60 = df_Nan60.notnull().mean().round(4) * 100
df_Nan60 = df_Nan60[df_Nan60.iloc[:]>= 90]
df_Nan60 = df_Nan60.rename('Qp', inplace=True)
print(df_Nan60)#33

# Average of non-missing data for last 30 years
df_Nan30 = df_Nan30.notnull().mean().round(4) * 100
df_Nan30 = df_Nan30[df_Nan30.iloc[:]>= 90]
df_Nan30 = df_Nan30.rename('Qp', inplace=True)
print(df_Nan30) #148


#%% LENGHT of TIME SERIES

#df_Nan90 = df_Nan90.index[:].astype('int')

gauges30 = pd.merge(df_Nan30, df_Q, left_index=True, right_index=True)
gauges30 = gauges30.drop(columns = ['Qp'])
print(gauges30)
df30 = pd.merge(gauges30,df_At, left_index=True, right_index=True)

#%% Apply filters

# 1st REGION

df30_4 = df30.loc[df30['region_cl'] == 4]
df30_5 = df30.loc[df30['region_cl'] == 5]
df30_6 = df30.loc[df30['region_cl'] == 6]
df30_7 = df30.loc[df30['region_cl'] == 7]
df30_8 = df30.loc[df30['region_cl'] == 8]
df30_9 = df30.loc[df30['region_cl'] == 9]
df30_13 = df30.loc[df30['region_cl'] == 13]
df30_15 = df30.loc[df30['region_cl'] == 15]


#

# 2nd - Natural/Not Natural (Dams = 1/NotDams = 0)
df30_4_Nat = df30_4.loc[df30_4['bigdam'] == 0]
df30_4_Dam = df30_4.loc[df30_4['bigdam'] == 1]

df30_5_Nat = df30_5.loc[df30_5['bigdam'] == 0]
df30_5_Dam = df30_5.loc[df30_5['bigdam'] == 1]

df30_6_Nat = df30_6.loc[df30_6['bigdam'] == 0]
df30_6_Dam = df30_6.loc[df30_6['bigdam'] == 1]

df30_7_Nat = df30_7.loc[df30_7['bigdam'] == 0]
df30_7_Dam = df30_7.loc[df30_7['bigdam'] == 1]

df30_8_Nat = df30_8.loc[df30_8['bigdam'] == 0]
df30_8_Dam = df30_8.loc[df30_8['bigdam'] == 1]

df30_9_Nat = df30_9.loc[df30_9['bigdam'] == 0]
df30_9_Dam = df30_9.loc[df30_9['bigdam'] == 1]

df30_13_Nat = df30_13.loc[df30_13['bigdam'] == 0]
df30_13_Dam = df30_13.loc[df30_13['bigdam'] == 1]

df30_15_Nat = df30_15.loc[df30_15['bigdam'] == 0]
df30_15_Dam = df30_15.loc[df30_15['bigdam'] == 1]

# 3th - Source of water
df30_4_Nat_Snow = df30_4_Nat.loc[df30_4_Nat['snow_frac'] > 0]
df30_4_Nat_NoSnow = df30_4_Nat.loc[df30_4_Nat['snow_frac'] == 0]
df30_4_Dam_Snow = df30_4_Dam.loc[df30_4_Dam['snow_frac'] > 0]
df30_4_Dam_NoSnow = df30_4_Dam.loc[df30_4_Dam['snow_frac'] == 0]

df30_5_Nat_Snow = df30_5_Nat.loc[df30_5_Nat['snow_frac'] > 0]
df30_5_Nat_NoSnow = df30_5_Nat.loc[df30_5_Nat['snow_frac'] == 0]
df30_5_Dam_Snow = df30_5_Dam.loc[df30_5_Dam['snow_frac'] > 0]
df30_5_Dam_NoSnow = df30_5_Dam.loc[df30_5_Dam['snow_frac'] == 0]


#%% DROP UN-USED COLUMNS, CHANGE TO NUMERIC, EDIT INDEX

## R4 NAT SOW

# Drop columns
df30_4_Nat_Snow = df30_4_Nat_Snow.drop(columns = ['gauge_name', 'area_km2', 'record_ini', 'record_end', 'bigdam', 'gauge_lat', 'gauge_lon', 'n_obs', 'gauge_elev', 'elev_mean', 'location', 'snow_frac', 'lc_glacier', 'region_cl'])
print(df30_4_Nat_Snow.head(5)) 
df30_4_Nat_Snow = df30_4_Nat_Snow.transpose()
print(df30_4_Nat_Snow.head(5)) #38374 rows x 7 columns

#Change to numerics
print(df30_4_Nat_Snow.dtypes)
df30_4_Nat_Snow = df30_4_Nat_Snow.apply(pd.to_numeric, errors = 'coerce')
print(df30_4_Nat_Snow.dtypes)

# Transform index to datetime (from 1913/02/15 to 2018/03/09)
Arange = pd.date_range('1913-02-15', '2018-03-09', freq='1d')
df30_4_Nat_Snow['Time']=pd.to_datetime(Arange) # Carefull with the order of date: YYYY-MM-DD
df30_4_Nat_Snow = df30_4_Nat_Snow.set_index('Time')
print(df30_4_Nat_Snow.head(5))


## R4 NAT NOSNOW

# Drop columns
df30_4_Nat_NoSnow = df30_4_Nat_NoSnow.drop(columns = ['gauge_name', 'area_km2', 'record_ini', 'record_end', 'bigdam', 'gauge_lat', 'gauge_lon', 'n_obs', 'gauge_elev', 'elev_mean', 'location', 'snow_frac', 'lc_glacier', 'region_cl'])
print(df30_4_Nat_NoSnow.head(5)) 
df30_4_Nat_NoSnow = df30_4_Nat_NoSnow.transpose()
print(df30_4_Nat_NoSnow.head(5)) #38374 rows x 7 columns

#Change to numerics
print(df30_4_Nat_NoSnow.dtypes)
df30_4_Nat_NoSnow = df30_4_Nat_NoSnow.apply(pd.to_numeric, errors = 'coerce')
print(df30_4_Nat_NoSnow.dtypes)

# Transform index to datetime (from 1913/02/15 to 2018/03/09)
Arange = pd.date_range('1913-02-15', '2018-03-09', freq='1d')
df30_4_Nat_NoSnow['Time']=pd.to_datetime(Arange) # Carefull with the order of date: YYYY-MM-DD
df30_4_Nat_NoSnow = df30_4_Nat_NoSnow.set_index('Time')
print(df30_4_Nat_NoSnow.head(5))

#%% DROUGHT ANALYSIS - FIXED THRESHOLD

## Fixed threshold
Q_df30_4_Nat_Snow = df30_4_Nat_Snow.iloc[:,0] # Select gauge 4302001!! Carefull for next steps!
#type(df30_4_Nat_Snow.gauge_id[0])


#pd.DataFrame(df30_4_Nat_Snow, columns=list('Q')) #Not sure if this goes here
Q_df30_4_Nat_Snow = Q_df30_4_Nat_Snow.rename('Q', inplace=True)
Q_df30_4_Nat_Snow.astype

# Find threshold
Q_df30_4_Nat_Snow = df30_4_Nat_Snow.dropna()
FT = np.percentile(Q_df30_4_Nat_Snow, 5)
print(FT)
# Apply threshold
FT_df30_4_Nat_Snow = Q_df30_4_Nat_Snow[Q_df30_4_Nat_Snow < FT]

#%% DROUGHT ANALYSIS - count drought periods (MISSING!!)

# Mind frame
#count (FT_df30_4_Nat_Snow.count) if FT_df30_4_Nat_Snow.index > 5

FT_df30_4_Nat_Snow['Time'] = pd.to_datetime(FT_df30_4_Nat_Snow.index, errors='coerce')#Dont know if change index to datetime is necessary
#FT_df30_4_Nat_Snow = FT_df30_4_Nat_Snow.set_index('Time') # Doesnt work
#FT_df30_4_Nat_Snow = FT_df30_4_Nat_Snow[:,0] #Doesnt work

# To count the consecutive days
s = FT_df30_4_Nat_Snow.groupby(FT_df30_4_Nat_Snow.index).diff().dt.days.ne(1).cumsum()
FT_df30_4_Nat_Snow.groupby(['Time', s]).size().reset_index(level=1, drop=True)

#%% DROUGHT ANALYSIS - PLOT

fig, ax = plt.subplots()
ax.set_title('Drought periods')
ax.plot(df30_4_Nat_Snow.index.values, df30_4_Nat_Snow, color='grey')
ax.axhline(y=FT, color = '#d62728')
ax.scatter(FT_df30_4_Nat_Snow.index.values, FT_df30_4_Nat_Snow, c='red')

#collection = collections.BrokenBarHCollection.span_where(FT_df30_4_Nat_Snow, ymin=-1, ymax=0, where=FT_df30_4_Nat_Snow < FT, facecolor='red', alpha=0.5)ax.add_collection(collection)
plt.show()


#%% DROUGHT ANALYSIS - Variable threshold

Drought_year = df30_4_Nat_Snow

Arange = pd.date_range('1913-02-15', '2018-03-09', freq='1d')
Drought_year.index=Arange

Drought_year['Q'] = Drought_year.iloc[:,0]

Drought_year = Drought_year.drop(Drought_year.columns[0:1:2], axis=1)


#Drought_months = Drought_year.iloc[:,0].resample("A").apply(['mean'])

#Drought_months['Q'] = Drought_months['mean']

#Drought_months = Drought_months.drop(columns=['mean'])

#1st
#Drought_year.index = Drought_year.index.strftime('%Y/%m/%d')
months=Drought_year.index.month # Not working! 'Index' object has no attribute 'month'

#2nd
#Arange = pd.date_range('1913-12-31', '2018-12-31', freq='1M')
#Drought_year.index=Arange

#3rd
#monthly_avg = Drought_months.groupby(months).Q.mean()

#4th

#Drought_months2 = Drought_months.reset_index(drop=True)
Drought_year['date'] = Drought_year.index

Drought_year.set_index('date',inplace=True)

monthly_avg = Drought_year.groupby(months).Q.mean()

#df30_4_Nat_Snow(df30_4_Nat_Snow.column[1], axis=1)
#months = df30_4_Nat_Snow.groupby('Time').dt.month


# 
df30_4_Nat_Snow.nunique() #only non-nans 
