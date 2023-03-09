## Functions for calculating hydrological signatures from Salwey et al. (2023) 
# Application of the script can be seen in assosiated notebook 
# The functions output signature values as well as variables used in the assosiated plots 


import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import math 
import pandas as pd

#WB

def calc_WB(flow_df, precip_df, pet_df):
    
    # Calculate runoff coefficient
    runoff_coefficient = flow_df.mean(axis = 0) / precip_df.mean(axis = 0)

    # Calculate aridity index 
    aridity_index = pet_df.mean(axis = 0) / precip_df.mean(axis = 0) 
    
    # Calculate WB metric
    WB = aridity_index - (1 - runoff_coefficient)
    
    return WB, runoff_coefficient, aridity_index
    
#Seg-FDC


def calc_SegFDC(flow_df):
        
    # Define sigmoidal function 
    
    def func(x, a, b):
        return a*(-(np.log((x/100)/(1-(x/100)))))+b
        
    # Define function for finding the index of exceedance values at specified percentile     
    
    def find_nearest(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx
    
    # Signature calculation 
    
    Seg_FDC=[]
    
    for gauge in range(0, len(flow_df.columns)): 
        
        # Extract observed FDC's from timeseries 
        sort = np.sort(flow_df.iloc[:, gauge])[::-1]
        sort = sort[np.logical_not(np.isnan(sort))]
        sort = np.where(sort<0.001, 0.001, sort)
        rank= np.arange(1.,len(sort)+1)
        exceedance = []
        for x in range(0,len(rank)): 
            exceedance.append(100*(rank[x]/(len(rank)+1)))
        
        x_data = np.array(exceedance)
        y_obs = np.array(np.log(sort))
        
        # Fit sigmoidal function to observed data 
        popt, pcov = curve_fit(func, x_data, y_obs)
        y_pred = func(x_data, *popt)
        p5 = find_nearest(exceedance, 5)
        p95 = find_nearest(exceedance, 95)
        
        # Calculate signature 
        Seg_FDC.append(mean_squared_error(y_obs[p5:p95], y_pred[p5:p95], squared=False)/np.std(y_obs))
        
    return Seg_FDC
    
#E

def calc_E(dates_df, flow_df, precip_df):
    
    catchments = len(flow_df.columns)
    
    # Copy dataframe for temporary ammendment
    flow_df_copy = flow_df.copy(deep=True)
    flow_df_copy['WY_Y'] = dates_df['WY_y']
    flow_yearly = flow_df_copy.groupby('WY_Y').mean()
    flow_yearly = flow_yearly.iloc[:, 0:catchments]
    
    # Copy dataframe for temporary ammendment
    precip_df_copy = precip_df.copy(deep=True)
    precip_df_copy['WY_Y'] = dates_df['WY_y']
    precip_yearly = precip_df_copy.groupby('WY_Y').mean()
    precip_yearly = precip_yearly.iloc[:, 0:catchments]
    
    # Calulate inter-annual (water year) change in average flow and precipitation for whole timeseries 
    dQ = np.diff(flow_yearly, axis = 0)
    dP = np.diff(precip_yearly, axis = 0)
    
    # This variable is output for the assosiated plot 
    annual_precip = precip_yearly.mean() *365
    
    # Calculate whole timeseries P/Q to normalise signature
    dQ_dP_overall =  precip_yearly.mean() / flow_yearly.mean()
    
    # Calculate inter-annual Q/P 
    dQ_dP = dQ/ dP
    
    # Normalise by P/Q
    for i in range(0, len(dQ_dP)):
        dQ_dP[i,:] =  dQ_dP[i,:] * dQ_dP_overall
        
    # Calculate signature 
    E = np.nanmedian(dQ_dP, axis = 0)
    
    return (E, annual_precip)

# SWRR

def calc_SWRR(dates_df, flow_df, precip_df):
    
    catchments = len(flow_df.columns)
    
    # Copy dataframe for temporary ammendment
    flow_df_copy = flow_df.copy(deep=True)
    flow_df_copy['Month'] = dates_df.Dates.dt.month
    flow_mean_m = flow_df_copy.groupby('Month').mean()
    
    # Copy dataframe for temporary ammendment
    precip_df_copy = precip_df.copy(deep=True)
    precip_df_copy['Month'] = dates_df.Dates.dt.month
    precip_mean_m = precip_df_copy.groupby('Month').mean()[:catchments]
    
    precip_df_copy['WY_Y'] = dates_df['WY_y']
    precip_yearly = precip_df_copy.groupby('WY_Y').mean()
    precip_yearly = precip_yearly.iloc[:, 0:catchments]
    
    # This variable is output for the assosiated plot 
    annual_precip = precip_yearly.mean() *365
    
    # Caculate monthly runoff coefficient 
    runoff_df_m = pd.DataFrame()
    runoff_df_m = flow_mean_m/precip_mean_m
    
    # Adjust for water year 
    # This variable is output for the assosiated plot 
    runoff_df_m_shift = pd.DataFrame(np.roll(runoff_df_m, 3, 0))

    # Calculate signature 
    SWRR = list(runoff_df_m.iloc[[4,5,6,7,8,9], :].mean() / runoff_df_m.iloc[[10,11,0,1,2,3], :].mean())[:131]

    return SWRR, runoff_df_m_shift, annual_precip

# LFV 

def calc_LFV(flow_df, dates_df):
    
    catchments = len(flow_df.columns)
    
    # Copy dataframe for temporary ammendment
    flow_df_copy = flow_df.copy(deep=True)
    flow_df_copy['Month'] = dates_df.Dates.dt.month
    flow_q80_m = flow_df_copy.groupby('Month').quantile(q = 0.20)
    flow_mean_m = flow_df_copy.groupby('Month').mean()
    
    # Calculate timeseries mean to normalise signature 
    overall_mean = flow_mean_m.mean()
    
    # Normalise Q80 
    q80_perc_annual = flow_q80_m/ overall_mean
    
    # This variable is output for the assosiated plot 
    q80_perc_annual_shift = pd.DataFrame(np.roll(q80_perc_annual, 3, 0))
    
    # Calculate signature
    LFV = list(1 -  (q80_perc_annual.max() - q80_perc_annual.min()))[:catchments]
    
    return LFV, q80_perc_annual_shift

