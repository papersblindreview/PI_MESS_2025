import os 
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gc
import sys
from functions import *

np.random.seed(1)

lake_obs = load_lake_obs()

#########################################################################################################
states = lake_obs['state']
names = lake_obs['lake_name']
gcm_cell = lake_obs['driver_gcm_cell_no']
lake_obs.drop(columns=['state','lake_name','driver_gcm_cell_no'], inplace=True)

unique_sites = lake_obs.site_id.value_counts().index

lags = np.arange(1,11) # lags for each meteorological driver
rollings = [14,30,60,90] # windows over which to take rolling averages 
train_stop_year = 2015  # training only up to 2015

## CREATING FULL DATASET TO INTERPOLATE / EXTRAPOLATE

# load feature_names
with open('../data/feature_names.pkl', 'rb') as f:
  feature_names = pickle.load(f)
  
time_features = feature_names['time']
space_features = feature_names['space']

time_vars, space_vars, depth, max_depth, temperature, year = load_data()
datasets, scales = split_data(time_vars, space_vars, depth, max_depth, temperature, year)

# only create full input dataset after 2015
lake_obs = lake_obs[(lake_obs.date.dt.year > 2015)]

# we only create full datasets for lakes with observations for more than 10 days
lake_subset = lake_obs.groupby("site_id").filter(lambda g: g["date"].nunique() > 10)
lake_subset.reset_index(inplace=True)
lake_subset.dropna(inplace=True)

unique_sites = lake_subset.site_id.unique()

print('Constructing full validation set...')
sys.stdout.flush()
list_full_dfs = Parallel(n_jobs=-1, verbose=5)(delayed(match_weather)(site, full=True) for site in unique_sites)
full_df = pd.concat(list_full_dfs, ignore_index=True).reset_index(drop=True).dropna()
sys.stdout.flush()
print('Done.\n')

time_full = (full_df[time_features] - scales['mins_time']) / (scales['maxs_time']-scales['mins_time'])
space_full = (full_df[space_features] - scales['mins_space']) / (scales['maxs_space']-scales['mins_space'])
 

np.savez(f'../data/full_data.npz', 
         features_time=time_full.to_numpy(),
         features_space=space_full.to_numpy(),
         max_depth=full_df[['max_depth']].to_numpy(),
         dates=full_df[['date']].to_numpy(),
         lonlat=full_df[['lon','lat']].to_numpy()) 

with open(f'../data/full_data_info.pkl', 'wb') as f:
  pickle.dump(list(full_df.site_id), f)  
