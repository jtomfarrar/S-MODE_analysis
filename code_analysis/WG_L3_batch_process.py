# %% [markdown]
# # Wave Glider Level 3 batch processing
# ## interp small gaps, average to 1 min, save files
# 



# %% 
# Set some parameters
savefig = True # set to true to save plots as file
figdir = '../plots/WG_L3_processing/' # set path to save plots

# Set paths and filenames
WG = 'WHOI43'#'Stokes'#'Kelvin'#
campaign = 'IOP1' # 'PFC' # 

# %%  
# Import modules
import os
import WG_L3_functions as WG_L3



# %%
WG_L3.WG_L3_processor_function(campaign,WG,savefig=savefig,figdir=figdir)

# %%
# I believe that xr.resample returns time labels that are the beginning of the averaging period
# To verify this, I will need some simple test; I don't know if this one is a good test
'''
foo = ds.time_1Hz.resample(time_1Hz = '1 min').mean()
time_diff = foo[0]-foo[0].time_1Hz
#express time_diff in seconds with 3 significant figures
print('Center of time interval is shifted by '+str(np.round(time_diff.values/np.timedelta64(1,'s'),7))+' seconds')
'''
