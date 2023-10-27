# %% [markdown]
# # Wave Glider Level 3 batch processing
# ## interp small gaps, average to 1 min, save files
# 



# %% 
# Set some parameters
savefig = True # set to true to save plots as file
figdir = '../plots/WG_L3_processing/' # set path to save plots

# Set paths and filenames
WG = ['Planck'] # it seems like Pascal is taking a really long time to process # 'WHOI43','WHOI22','WHOI32','Stokes','Kelvin','Pascal','CARSON'
campaign = 'IOP1' # 'PFC' # 

# %%  
# Import modules
import os
import WG_L3_functions as WG_L3
from tqdm.contrib.concurrent import thread_map, process_map # allows parallel processing of movies

# %%
# turn (campaign,WG,savefig,figdir) into an iterable
# first make a list of campaign that is the same length as WG
campaign = [campaign]*len(WG)
# now do the same for savefig and figdir
savefig = [savefig]*len(WG)
figdir = [figdir]*len(WG)

foo=zip(campaign,WG,savefig,figdir)
foo2=list(foo)

# %%
# Loop over wave gliders using tqdm.contrib.concurrent to apply the function WG_L3.WG_L3_processor_function()
# with inputs campaign,WG,savefig=savefig,figdir=figdir

result = process_map(WG_L3.WG_L3_processor_function, foo2)
# The line above accomplishes the same thing as the loop below, but it does it in parallel
'''
for entry in foo2:
    WG_L3.WG_L3_processor_function(entry)
'''

# %%
# I believe that xr.resample returns time labels that are the beginning of the averaging period
# To verify this, I will need some simple test; I don't know if this one is a good test
'''
foo = ds.time_1Hz.resample(time_1Hz = '1 min').mean()
time_diff = foo[0]-foo[0].time_1Hz
#express time_diff in seconds with 3 significant figures
print('Center of time interval is shifted by '+str(np.round(time_diff.values/np.timedelta64(1,'s'),7))+' seconds')
'''
