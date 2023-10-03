# %% [markdown]
# ## Download and clean up near-real-time Wave Glider data

# %% [markdown]
# * read in data
# 
# first cut by Tom, 10/18/2021  
# Updated for IOP1, 10/9/2022

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cftime
import requests
import cartopy.crs as ccrs                   # import projections
import cartopy
import gsw
import functions  # requires functions.py from this directory

# %%
# Set up plotting parameters
plt.rcParams['figure.figsize'] = (7,4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 400
plt.close('all')

savefig = True # set to true to save plots as file
# directory for saving figures:
__figdir__ = '../plots/WG_IOP1' 
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# make figure directory if it doesn't exist
is_path = os.path.exists(__figdir__)
if not is_path:
    os.makedirs(__figdir__)

# %%
zoom = True
if zoom:
    xmin, xmax = (-127,-121)
    ymin, ymax = (36.25,38.5)
    levels = np.linspace(14,17,21)-2.5
else:
    xmin, xmax = (-127,-121)
    ymin, ymax = (35, 41)
    levels = np.linspace(13,18,11)

    

# %% [markdown]
# I made a symbolic link to the data directory
# ln -s /mnt/d/tom_data/S-MODE/S-MODE_data/final/IOP1 ./S-MODE_analysis/data/raw/IOP1

# %%
# List of WGs
input_list = ['WHOI22','WHOI32','WHOI43','Stokes', 'Planck', 'Pascal', 'Kelvin', 'CARSON']
input_path = '../data/raw/IOP1/Wave_Gliders/'

file_prefix = 'SMODE_IOP1_Wavegliders_'
file_postfix = '.nc'
WG_list = ['WHOI22','WHOI32','WHOI43','STOKES', 'PLANCK', 'PASCAL', 'KELVIN', 'CARSON']

outpath='../data/raw/WG_/'
# %%
'''tab1_postfix = '_PLD2_TAB1.nc'
tab2_postfix = '_PLD2_TAB2.nc'
position_postfix = '_position.nc'
WG_list = ['WHOI22','WHOI32','WHOI43','STOKES', 'PLANCK', 'PASCAL', 'KELVIN', 'CARSON']
'''
# %%

n=0
for WG in WG_list:
    input_WG=input_list[n]
    # Read in each WG file
    file = input_path + file_prefix + input_list[n] + file_postfix
    varstr = 'ds_'+WG
    locals()[varstr]=xr.open_dataset(file,decode_times=True)
    n=n+1
    print(file)


#%%

eval('ds_'+WG)

# %%


# %%
# Now we can access these in a loop using syntax like:
# eval('adcp_'+WG_list[7])

# %%
eval('met_'+WG_list[0])

# %% [markdown]
# OK, we have many different variables on many different time bases. Let's assess. 
# We have time_1Hz, time_10Hz, time_20Hz, time_15min, Workhorse_time (1 Hz)
# For most analysis purposes, we will want a 5 or 15 minute time base.  Let's make a L3 product with 
# all variables on a 5 minute time base.
#
# First, let's look at the variables in each file.  We can use the xarray dataset method .data_vars to 
# list all the variables in the dataset.  We can also use .variables to list all the variables and their attributes.  Let's try that for the first file.
#
# We can also use the xarray dataset method .info() to get a summary of the dataset.  
#
# Note that I went through a lot of the necessary steps to regrid data in code/WG_L2_look.ipynb,  
# code/WG_L2b_look.ipynb, code/WG_L3_processing.ipynb, code/WG_L3a_processing.ipynb, and code/WG_L3b_processing.ipynb, 
#
# Maybe this is the latest best example:
# Python/S-MODE_analysis/code_IOP2/WG_realtime_met.ipynb
#
# There are definitely soem relevant steps and functions in Python/S-MODE_analysis/code/WG_L3b_processing.ipynb, 
# especially make_var_list(ds_in,time_coord) (which makes a list of all variables with a given time coord),
#  remove_nan(), subset(var_list, ds_in), and add_vars(var_list, ds_in, ds_out)


# %%
# make a list of all data variables in the xarray dataset
ds_WHOI22.data_vars

#ds_WHOI22.variables


# %%
#Compute density from T and cond
p = 1
for WG in WG_list:
    ds = eval('met_'+WG)
    ds['uctd_psu_Avg']=gsw.conversions.SP_from_C(10*ds.uctd_cond_Avg, ds.uctd_temp_Avg, p)
    SA = gsw.conversions.SA_from_SP(ds.uctd_psu_Avg, 1,ds.longitude_1hz_Avg, ds.latitude_1hz_Avg)
    CT = gsw.conversions.CT_from_t(SA, ds.uctd_temp_Avg, p)
    ds['uctd_sigma0_Avg'] = gsw.density.sigma0(SA, CT)
    varstr = 'met_'+WG
    locals()[varstr]= ds

# %% [markdown]
# OK, now let's look at RDI files (Table 2)

# %% [markdown]
# OK, we have 15 minute files from the ADCP and 5 minute from the position files.  Interpolate the position files to the ADCP times.  That should be easy using xarray interp package, following:  
# https://docs.xarray.dev/en/stable/user-guide/interpolation.htmlhttps://docs.xarray.dev/en/stable/user-guide/interpolation.html  
# 
# ```
# new_lon = -126.1
# new_lat = 37.1
# new_time = ds.time[-3]
# dsi = ds.interp(time=new_time,latitude=new_lat, longitude=new_lon)
# ```
# 
# ```
# new_time = ds_adcp.time
# ds_pos_i = ds_pos.interp(time=new_time)
# ```

# %%
# Interpolate each WG's position to ADCP time and add to ADCP file
for WG in WG_list:
    ds_adcp = eval('adcp_'+WG)
    ds_pos = eval('pos_'+WG)
    ds_pos_i = ds_pos.interp(time=ds_adcp.time)
    ds_adcp['Longitude']=ds_pos_i.Longitude
    ds_adcp['Latitude']=ds_pos_i.Latitude
    varstr = 'adcp_'+WG
    locals()[varstr]= ds_adcp
    del ds_adcp

# %% [markdown]
# OK, that's very cool!  I have all the files cleaned up and have added the lat/lon.  Let's save the cleaned up files for met and adcp.  First, add z for adcp files.

# %%
for WG in WG_list:
    fout = outpath + 'adcp_'+WG + '.nc'
    ds_adcp = eval('adcp_'+WG)
    ind=np.flatnonzero(np.isnan(ds_adcp.z_matrix[1][:])==False)
    if WG=='PASCAL': #Special case because PASCAL has 600 kHz RDI
        depth = ds_adcp.z_matrix[:,ind[0]]/2
    else:
        depth = ds_adcp.z_matrix[:,ind[0]]
    ds_adcp['depth'] = depth
    ds_adcp.to_netcdf(fout)



# %%


# %%
ds_adcp

# %%
vmin = -0.5
vmax = 0.5
fig = plt.figure()
plt.set_cmap(cmap=plt.get_cmap('turbo'))
# ax1 = plt.subplot(len(WG_list),1,len(WG_list))
# ax1.set_xlim(tmin,tmax)
ds = adcp_CARSON
im = plt.pcolor(ds.time.values,ds.z_matrix,ds.current_north,vmin=vmin,vmax=vmax)
# plt.contourf(ds.time.values,ds.z_matrix[:,1],ds.current_east,levels)
plt.ylim(-60, 0)
plt.title(' Carson North vel')
fig=plt.gcf()
fig.autofmt_xdate()


# %%
ds.time[-1]

# %%



