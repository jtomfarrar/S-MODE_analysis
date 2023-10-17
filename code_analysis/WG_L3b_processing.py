# %% [markdown]
# # Wave Glider Level 3b processor
# ## smooth, subsample, save subsets; primarily met and vel
# 

# %% [markdown]
# ## About Level 3a and Level 3b data
# 
# Level 3a data:
# - The only difference from Level 2 is that NaNs are removed by interpolation
# - This is a "wave resolving" data set with modest QC (removal/interpolation of gaps<1 sec)
# 
# Level 3b data:
# - Same as Level 3a but variables are averaged to 1 min
# - subsample all variables to 1-min time base
# 
# Level 3b velocity data (skip?):
# - Velocity data and averaged to 5 minutes
# 
# ## This notebook is for level 3b data
# 1. Load L3a data files
# 1. put all variables on continuous (uniformly spaced) time base
# 1. Do L3b processing (smooth, subsample, save subsets; primarily met and vel)
# 
# 

# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib
import datetime as dt
################
# This allows us to import Tom_tools
import sys
sys.path.append('../../Tom_tools/') # you may need to adjust this path
# sys.path.append('../SWOT_IW_SSH/jtf/Tom_tools/') # you may need to adjust this path
import Tom_tools_v1 as tt
################

# %%
# %matplotlib inline
%matplotlib qt 
savefig = False # set to true to save plots as file
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.close('all')
plt.rcParams['axes.xmargin'] = 0
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

__figdir__ = '../plots/WG_timeseries/'
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# %%
path = '/mnt/d/tom_data/S-MODE/S-MODE_data/final/PFC/Wavegliders/L3a/'
path_out = '/mnt/d/tom_data/S-MODE/S-MODE_data/final/PFC/Wavegliders/L3b/'
# make directory if it doesn't exist
if not os.path.exists(path_out):
    os.makedirs(path_out)

WG = 'Kelvin'#'Stokes'#'WHOI43'#

file =  WG+'_L3a.nc'

# %%

ds = xr.open_dataset(path+file, engine = 'netcdf4', decode_times = True) #decode_times = False, 

# %%
def make_var_list(ds_in,time_coord):
    """
    Find all the variables with a given time coord
 
    Parameters
    ----------
    ds_in : xarray.dataset
    time_coord : str 
    (time_coord = None will return all variables that do not have a time coord)

    Returns
    -------
    result : list of str
        list of vars meeting criterion
    """
    
    var_list = []  
    not_used = []  
    for var in ds_in.data_vars:
        try:
            if ds_in.data_vars.get(var).dims[0]==time_coord:
                var_list.append(var)
                print(var)
        except Exception: # if there is no time coord, just skip it
            not_used.append(var)

    if time_coord is None:
        var_list = not_used

    return var_list

# %%
def remove_nans(var_list, ds_in):
    """
    remove all nans in variables in an input xarray.dataset
 
    Parameters
    ----------
    ds_in : xarray.dataset
    var_list : list of strings

    Returns
    -------
    result : revised xarray.dataset
    """
    #t = ds_in[ds_in.data_vars.get(var).coords.dims[0]]
    #t_noninterp = t[ff]

    for var in var_list:
        var_raw = ds_in.data_vars.get(var).copy()
        ff = np.flatnonzero(np.isnan(var_raw)==0)
        var_nanfree = var_raw[ff]

        numnans = np.size(np.flatnonzero(np.isnan(var_raw)))
        locals()[var] = var_raw.rename(var) #locals()['string'] makes a variable with the name string
        print(var+' created, '+str(np.round(100*(1-numnans)/np.size(t),3))+'% of values are NaN'+', number of nans=' + str(numnans))
        locals()[var].values = var_nanfree
        ds_in[var] = locals()[var]
    return ds_new

# %%
def subset(var_list, ds_in):

    ds_new = ds_in[var_list]
    return ds_new

# %%
def add_vars(var_list, ds_in, ds_out):
    '''
    Copies variables in var_list from ds_in to ds_out.  
    '''
    var_existing = []
    for var in ds_in.data_vars:
        var_existing.append(var)

    #ds_out = ds_out[var_existing]
    ds_out[var_list] = ds_in[var_list].copy()
    return ds_out

# %%
'''
ds_new = xr.Dataset().assign_attrs(ds.attrs)  # make empty xr.Dataset but copy attributes from original file
ds_WH = xr.Dataset().assign_attrs(ds.attrs)  # make empty xr.Dataset but copy attributes from original file
'''

# %%
# ds_new = add_vars(var_list, ds, ds_new)

# %%
# Write data to netcdf file
'''
new_file = path_out+WG+'_L3a.nc'
print ('saving to ', new_file)
ds_new.to_netcdf(path=new_file)
ds_new.close()
print ('finished saving')

# Write Workhorse ADCP data to netcdf file
new_file = path_out+WG+'_L3a_ADCP.nc'
print ('saving to ', new_file)
ds_WH.to_netcdf(path=new_file)
ds_WH.close()
print ('finished saving')
'''


# %%
# compare different time vectors to decide how to line up variables from different time bases
# ds.time_20Hz
# ds.time_1Hz
# ds.Workhorse_time

# I think they are already nearly lined up
ds.time_1Hz[0:10:1]-ds.time_20Hz[0:200:20]
# so, try 0,20,40... in 20 Hz array

tfoo = ds.time_20Hz[0:-1:20]
#tfoo and time_1Hz are very close, within 10^-4 sec

# %%
nsec=60

# %%
# ds.time_20Hz variables
# wdir is positive clockwise from North
# need to make U, V
wind_speed_low = tt.run_avg1d(ds.wind_speed, nsec*20)
wind_east_low = tt.run_avg1d(ds.wind_east,nsec*20)
wind_north_low = tt.run_avg1d(ds.wind_north,nsec*20)
pitch_low = tt.run_avg1d(ds.pitch_20Hz,nsec*20)
roll_low = tt.run_avg1d(ds['roll_20Hz'],nsec*20)

# %%
# ds.time_1Hz variables
WXT_U=ds.WXT_wind_speed*np.cos(ds.WXT_wind_direction/np.pi/2)
WXT_V=ds.WXT_wind_speed*np.sin(ds.WXT_wind_direction/np.pi/2)
WXT_U_low = tt.run_avg1d(WXT_U,nsec)
WXT_V_low = tt.run_avg1d(WXT_V,nsec)
WXT_wspd_low = tt.run_avg1d(ds.WXT_wind_speed,nsec)
WXT_atmp_low = tt.run_avg1d(ds.WXT_air_temperature,nsec)
WXT_rh_low = tt.run_avg1d(ds.WXT_relative_humidity,nsec)
lat_low = tt.run_avg1d(ds.latitude_1Hz,nsec)
lon_low = tt.run_avg1d(ds.longitude_1Hz,nsec)

# Not all files have SWR and LWR
try:
    swr_low = tt.run_avg1d(ds.SMP21_shortwave_flux,nsec)
    lwr_low = tt.run_avg1d(ds.SGR4_longwave_flux,nsec)
except Exception:
    print('No SWR or LWR in this file')

# %%
# Workhorse_time variables
U_low=tt.run_avg2d(ds.Workhorse_vel_east,nsec,1)
V_low=tt.run_avg2d(ds.Workhorse_vel_north,nsec,1)
W_low=tt.run_avg2d(ds.Workhorse_vel_up,nsec,1)

# %%
# Subsample smoothed 20 Hz vars to 1 Hz
wind_speed_low = wind_speed_low[0::20]
wind_east_low = wind_east_low[0::20]
wind_north_low = wind_north_low[0::20]
pitch_low = pitch_low[0::20]
roll_low = roll_low[0::20]


# %%
np.shape(wind_speed_low)
np.shape(wind_east_low)


# %%
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_1Hz,pitch_low)
plt.plot(ds.time_1Hz,roll_low)
plt.title(str(nsec/60)+'-min average pitch and roll')
plt.ylabel('Degrees')
plt.ylim(-2,2)

if savefig:
    plt.savefig(__figdir__+WG+'_mean_pitch_roll' + '.' +plotfiletype,**savefig_args)


# %%
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_1Hz,wind_speed_low)
plt.plot(ds.time_1Hz,WXT_wspd_low)
plt.title(str(nsec/60)+'-min average wind speed')
plt.ylabel('Degrees')
#plt.ylim(-2,2)

if savefig:
    plt.savefig(__figdir__+WG+'_mean_wspd' + '.' +plotfiletype,**savefig_args)


# %%


# %%
'''
new_file = path_out+WG+'_L3b_met.nc'
print ('saving to ', new_file)
ds_new.to_netcdf(path=new_file)
ds_new.close()
print ('finished saving')
'''

# %%
WG

# %%
# Plot spectra of 'raw' L3a wind speed
fig, axs = plt.subplots(1, 1)
M=151
wspd=ds.wind_speed.values
wspd_WXT=ds.WXT_wind_speed.values
tt.spectrum_band_avg(wspd,1/20,M,winstr=None,plotflag=True,ebarflag=None)
tt.spectrum_band_avg(wspd_WXT,1,M,winstr=None,plotflag=True,ebarflag=False)
tt.spectrum_band_avg(wind_speed_low,1,M,winstr=None,plotflag=True,ebarflag=False)
plt.title('Gill, WXT wind speed')
plt.xlabel('Hz')
plt.ylabel('Spectral density [m$^2$/s$^2$/Hz]')
plt.legend(['Gill Wind speed','WXT wind speed'],loc='upper right')
if savefig:
    plt.savefig(__figdir__+WG+'_wsp_spectra' + '.' +plotfiletype,**savefig_args)


# %% [markdown]
# ## Preparing for flux computation
# 
# Inputs for COARE 3.5:  
#     u = ocean surface relative wind speed (m/s) at height zu(m)  
#     t = bulk air temperature (degC) at height zt(m)  
#     rh = relative humidity (%) at height zq(m)  
#     ts = sea water temperature (degC) - see jcool below  
#     P = surface air pressure (mb) (default = 1015)  
#     Rs = downward shortwave radiation (W/m^2) (default = 150)  
#     Rl = downward longwave radiation (W/m^2) (default = 370)  
#     zu = wind sensor height (m) (default = 18m)  
#     zt = bulk temperature sensor height (m) (default = 18m)  
#     zq = RH sensor height (m) (default = 18m)  
#     lat = latitude (default = 45 N)  
#     zi = PBL height (m) (default = 600m)  
#     rain = rain rate (mm/hr)  
#     cp = phase speed of dominant waves (m/s)  
#     sigH =  significant wave height (m)  
#     jcool = cool skin option (default = 1 for bulk SST)  
#     
# Note: I don't see an input for SST depth--> That's because there is no Warm Layer correction in this version of the code.  
# 
# Here is a link to an older matlab version that does have it:  
# https://github.com/carsonwitte/Falkor-DWL-Code  
# 
# It wouldn't be that hard to add the WL correction-- basically, estimate fluxes, estimate WL correction, and re-estimate fluxes.  
# 
# 
#   
#   
# Required inputs:  
# u	zu	t	zt	rh	zq	P	ts	Rs	Rl	lat	zi	rain	cp	sigH
# 

# %% [markdown]
# This could be a good way to save the output, but maybe not:  
# 
# ```
# A = coare35vn(u, ta, rh, ts, P=Pa, Rs=rs, Rl=rl, zu=16, zt=16, zq=16,
#                 lat=Lat, zi=ZI, rain=Rain, jcool=1)
# fnameA = os.path.join(path,'test_35_output_py_04022022.txt')
# A_hdr = 'usr\ttau\thsb\thlb\thlwebb\ttsr\tqsr\tzot\tzoq\tCd\t'
# A_hdr += 'Ch\tCe\tL\tzet\tdter\tdqer\ttkt\tRF\tCdn_10\tChn_10\tCen_10'
# np.savetxt(fnameA,A,fmt='%.18e',delimiter='\t',header=A_hdr)
# ```
# 

# %%


# %%



