# %% [markdown]
# # Wave Glider Level 3a processing
# ## interp small gaps, generate subsets, save files
# 

# %% [markdown]
# ## About Level 3a and Level 3b data
# 
# The new thing with this processing file (compared to the ones by similar names with L3a and L3b)
# is that I am using xr.resample to average the data to a uniform time base.
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
# ## This notebook is for level 3a data
# 1. Do L3a processing (interp small gaps, generate subsets, save files)
# 
# # Plan for new version of L3a processing
# - Modify make_var_list to also return a list of variables that don't have a time coord
# - Interpolate small gaps in all variables with a time coord
# - Add variables that don't have a time coord to the new dataset
# - Save the new dataset; it will be a complete copy of the old dataset, but with small gaps interpolated
# - At the next level of processing, average all time variables to 1 min (except waves)


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
import os
################

# %%
# %matplotlib inline
# %matplotlib qt 
savefig = False # set to true to save plots as file
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.close('all')
plt.rcParams['axes.xmargin'] = 0


__figdir__ = '../plots/WG_timeseries/'
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# %%
# path = '/mnt/e/SMODE_data/pilot/WG/L2/'
path = '/mnt/d/tom_data/S-MODE/S-MODE_data/final/PFC/Wavegliders/'
# path = '/mnt/c/D_drive/SMODE_data/pilot/WG/L2/'
path_out = '/mnt/d/tom_data/S-MODE/S-MODE_data/final/PFC/Wavegliders/L3a/'
# make directory if it doesn't exist
if not os.path.exists(path_out):
    os.makedirs(path_out)


WG = 'Kelvin'#'Stokes'#'WHOI43'#
file = 'SMODE_PFC_Wavegliders_'+WG+'.nc'

# %%
naninterp='True' # Interpolate NaNs

# %%
# With the preliminary data, it was the case that this needed to be set before 
# any dates are encoded as datetime64-- not sure if it is still needed
# Because the time increment is so small, we get 'out of range' when starting from the standard datetime64 epoch of 1970
# matplotlib.dates.set_epoch('2000-01-01T00:00:00') 


# %%
ds = xr.open_dataset(path+file, engine = 'netcdf4', decode_times = True) #decode_times = False, 

# %%
# xr.resample returns time labels that are the beginning of the averaging period
# To verify this, look at the time labels for the 1 Hz data
foo = ds.time_1Hz.resample(time_1Hz = '1 min').mean()
time_diff = foo[0]-foo[0].time_1Hz
#express time_diff in seconds with 3 significant figures
print('Center of time interval is shifted by '+str(np.round(time_diff.values/np.timedelta64(1,'s'),7))+' seconds')


# %%
# This would work fine, but it requires a 291 GB array
# ds_1min = ds.resample(time_1Hz = '1 min',skipna = True).mean()
# This one crashes the kernel on my laptop
ds_1min = ds.resample(Workhorse_time = '1 min',skipna = True).mean()

# %%
# Raw met plot from WG:
fig, axs = plt.subplots(5, 1, sharex=True)
fig.autofmt_xdate()
plt.subplot(5,1,1)
h1, = plt.plot(ds.time_1Hz, ds.WXT_air_temperature)
h2, = plt.plot(ds.time_1Hz, ds.UCTD_sea_water_temperature)
plt.legend([h1, h2],['Air temp.','SST'])
plt.ylabel('T [$^\circ$C]')
plt.title(WG+': raw 1 Hz WXT measurements')

plt.subplot(5,1,2)
plt.plot(ds.time_1Hz, ds.WXT_relative_humidity)
plt.ylabel('[%]')
plt.legend(['Rel. Humidity'])

plt.subplot(5,1,3)
plt.plot(ds.time_15min, ds.wave_significant_height)
plt.ylabel('[m]')
plt.legend(['Sig. wave height'],loc='upper right')

plt.subplot(5,1,4)
plt.plot(ds.time_20Hz, ds.wind_speed)
plt.plot(ds.time_1Hz, ds.WXT_wind_speed)
plt.ylabel('[m/s]')
plt.legend(['Gill Wind speed','WXT wind speed'],loc='upper right')

plt.subplot(5,1,5)
plt.plot(ds.Workhorse_time, ds.Workhorse_altitude)
plt.ylabel('[m]')
plt.legend(['Workhorse altitude (from IMU/GPS)'],loc='upper right')

if savefig:
    plt.savefig(__figdir__+WG+'_raw_met' + '.' +plotfiletype,**savefig_args)


# %%
# Raw met plot from WG:
fig, axs = plt.subplots(4, 1, sharex=True)
fig.autofmt_xdate()
plt.subplot(4,1,1)
plt.plot(ds.time_1Hz, ds.WXT_atmospheric_pressure)
plt.ylabel('[mbar]')
plt.legend(['Atm. pressure'])
plt.title(WG+': raw measurements')

plt.subplot(4,1,2)
try:
    plt.plot(ds.time_1Hz, ds.SGR4_longwave_flux)
    plt.ylabel('W/m^2')
    plt.legend(['Longwave radiation'])
except:
    plt.plot(ds.time_1Hz, ds.WXT_relative_humidity)
    plt.ylabel('%')
    plt.legend(['Relative humidity'])
    

plt.subplot(4,1,3)
try: 
    plt.plot(ds.time_1Hz, ds.SMP21_shortwave_flux)
    plt.ylabel('W/m^2')
    plt.legend(['Shortwave radiation'])
except:
    plt.plot(ds.time_15min, ds.wave_significant_height)
    plt.ylabel('m')
    plt.legend(['SWH'])

plt.subplot(4,1,4)
plt.plot(ds.time_1Hz, ds.WXT_rain_intensity)
plt.ylabel('[mm/hr]')
plt.legend(['Precip. rate'])

if savefig:
    plt.savefig(__figdir__+WG+'_raw_met2' + '.' +plotfiletype,**savefig_args)


# %%
# Raw met plot from WG:
fig, axs = plt.subplots(5, 1, sharex=True,figsize=(5,7))
fig.autofmt_xdate()
plt.subplot(5,1,1)
h1, = plt.plot(ds.time_1Hz, ds.WXT_air_temperature)
h2, = plt.plot(ds.time_1Hz, ds.UCTD_sea_water_temperature)
plt.legend([h1, h2],['Air temp.','SST'])
plt.ylabel('T [$^\circ$C]')
plt.title(WG + ' surface measurements')

try:
    plt.subplot(5,1,2)
    plt.plot(ds.time_1Hz, tt.run_avg1d(ds.SGR4_longwave_flux,15*60))
    plt.ylabel('W/m^2')
    plt.legend(['Longwave radiation'])
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.subplot(5,1,3)
    plt.plot(ds.time_1Hz, tt.run_avg1d(ds.SMP21_shortwave_flux,15*60))
    plt.ylabel('W/m^2')
    plt.legend(['Shortwave radiation'])
except:
    plt.subplot(5,1,3)
    plt.text(0.5, 0.5, 'No radiation data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.subplot(5,1,2)
    plt.text(0.5, 0.5, 'No radiation data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.subplot(5,1,4)
plt.plot(ds.time_1Hz, tt.run_avg1d(ds.WXT_wind_speed,15*60))
plt.ylabel('[m/s]')
plt.legend(['wind speed'],loc='upper right')

plt.subplot(5,1,5)
plt.plot(ds.time_15min, ds.wave_significant_height)
plt.ylabel('[m]')
plt.legend(['Sig. wave height'],loc='upper right')

if savefig:
    plt.savefig(__figdir__+WG+'_raw_met3' + '.' +plotfiletype,**savefig_args)


# %%
# There is a limited number of NaNs in WXT wind speed
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_1Hz, np.isnan(ds.WXT_wind_speed.values))
plt.title('NaNs in WXT wind speed, ' + WG)

# %%
# Examine nans
ff = np.where(np.isnan(ds.WXT_wind_speed.values)==0)
t = ds.time_1Hz[ff]
print(str(np.round(100*(1-np.size(ff)/np.size(ds.time_1Hz)),3))+'% of values are NaN')

# %%
gaps = np.where(np.diff(ds.time_1Hz)>np.timedelta64(60,'s'))

# This is the start of the 2nd gap
try:
    ds.time_1Hz[gaps[0][1]]
except:
    print('There appear to be no gaps')

# %%
# There is also a limited number of NaNs in Gill wind speed
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_20Hz, np.isnan(ds.wind_speed.values))
plt.title('NaNs in Gill wind speed, ' + WG)

# %%
# How many nans?
ff = np.where(np.isnan(ds.wind_speed.values)==0)
t = ds.time_20Hz[ff]
print(str(np.round(100*(1-np.size(ff)/np.size(ds.time_20Hz)),3))+'% of values are NaN')

# %%
gap_size = np.max(np.diff(ds.time_1Hz))
np.timedelta64(gap_size,'h')

# %%
# Make wind vector before smoothing, for both Gill Sonic anemometer and WXT
ds['WXT_wind_east'] = ds.WXT_wind_speed*np.cos(ds.WXT_wind_direction*np.pi/180)
ds['WXT_wind_north'] = ds.WXT_wind_speed*np.sin(ds.WXT_wind_direction*np.pi/180)
ds['wind_east']=ds.wind_speed*np.cos(ds.wind_direction*np.pi/180)
ds['wind_north']=ds.wind_speed*np.sin(ds.wind_direction*np.pi/180)


# %%
gapsize = np.timedelta64(60,'s')
gaps_1Hz = np.where(np.diff(ds.time_1Hz)>gapsize)
gaps_20Hz = np.where(np.diff(ds.time_20Hz)>gapsize)
gaps_WH = np.where(np.diff(ds.Workhorse_time)>gapsize)

print('1 Hz gaps exceeding '+str(gapsize) + ' start at:')
for n in range(np.size(gaps_1Hz[0][:])):
    print(ds.time_1Hz[gaps_1Hz[0][n]].values)

print('There are ' + str(np.size(gaps_1Hz[0][:])) + ' 1 Hz gaps exceeding '+str(gapsize))
print('There are ' + str(np.size(gaps_20Hz[0][:])) + ' 20 Hz gaps exceeding '+str(gapsize))
print('There are ' + str(np.size(gaps_WH[0][:])) + ' Workhorse gaps exceeding '+str(gapsize))

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
def interp_nans(var_list, ds_in, gaps):

    for var in var_list:
        var_raw = ds_in.data_vars.get(var).copy()
        if naninterp=='True': # Interpolate NaNs
            ff = np.flatnonzero(np.isnan(var_raw)==0)
            t = ds_in[ds_in.data_vars.get(var).coords.dims[0]]
            t_noninterp = t[ff]
            var_noninterp = var_raw[ff]
            var_value = np.interp(t, t_noninterp, var_noninterp)
        else: #just provide raw data as input to smoothing
            var_value = var_raw

        numnans = np.size(np.flatnonzero(np.isnan(var_raw)))
        var_value[gaps[0][:]]= np.nan # insert 1 nan to make gaps evident in plots
        locals()[var] = var_raw.rename(var) #locals()['string'] makes a variable with the name string
        print(var+' created, '+str(np.round(100*(1-numnans)/np.size(t),3))+'% of values are NaN'+', number of nans=' + str(numnans))
        locals()[var].values = var_value
        ds_new[var] = locals()[var]
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
ds_new = xr.Dataset().assign_attrs(ds.attrs)  # make empty xr.Dataset but copy attributes from original file

# %%
var_list = make_var_list(ds, 'time_1Hz')
ds_new = interp_nans(var_list, ds, gaps_1Hz)

# %%
# verify nans removed from fields as intended:
var = 'WXT_wind_speed'
testvar = ds_new.data_vars.get(var).copy()
ff = np.flatnonzero(np.isnan(testvar))
t = ds_new[ds_new.data_vars.get(var).coords.dims[0]]
#print(var+' should have 0 nans, '+str(np.round(100*(1-np.size(ff)/np.size(t)),3))+'% of values are NaN'+', max ff=' + str(np.size(ff)))
print(var+' should only have nans at large gaps; there are '+str(np.size(ff)) + ' nans')

# %%
# There is a limited number of NaNs in WXT wind speed
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_1Hz, np.isnan(ds_new.WXT_wind_speed.values))
plt.title(WG + ': NaNs in WXT wind speed, after removal of NaNs')

# %%
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds_new.time_1Hz,ds_new.WXT_wind_east)
plt.plot(ds_new.time_1Hz,ds_new.WXT_wind_north)
plt.plot(ds_new.time_1Hz,ds_new.WXT_wind_speed)
plt.title('WXT wind, interpolated, '+WG)
plt.ylabel('m/s')
#plt.ylim(-2,2)
plt.legend(['U','V','speed'])

# %%
# Now find 20 Hz variables (IMU and Gill sonic) and then interpolate nans
var_list = make_var_list(ds, 'time_20Hz')
ds_new = interp_nans(var_list, ds, gaps_20Hz)

# %%
# Now find Workhorse variables and then interpolate nans
var_list = make_var_list(ds, 'Workhorse_time')
ds_new = add_vars(var_list, ds, ds_new)

# %%
# verify nans removed from fields as intended:
var = 'WXT_wind_speed'
testvar = ds_new.data_vars.get(var).copy()
ff = np.flatnonzero(np.isnan(testvar))
t = ds_new[ds_new.data_vars.get(var).coords.dims[0]]
#print(var+' should have 0 nans, '+str(np.round(100*(1-np.size(ff)/np.size(t)),3))+'% of values are NaN'+', max ff=' + str(np.size(ff)))
print(var+' should only have nans at large gaps, there are '+str(np.size(ff)) + ' nans')

# %%
# Put wave spectral variables into the new dataset without interpolating
var_list1 = make_var_list(ds, 'time_15min')
var_list2 = make_var_list(ds, 'wave_frequency')
ds_new = add_vars(var_list1+var_list2, ds, ds_new)

#%%
# Finally, add variables that don't have a time coord
var_list = make_var_list(ds, None)
ds_new = add_vars(var_list, ds, ds_new)

# %%
# Write data to netcdf file
new_file = path_out+WG+'_L3a.nc'
print ('saving to ', new_file)
ds_new.to_netcdf(path=new_file)
ds_new.close()
print ('finished saving')

# %%
# Make a function that will interpolate the time dimension to a uniform time base
def interp_time(ds_in, time_coord, time_base, var_list):
    '''
    Interpolate all variables in var_list to a uniform time base
    '''
    ds_out = ds_in[var_list].copy()
    for var in var_list:
        var_raw = ds_in.data_vars.get(var).copy()
        var_value = np.interp(time_base, time_coord, var_raw)
        locals()[var] = var_raw.rename(var) #locals()['string'] makes a variable with the name string
        locals()[var].values = var_value
        ds_out[var] = locals()[var]
    return ds_out

# %%
# Make a uniform 1 Hz time base

np.diff(ds.time_1Hz).min()

# %%
