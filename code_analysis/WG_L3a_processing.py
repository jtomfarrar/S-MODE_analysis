# %% [markdown]
# # Wave Glider Level 3a processing
# ## interp small gaps, generate subsets, save files
# 

# %% [markdown]
# ## About Level 3a and Level 3b data
# 
# Level 3a data: (L3a_all)
# - The only difference from Level 2 is that NaNs are removed by interpolation
# - This is a "wave resolving" data set with modest QC (removal/interpolation of gaps<1 sec)
# 
# Level 3a subsets: (just interpolate gaps<1 sec and extract individual data sets)
# - Make a velocity subset (L3a_vel)
# - make a wave subset (L3a_wave)
# - make a met subset (L3a_met)
# 
# Level 3b met data:
# - Same as Level 3a but variables are averaged to 1 min
# - 20 Hz variables are subsampled to 1 Hz time base
# - Maybe in the future, will want to subsample all variables to a 1 min time base
# - May not produce for all variables (eg, vechiles with met packages off)
# 
# Level 3b velocity data:
# - Velocity data and averaged to 5 minutes
# 
# ## This notebook is for level 3a data
# 1. Do L3a processing (interp small gaps, generate subsets, save files)
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

WG = 'Kelvin'#'Stokes'#'WHOI43'#
file = 'SMODE_PFC_Wavegliders_'+WG+'.nc'

# %%
naninterp='True' # Interpolate NaNs

# %%
# This needs to be set before any dates are encoded as datetime64
# Because the time increment is so small, we get 'out of range' when starting from the standard datetime64 epoch of 1970
matplotlib.dates.set_epoch('2000-01-01T00:00:00') 

# %%
# %%time
ds = xr.open_dataset(path+file, engine = 'netcdf4', decode_times = True) #decode_times = False, 

# %%
#ds

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
plt.plot(ds.time_1Hz, ds.WXT_rainfall_rate)
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

# I guess what I need to do is interpolate and then go back and set gaps to nans

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
        except:
            not_used.append(var)

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
ds_WH = xr.Dataset().assign_attrs(ds.attrs)  # make empty xr.Dataset but copy attributes from original file


# %%
var_list = make_var_list(ds, 'time_1Hz')

# %%
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
ff

# %%
# There is a limited number of NaNs in WXT wind speed
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_1Hz, np.isnan(ds_new.WXT_wind_speed.values))
plt.title(WG + ': NaNs in WXT wind speed, after removal of NaNs')

# %%


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
# Now find 20 Hz variables (IMU and Gill sonic)
var_list = make_var_list(ds, 'time_20Hz')

# %%
ds_new = interp_nans(var_list, ds, gaps_20Hz)

# %%
var_list = make_var_list(ds, 'Workhorse_time')

# %%
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
var_list1 = make_var_list(ds, 'time_15min')
var_list2 = make_var_list(ds, 'wave_frequency')
ds_new = add_vars(var_list1+var_list2, ds, ds_new)

# %%
ds_waves = subset(var_list1+var_list2, ds_new)

# %%
var_list = make_var_list(ds_new, 'time_1Hz')
ds_1Hz = subset(var_list, ds_new)

# %%
var_list = make_var_list(ds_new, 'time_20Hz')
ds_20Hz = subset(var_list, ds_new)

# %%
var_list = make_var_list(ds_new, 'Workhorse_time')
ds_WH = subset(var_list, ds_new)

# %%
ds_WH

# %%
ds_20Hz

# %%
ds_new

# %%
# Write data to netcdf file
new_file = path_out+WG+'_L3a.nc'
print ('saving to ', new_file)
ds_new.to_netcdf(path=new_file)
ds_new.close()
print ('finished saving')

# Write 1 Hz data to netcdf file
new_file = path_out+WG+'_L3a_1Hz_met.nc'
print ('saving to ', new_file)
ds_1Hz.to_netcdf(path=new_file)
ds_1Hz.close()
print ('finished saving')

# Write 20 Hz data to netcdf file
new_file = path_out+WG+'_L3a_20Hz_met.nc'
print ('saving to ', new_file)
ds_20Hz.to_netcdf(path=new_file)
ds_20Hz.close()
print ('finished saving')

# Write Workhorse ADCP data to netcdf file
new_file = path_out+WG+'_L3a_ADCP.nc'
print ('saving to ', new_file)
ds_WH.to_netcdf(path=new_file)
ds_WH.close()
print ('finished saving')

# Write wave data to netcdf file
new_file = path_out+WG+'_L3a_waves.nc'
print ('saving to ', new_file)
ds_waves.to_netcdf(path=new_file)
ds_waves.close()
print ('finished saving')

# %%
skfhdskahfka

# %%
# compare different time vectors to decide how to line up variables from different time bases
# ds.time_20Hz
# ds.time_1Hz
# ds.Workhorse_time

# Try the simple thing
# xy, x_ind, y_ind = np.intersect1d(ds.time_1Hz, ds.time_20Hz, assume_unique=True, return_indices=True)
# That doesn't work (no surprise) because the times don't line up exactly (possibly numerical precision issue)

# I think they are already nearly lined up
ds.time_1Hz[0:10:1]-ds.time_20Hz[0:200:20]
# so, try 0,20,40... in 20 Hz array

tfoo = ds.time_20Hz[0:-1:20]
#tfoo and time_1Hz are very close, within 10^-4 sec



# %%


# %%
# ds.time_20Hz variables
# wdir is positive clockwise from North
# need to make U, V
Gill_U=ds.wind_speed*np.cos(ds.wind_direction*np.pi/180)
Gill_V=ds.wind_speed*np.sin(ds.wind_direction*np.pi/180)
Gill_wspd_low = tt.run_avg1d(ds.wind_speed, nsec*20)
Gill_U_low = tt.run_avg1d(Gill_U,nsec*20)
Gill_V_low = tt.run_avg1d(Gill_V,nsec*20)
pitch_low = tt.run_avg1d(ds.pitch,nsec*20)
roll_low = tt.run_avg1d(ds['roll'],nsec*20)

# %%
# ds.time_1Hz variables
WXT_U=ds.WXT_wind_speed*np.cos(ds.WXT_wind_direction/np.pi/2)
WXT_V=ds.WXT_wind_speed*np.sin(ds.WXT_wind_direction/np.pi/2)
WXT_U_low = tt.run_avg1d(WXT_U,nsec)
WXT_V_low = tt.run_avg1d(WXT_V,nsec)
WXT_wspd_low = tt.run_avg1d(ds.WXT_wind_speed,nsec)
WXT_atmp_low = tt.run_avg1d(ds.WXT_air_temperature,nsec)
WXT_rh_low = tt.run_avg1d(ds.WXT_relative_humidity,nsec)
swr_low = tt.run_avg1d(ds.SMP21_shortwave_flux,nsec)
lwr_low = tt.run_avg1d(ds.SGR4_longwave_flux,nsec)
lat_low = tt.run_avg1d(ds.latitude_1Hz,nsec)
lon_low = tt.run_avg1d(ds.longitude_1Hz,nsec)

# %%
# Workhorse_time variables
U_low=tt.run_avg2d(ds.Workhorse_vel_east,nsec,1)
V_low=tt.run_avg2d(ds.Workhorse_vel_north,nsec,1)
W_low=tt.run_avg2d(ds.Workhorse_vel_up,nsec,1)

# %%
# Subsample smoothed 20 Hz vars to 1 Hz
Gill_wspd_low = Gill_wspd_low[0:-1:20]
Gill_U_low = Gill_U_low[0:-1:20]
Gill_V_low = Gill_V_low[0:-1:20]
pitch_low = pitch_low[0:-1:20]
roll_low = roll_low[0:-1:20]


# %%
fig, axs = plt.subplots(1, 1)
fig.autofmt_xdate()

plt.plot(ds.time_1Hz[0:-1:],pitch_low)
plt.plot(ds.time_1Hz[0:-1:],roll_low)
plt.title('1-hr average pitch and roll')
plt.ylabel('Degrees')
plt.ylim(-2,2)

if savefig:
    plt.savefig(__figdir__+WG+'_mean_pitch_roll' + '.' +plotfiletype,**savefig_args)


# %%
plt.ylim(-2,2)

# %%
new_file = path+WG+'_L2b_met.nc'
print ('saving to ', new_file)
ds_new.to_netcdf(path=new_file)
ds_new.close()
print ('finished saving')

# %%
WG

# %%
# this is broken for now
fig, axs = plt.subplots(1, 1)
M=151
tt.spectrum_band_avg(wspd_interp,1/20,M,winstr=None,plotflag=True,ebarflag=None)
tt.spectrum_band_avg(WXT_wspd_interp,1,M,winstr=None,plotflag=True,ebarflag=False)
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



