    # %% [markdown]
    # # Wave Glider Level 3a processing
    # ## interp small gaps, average to 1 min, save files
    # 

    # %% [markdown]
    # ## About Level 3 data
    # 
    # The new thing with this processing file (compared to the ones by similar names with L3a and L3b)
    # is that I am using xr.resample to average the data to a uniform time base.
    # 
    # Level 3a data:
    # - All data (except wave data) are averaged to a uniform 1-minute time base
    # 



# %%

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib
import datetime as dt
import os

# %%
def WG_L3_processor_function(input_tuple):
    """
    This function takes the raw L2 Wave Glider data and outputs a netcdf file with
    the data averaged to a uniform time base. The time base is 1 minute 
    for all variables except wave data, which is 15 minutes. The function also adds
    wind vectors and associated metadata. 


    Inputs
    ------
    input_tuple : tuple
        (campaign, WG, savefig, figdir)
    campaign : str
        'PFC' or 'IOP1'
    WG : str
        'WHOI43' or 'Kelvin'
    savefig : bool
        True to save figures, False to not save figures
    figdir : str
        path to save figures
    
    Outputs
    -------
    None (but saves a netcdf file and figures)

    """
    # Unpack input_tuple
    campaign, WG, savefig , figdir = input_tuple
    print('Processing ' + WG + ' ' + campaign + ' data \n -------------------------------------')

    plt.rcParams['figure.figsize'] = (10,7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 400
    plt.close('all')
    plt.rcParams['axes.xmargin'] = 0


    savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
    plotfiletype='png'
    # make figdir if it doesn't exist
    if not os.path.exists(figdir):
        os.makedirs(figdir)



    path = '/mnt/d/tom_data/S-MODE/S-MODE_data/final/' + campaign + '/Wavegliders/'
    path_out = '../data/processed/Waveglider_L3/' + campaign + '/' 
    #path_out = '/mnt/d/tom_data/S-MODE/S-MODE_data/final/' + campaign + '/Wavegliders/L3a/'
    # make directory if it doesn't exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    file = 'SMODE_' + campaign + '_Wavegliders_'+WG+'.nc'
    file_out = 'SMODE_' + campaign + '_Wavegliders_'+WG+'_L3.nc'

    # check if file exists
    if not os.path.exists(path+file):
        print('File ' + path + file + ' does not exist-- skipping')
        return

    # %% Some functions
    # Some functions
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
            except Exception: # if there is no time coord, just skip it
                not_used.append(var)

        if time_coord is None:
            var_list = not_used

        return var_list


    # Make a function that will use the xr.resample method to resample a single variable
    # This function will determine the time base from the input variable
    # I am writing it to work on a single variable so that it can be parallelized in a loop using tqdm
    def resample_var_1Hz(var, ds_in):
        '''
        Resample a 1Hz variable to a uniform time base
        '''
        var_raw = ds_in.data_vars.get(var).copy()
        var_resampled = var_raw.resample(time_1Hz = '1 min',skipna = True).mean()
        return var_resampled

    def resample_var_20Hz(var, ds_in):
        '''
        Resample a 20Hz variable to a uniform time base
        '''
        var_raw = ds_in.data_vars.get(var).copy()
        var_resampled = var_raw.resample(time_20Hz = '1 min',skipna = True).mean()
        return var_resampled

    def resample_var_WH(var, ds_in):
        '''
        Resample a Workhorse variable to a uniform time base
        '''
        var_raw = ds_in.data_vars.get(var).copy()
        # remove any values corresponding to duplicate times
        # This is a kludge to deal with the fact that the Workhorse time base is not monotonic
        # find duplicate times
        _, index = np.unique(var_raw.Workhorse_time, return_index=True)
        # remove duplicate times
        var_raw = var_raw.isel(Workhorse_time=index)
        # resample
        var_resampled = var_raw.resample(Workhorse_time = '1 min',skipna = True).mean()

        return var_resampled

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
    # With the preliminary data, it was the case that this needed to be set before 
    # any dates are encoded as datetime64-- not sure if it is still needed
    # Because the time increment is so small, we get 'out of range' when starting from the standard datetime64 epoch of 1970
    # matplotlib.dates.set_epoch('2000-01-01T00:00:00') 

    # %%
    ds = xr.open_dataset(path+file, engine = 'netcdf4', decode_times = True) #decode_times = False, 


    # %%
    # The 20 Hz variables from WHOI43 for the pilot campaign has a 20 year time offset
    # This is a kludge to fix it
    if WG == 'WHOI43':
        # note that ds.time_20Hz is datetime64[ns]
        new_time = ds.time_20Hz.values - np.timedelta64(20,'Y').astype('timedelta64[ns]')
        # replace the time variable with the new one
        ds = ds.assign_coords(time_20Hz = new_time)


    # %%
    # Make wind vectors from WXT wind speed and direction before smoothing
    ds['WXT_wind_east'] = ds.WXT_wind_speed*np.cos(ds.WXT_wind_direction*np.pi/180)
    ds['WXT_wind_north'] = ds.WXT_wind_speed*np.sin(ds.WXT_wind_direction*np.pi/180)

    # add metadata for new WXT variables
    ds['WXT_wind_east'].attrs = ds.WXT_wind_speed.attrs
    ds['WXT_wind_east'].attrs['long_name'] = 'WXT wind zonal component (positive to the east)'
    ds['WXT_wind_east'].attrs['standard_name'] = 'eastward_wind'
    ds['WXT_wind_east'].attrs['units'] = 'm s-1'
    ds['WXT_wind_east'].attrs['comment'] = 'WXT wind speed and direction are measured at 1 Hz.  This variable is the eastward component of the wind vector, calculated from the WXT wind speed and direction.'
    ds['WXT_wind_north'].attrs['instrument'] = 'INST_WXT'
    ds['WXT_wind_north'].attrs['valid_min'] = -50
    ds['WXT_wind_north'].attrs['valid_max'] = 50

    ds['WXT_wind_north'].attrs = ds.WXT_wind_speed.attrs
    ds['WXT_wind_north'].attrs['long_name'] = 'WXT wind meridional component (positive to the north))'
    ds['WXT_wind_north'].attrs['standard_name'] = 'northward_wind'
    ds['WXT_wind_north'].attrs['units'] = 'm s-1'
    ds['WXT_wind_north'].attrs['comment'] = 'WXT wind speed and direction are measured at 1 Hz.  This variable is the northward component of the wind vector, calculated from the WXT wind speed and direction.'
    ds['WXT_wind_north'].attrs['instrument'] = 'INST_WXT'
    ds['WXT_wind_north'].attrs['valid_min'] = -50
    ds['WXT_wind_north'].attrs['valid_max'] = 50

    # Now do the same for Gill R3 wind speed and direction
    # First check if there is a Gill wind speed variable
    flag_20Hz = False
    if 'wind_speed' in ds.data_vars:
        flag_20Hz = True
        ds['wind_east']=ds.wind_speed*np.cos(ds.wind_direction*np.pi/180)
        ds['wind_north']=ds.wind_speed*np.sin(ds.wind_direction*np.pi/180)

        ds['wind_east'].attrs = ds.wind_speed.attrs
        ds['wind_east'].attrs['long_name'] = 'Gill wind east component (positive to the east)'
        ds['wind_east'].attrs['standard_name'] = 'eastward_wind'
        ds['wind_east'].attrs['units'] = 'm s-1'
        ds['wind_east'].attrs['comment'] = 'Gill wind speed and direction are measured at 20 Hz.  This variable is the eastward component of the wind vector, calculated from the Gill wind speed and direction.'
        ds['wind_east'].attrs['instrument'] = 'INST_GILL'
        ds['wind_east'].attrs['valid_min'] = -50
        ds['wind_east'].attrs['valid_max'] = 50

        ds['wind_north'].attrs = ds.wind_speed.attrs
        ds['wind_north'].attrs['long_name'] = 'Gill wind north component (positive to the north)'
        ds['wind_north'].attrs['standard_name'] = 'northward_wind'
        ds['wind_north'].attrs['units'] = 'm s-1'
        ds['wind_north'].attrs['comment'] = 'Gill wind speed and direction are measured at 20 Hz.  This variable is the northward component of the wind vector, calculated from the Gill wind speed and direction.'
        ds['wind_north'].attrs['instrument'] = 'INST_GILL'
        ds['wind_north'].attrs['valid_min'] = -50
        ds['wind_north'].attrs['valid_max'] = 50


    # %%
    # # This would work fine, but it requires a 291 GB array
    # ds_1min = ds.resample(time_1Hz = '1 min',skipna = True).mean()
    # Alternate plan: make a list of variables that have a time coord, 
    # and then resample each one individually
    ds_new = xr.Dataset().assign_attrs(ds.attrs)  # make empty xr.Dataset but copy attributes from original file
    var_list = make_var_list(ds, 'time_1Hz')
    #ds_new = interp_nans(var_list, ds, gaps_1Hz)

    # loop through variables in var_list and resample each one
    # Also, keep the variable name the same, but remove the suffix '_1Hz'
    for var in var_list:
        print(var)
        var_resampled = resample_var_1Hz(var, ds)
        locals()[var] = var_resampled.rename(var) #locals()['string'] makes a variable with the name string
        ds_new[var] = locals()[var]
        ds_new[var].attrs = ds[var].attrs # copy attributes from original variable

    # %%
    # Do the same for time_20Hz variables
    var_list = make_var_list(ds, 'time_20Hz')
    for var in var_list:
        print(var)
        var_resampled = resample_var_20Hz(var, ds)
        # interpolate variable to time_1Hz (which is actually a 1-min time base)
        var_resampled = var_resampled.interp(time_20Hz = ds_new.time_1Hz)
        locals()[var] = var_resampled.rename(var) #locals()['string'] makes a variable with the name string
        ds_new[var] = locals()[var]
        ds_new[var].attrs = ds[var].attrs # copy attributes from original variable


    # %%
    # Do the same for Workhorse variables
    var_list = make_var_list(ds, 'Workhorse_time')
    for var in var_list:
        print(var)
        var_resampled = resample_var_WH(var, ds)
        # interpolate variable to time_1Hz (which is actually a 1-min time base)
        var_resampled = var_resampled.interp(Workhorse_time = ds_new.time_1Hz)
        locals()[var] = var_resampled.rename(var) #locals()['string'] makes a variable with the name string
        ds_new[var] = locals()[var]
        ds_new[var].attrs = ds[var].attrs # copy attributes from original variable

    # %%
    # Now find variables that don't have a time coord and then add them to the new dataset
    var_list = make_var_list(ds, None)
    ds_new = add_vars(var_list, ds, ds_new)

    # %%
    # Change the name of the associated coordinate to 'time'
    ds_new = ds_new.rename({'time_1Hz':'time'})
    # Now drop the time_20Hz and Workhorse_time coordinates (dimensions do not need to be dropped)
    if flag_20Hz:
        ds_new = ds_new.drop_vars(['time_20Hz','Workhorse_time'])
    else:
        ds_new = ds_new.drop_vars(['Workhorse_time'])

    # %%
    # Put wave spectral variables into the new dataset without interpolating
    var_list1 = make_var_list(ds, 'time_15min')
    var_list2 = make_var_list(ds, 'wave_frequency')
    ds_new = add_vars(var_list1+var_list2, ds, ds_new)

    # %% 
    # Make plots to compare the original and resampled data
    # Plot the original and resampled data

    # Raw met plot from WG:
    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.autofmt_xdate()
    plt.subplot(5,1,1)
    h1, = plt.plot(ds.time_1Hz, ds.WXT_air_temperature)
    h2, = plt.plot(ds.time_1Hz, ds.UCTD_sea_water_temperature)
    # add resampled data
    h3, = plt.plot(ds_new.time, ds_new.WXT_air_temperature)
    h4, = plt.plot(ds_new.time, ds_new.UCTD_sea_water_temperature)
    plt.legend([h1, h2, h3, h4],['Air temp.','SST','Air temp. bin avg.','SST bin avg.'],loc='upper right')
    plt.ylabel('T [$^\circ$C]')
    plt.title(WG+': raw 1 Hz WXT measurements')

    plt.subplot(5,1,2)
    plt.plot(ds.time_1Hz, ds.WXT_relative_humidity)
    plt.plot(ds_new.time, ds_new.WXT_relative_humidity)
    plt.ylabel('[%]')
    plt.legend(['Rel. Humidity','Rel. Humidity bin avg.'],loc='upper right')

    plt.subplot(5,1,3)
    plt.plot(ds.time_15min, ds.wave_significant_height)
    plt.plot(ds_new.time_15min, ds_new.wave_significant_height)
    plt.ylabel('[m]')
    plt.legend(['Sig. wave height','Sig. wave height bin avg.'],loc='upper right')

    plt.subplot(5,1,4)
    if flag_20Hz:
        plt.plot(ds.time_20Hz, ds.wind_speed)
        plt.plot(ds_new.time, ds_new.wind_speed)
        plt.plot(ds.time_1Hz, ds.WXT_wind_speed)
        plt.plot(ds_new.time, ds_new.WXT_wind_speed)
        plt.ylabel('[m/s]')
        plt.legend(['Gill Wind speed','Gill Wind speed bin avg.','WXT wind speed','WXT wind speed bin avg.'],loc='upper right')
    else:
        plt.plot(ds.time_1Hz, ds.WXT_wind_speed)
        plt.plot(ds_new.time, ds_new.WXT_wind_speed)
        plt.ylabel('[m/s]')
        plt.legend(['WXT wind speed','WXT wind speed bin avg.'],loc='upper right')

    plt.subplot(5,1,5)
    plt.plot(ds.Workhorse_time, ds.Workhorse_vel_east[:,10])
    plt.plot(ds_new.time, ds_new.Workhorse_vel_east[:,10])
    plt.ylabel('[m/s]')
    plt.legend(['WH east vel.','WH east vel. bin avg.'],loc='upper right')


    if savefig:
        plt.savefig(figdir+WG+'_'+ campaign +'_raw_met' + '.' +plotfiletype,**savefig_args)

    # %%
    # Raw met plot from WG:
    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.autofmt_xdate()
    plt.subplot(4,1,1)
    plt.plot(ds.time_1Hz, ds.WXT_atmospheric_pressure)
    plt.plot(ds_new.time, ds_new.WXT_atmospheric_pressure)
    plt.ylabel('[mbar]')
    plt.legend(['Atm. pressure','Atm. pressure bin avg.'])
    plt.title(WG+': raw measurements')

    plt.subplot(4,1,2)
    try:
        plt.plot(ds.time_1Hz, ds.SGR4_longwave_flux)
        plt.plot(ds_new.time, ds_new.SGR4_longwave_flux)
        plt.ylabel('W/m^2')
        plt.legend(['Longwave radiation','Longwave radiation bin avg.'],loc='upper right')
    except Exception:
        plt.plot(ds.time_1Hz, ds.WXT_relative_humidity)
        plt.plot(ds_new.time, ds_new.WXT_relative_humidity)
        plt.ylabel('%')
        plt.legend(['Relative humidity','Relative humidity bin avg.'],loc='upper right')

    plt.subplot(4,1,3)
    try:
        plt.plot(ds.time_1Hz, ds.SMP21_shortwave_flux)
        plt.plot(ds_new.time, ds_new.SMP21_shortwave_flux)
        plt.ylabel('W/m^2')
        plt.legend(['Shortwave radiation','Shortwave radiation bin avg.'],loc='upper right')
    except Exception:
        plt.plot(ds.time_15min, ds.wave_significant_height)
        plt.plot(ds_new.time_15min, ds_new.wave_significant_height)
        plt.ylabel('m')
        plt.legend(['SWH','SWH bin avg.'],loc='upper right')

    plt.subplot(4,1,4)
    plt.plot(ds.time_1Hz, ds.WXT_rain_intensity)
    plt.plot(ds_new.time, ds_new.WXT_rain_intensity)
    plt.ylabel('[mm/hr]')
    plt.legend(['Precip. rate','Precip. rate bin avg.'],loc='upper right')

    if savefig:
        plt.savefig(figdir+WG+'_'+ campaign +'_raw_met2' + '.' +plotfiletype,**savefig_args)

    # %%
    # Examine nans in WXT wind speed
    ff = np.where(np.isnan(ds.WXT_wind_speed.values)==0)
    t = ds.time_1Hz[ff]
    WXT_nan_str = str(np.round(100*(1-np.size(ff)/np.size(ds.time_1Hz)),3))+'% of WXT original values are NaN'
    print(WXT_nan_str)

    # %%
    # There is a limited number of NaNs in WXT wind speed
    fig, axs = plt.subplots(1, 1)
    fig.autofmt_xdate()
    plt.plot(ds.time_1Hz, np.isnan(ds.WXT_wind_speed.values))
    plt.plot(ds_new.time, np.isnan(ds_new.WXT_wind_speed.values))
    plt.legend(['WXT wind speed','WXT wind speed bin avg.'],loc='upper right')
    plt.title('NaNs in WXT wind speed, ' + WG + '\n' + WXT_nan_str)

    if savefig:
        plt.savefig(figdir+WG+'_'+ campaign +'_WXT_wspd_NaNs' + '.' +plotfiletype,**savefig_args)

    # %%
    # NaNs in Gill wind speed
    if flag_20Hz:
        ff = np.where(np.isnan(ds.wind_speed.values)==0)
        t = ds.time_20Hz[ff]
        Gill_nan_str = str(np.round(100*(1-np.size(ff)/np.size(ds.time_20Hz)),3))+'% of values are NaN'
        print(Gill_nan_str)


    # %%
    # There is also a limited number of NaNs in Gill wind speed
    if flag_20Hz:
        fig, axs = plt.subplots(1, 1)
        fig.autofmt_xdate()
        plt.plot(ds.time_20Hz, np.isnan(ds.wind_speed.values))
        plt.plot(ds_new.time, np.isnan(ds_new.wind_speed.values))
        plt.legend(['Gill wind speed','Gill wind speed bin avg.'],loc='upper right')
        plt.title('NaNs in Gill wind speed, ' + WG + '\n' + Gill_nan_str + '\n (includes large blocks of NaNs when instrument is off)')

        if savefig:
            plt.savefig(figdir+WG+'_'+ campaign +'_Gill_wspd_NaNs' + '.' +plotfiletype,**savefig_args)




    # %%
    # Write data to netcdf file
    new_file = path_out + file_out
    print ('saving to ', new_file)
    ds_new.to_netcdf(path=new_file)
    ds_new.close()
    print ('finished saving')


    # %%
