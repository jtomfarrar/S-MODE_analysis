# -*- coding: utf-8 -*-
"""
Some function definitions for S-MODE analysis tools

Created on Sun Oct 24 10:37:33 2021

@author: jtomfarrar
"""

def plot_ops_area(ax,**kwargs):
   """ Add polygon to show S-MODE IOP1 operations area.
         
   Inputs
   - matplotlib.pyplot.plot kwargs

   Return
   - exit code (True if OK)
   """
    # Add S-MODE IOP operations area
   '''
    New corners of polygon:
    35.790897° -125.538656°
    38.182585° -126.244555°
    38.113294° -125.438469°
    37.711709° -123.994657°
    37.795817° -123.383396°
    36.998690° -122.922778°
    '''
    
   coord = [[-125.538656,35.790897], [-126.244555,38.182585], [-125.438469,38.113294], [-123.994657,37.711709], [-123.383396,37.795817], [-122.922778, 36.998690]]
   coord.append(coord[0]) #repeat the first point to create a 'closed loop'

   xs, ys = zip(*coord) #create lists of x and y values

   if ax is None:
       ax = plt.gca()    
   # ax.plot(xs,ys,transform=ccrs.PlateCarree()) 
   ax.plot(xs,ys,**kwargs) 

   SF_lon=-(122+25/60)
   SF_lat= 37+47/60

   # mark a known place to help us geo-locate ourselves
   ax.plot(SF_lon, SF_lat, 'o', markersize=3, zorder=10, **kwargs)
   ax.text(SF_lon-5/60, SF_lat+5/60, 'San Francisco', fontsize=8, zorder=10, **kwargs)
   # ax.text(np.mean(xs)-.6, np.mean(ys)-.3, 'S-MODE ops area', fontsize=8, **kwargs)
   print(kwargs)

   return(xs,ys,ax)



def plot_ops_area_old(ax,**kwargs):
   """ Add polygon to show S-MODE pilot operations area.
         
   Inputs
   - matplotlib.pyplot.plot kwargs

   Return
   - exit code (True if OK)
   """
    # Add S-MODE pilot operations area
   '''
    New corners of pentagon:
    38° 05.500’ N, 125° 22.067’ W
    37° 43.000’ N, 124° 00.067’ W
    37° 45.000’ N, 123° 26.000‘ W
    36° 58.000’ N, 122° 57.000’ W
    36° 20.000’ N, 124° 19.067’ W 
    '''
    
   coord = [[-(125+22.067/60),38+5.5/60], [-(124+0.067/60),37+43/60], [-(123+26/60),37+45/60], [-(122+57/60),36+58/60], [-(124+19.067/60),36+20/60]]
   coord.append(coord[0]) #repeat the first point to create a 'closed loop'

   xs, ys = zip(*coord) #create lists of x and y values

   if ax is None:
       ax = plt.gca()    
   # ax.plot(xs,ys,transform=ccrs.PlateCarree()) 
   ax.plot(xs,ys,**kwargs) 

   SF_lon=-(122+25/60)
   SF_lat= 37+47/60

   # mark a known place to help us geo-locate ourselves
   ax.plot(SF_lon, SF_lat, 'o', markersize=3, zorder=10, **kwargs)
   ax.text(SF_lon-5/60, SF_lat+5/60, 'San Francisco', fontsize=8, zorder=10, **kwargs)
   # ax.text(np.mean(xs)-.6, np.mean(ys)-.3, 'S-MODE ops area', fontsize=8, **kwargs)
   print(kwargs)

   return(xs,ys,ax)

def get_current_position(platform_str,**kwargs):
    """
    Plot current position of particular S-MODE assets (using JSON file from http://smode.whoi.edu/status.php?format=json).
    
    Parameters
    ----------
    platform_str : str
        Name of platform: can be 'saildrone', 'navo_glider', 'drifter', 'ship' or 'waveglider'.

    Returns
    -------
    A Pandas DataFrame with positions and other info.  Plot with ax.plot(platform['longitude'],platform['latitude'],**kwargs)

    """
    import matplotlib.pyplot as plt
    import pandas as pd
    df1=pd.read_json('http://smode.whoi.edu/status.php?format=json')
    platform=df1[df1['type']==platform_str]
    
    
    return(platform)

def sst_map_SMODE(url,zoom,V,time_window):
    """
    Plot map of SST for S-MODE region.

    Parameters
    ----------
    url : URL for netcdf file with SST image (like from smode.whoi.edu)
    zoom : Int, 0 to 4
        zoom level for plots, larger value gives closer view, 
            zoom = 0: #wide view of S-MODE ops area and San Francisco
            zoom = 1: #centered on S-MODE ops area, but shows San Francisco
            zoom = 2: #tight on S-MODE ops area
            zoom = 3: #zoom on eastern part of ops area
    V : list with 2 elements (e.g., [12, 16])
        max and min of color range
    time_window : float
        number of hours to average HF radar data around time of SST image
    
    Returns
    -------
    None.

    """
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patheffects import Stroke
    import copy
    import cartopy.crs as ccrs                   # import projections
    import cartopy
    import datetime
    import shapely.geometry

    
    if zoom == 0: #wide view of S-MODE ops area and San Francisco
        xmin, xmax = (-126,-121)
        ymin, ymax = (36, 39)
        zoom_str='_wide'
    elif zoom == 1: #centered on S-MODE ops area, but shows San Francisco
        xmin, xmax = (-126.0,-122.0)
        ymin, ymax = (36.0, 39.0)
        zoom_str='_zoom1'
    elif zoom == 2: #tight on S-MODE ops area
        xmin, xmax = (-125.5,-123)
        ymin, ymax = (36.5,38)
        zoom_str='_zoom2'
    elif zoom == 3: #zoom on eastern part of ops area
        xmin, xmax = (-124.75,-122.5)
        ymin, ymax = (36.3,38)
        zoom_str='zoom3'
    elif zoom == 4: #not yet determined
        xmin, xmax = (-126,-122)
        ymin, ymax = (36, 39)
        zoom_str='_zoom4'
    
    #Set color scale range (define V with input file)
    levels = np.linspace(V[0],V[1],21)
    sst = xr.open_dataset(url)
    
    plt.figure()
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=200))  # Orthographic
    extent = [xmin, xmax, ymin, ymax]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    day_str=sst.time.dt.strftime("%Y-%m-%d %H:%M").values[0]
    day_str2=sst.time.dt.strftime("%Y-%m-%d").values[0]
    #plt.set_cmap(cmap=plt.get_cmap('nipy_spectral'))
    plt.set_cmap(cmap=plt.get_cmap('turbo'))
    cmap = copy.copy(matplotlib.cm.get_cmap("turbo"))
    cmap.set_bad(color='white')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=3, facecolor=[.6,.6,.6], edgecolor='black')
    arr = np.ma.array(sst.sea_surface_temperature, mask=(sst.quality_level == 1))
    cs = ax.pcolormesh(sst.lon,sst.lat,np.squeeze(arr)-273.15, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs,fraction = 0.022,extend='both')
    cb.set_label('SST [$\circ$C]',fontsize = 10)
    #plot_ops_area(ax,transform=ccrs.PlateCarree(),color='k')
    plot_IOP1_ops_area(ax,transform=ccrs.PlateCarree(),color='k')

    ################################################
    ## Now read in HF radar for appropriate time
    # This could easily be spun off into a seperate function
    center_timeDT = datetime.datetime.strptime(str(sst.time[0].values),'%Y-%m-%d %H:%M:%S') #time of SST image
    # Choose time interval
    time_delta=datetime.timedelta(hours=time_window/2)
    
    startTimeDT=center_timeDT-time_delta
    endTimeDT=center_timeDT+time_delta
    url = 'http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd'

    ds=xr.open_dataset(url,drop_variables=['hdop','dopx','dopy',]).sel(time=slice(startTimeDT,endTimeDT))
    nanmask = np.isnan(ds.u).sum('time')>time_window/4
    mask = np.where(nanmask.values,np.nan,0)
    
    # Try an experiment to plot velocity relative to a particular spot
    vel_offset = False
    if vel_offset:
        x00 = -124.18
        y00 = 37.3
        u00 = ds.u.sel(lon=x00, lat=y00, method='nearest').mean('time').values
        v00 = ds.v.sel(lon=x00, lat=y00, method='nearest').mean('time').values
    else:
        u00 = 0
        v00 = 0

    ax.quiver(ds.lon.values, ds.lat.values, mask+ds.u.mean('time').values-u00, ds.v.mean('time').values-v00,  scale=3, transform=ccrs.PlateCarree())
    x0 = -122.8 
    y0 = 37
    ax.quiver(np.array([x0]), np.array([y0]), np.array([0.25/np.sqrt(2)]), np.array([0.25/np.sqrt(2)]), color='w', scale=3, transform=ccrs.PlateCarree())
    ax.text(x0+3/60, y0+.15/60, '0.25 m/s', color = 'w', fontsize=6, transform=ccrs.PlateCarree())
    ax.set_title('SST, ' + day_str + ', surface currents averaged over +/- ' + f'{time_window/2:.1f}' + ' hr', size = 10.)
    
    # Create an inset GeoAxes showing the location of the Solomon Islands.
    sub_ax = plt.axes([0.65, 0.655, 0.2, 0.2], projection=ccrs.PlateCarree())
    sub_ax.set_extent([-130, -115, 29, 45])

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=2, foreground='wheat', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the Solomon Islands.
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()
    sub_ax.add_feature(cartopy.feature.STATES, zorder=3, linewidth=0.5)
    # extent_box = shapely.geometry.box(xmin, ymin, xmax, ymax)
    coord = [[xmin,ymin], [xmax,ymin], [xmax,ymax], [xmin,ymax]]
    coord.append(coord[0]) #repeat the first point to create a 'closed loop'
    xs, ys = zip(*coord) #create lists of x and y values
    sub_ax.plot(xs,ys, transform=ccrs.PlateCarree(), linewidth=2)
    plot_ops_area(sub_ax,transform=ccrs.PlateCarree(),color='k')

    # sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none', edgecolor='blue', linewidth=2)

    
    return(ax,startTimeDT,endTimeDT,day_str2)
