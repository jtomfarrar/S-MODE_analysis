# %% [markdown]
# ## S-MODE IOP2 summary plots
# Generate plots through time with S-MODE asset postions.
# 
# I was doing this withthe output from SMODE_VIIRS_climatology.ipynb on an EC2 instance using the VIIRS data with clear scenes saved to Zarr format.  
# 
# 
# Now, I want to instead use the higher resolution VIIRS data that are already subsetted on the WHOI SMODE THREDDS server
# Here is an example of how this can be done, from https://rabernat.github.io/research_computing_2018/xarray-tips-and-tricks.html
'''


http://smode.whoi.edu:8080/thredds/dodsC/IOP2_2023/satellite/VIIRS_NPP/VIIRS_NPP_20230309T090000Z.nc
'''
# %%
import os
import xarray as xr
import functions
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker, rc, cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from cartopy.io.shapereader import Reader

# %matplotlib widget

# %%
os.system('cd ~/Python/S-MODE_analysis/code_analysis')

# %%
plt.rcParams['figure.figsize'] = (7.5,6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 400

savefig = True # set to true to save plots as file

__figdir__ = '../plots/IOP1_summary/'
os.system('mkdir ' + __figdir__) #make directory if it doesn't exist
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'


# %%
WG_list = ['WHOI22','WHOI32','WHOI43','STOKES', 'PLANCK', 'KELVIN', 'CARSON','PASCAL','WHOI1102']
path='../data/raw/WG_NRT/'

# Make a list of the files:
n=0
file_list = []
for WG in WG_list:
    file = path+'adcp_'+WG+'.nc'
    file_list.append(file)

# Read in cleaned ADCP files from all WG
n=0
for WG in WG_list:
    file = file_list[n]
    varstr = 'adcp_'+WG
    try:
        locals()[varstr]=xr.open_dataset(file,decode_times=True) #Time and z already fixed in WG_realtime_cleanup.ipynb
        print(file)
    except:
        WG_list.remove(WG)
        print('There is no file for WG ' + WG)
    n=n+1
    

# %% 
# Read in Bold Horizon position data
BH_position_file = 'http://smode.whoi.edu:8080/thredds/dodsC/IOP1_2022/Bold_Horizon/BH_drifter_position/SMODE_IOP1_surface_drifter_0-4442680.nc'
ship_position = xr.open_dataset(BH_position_file)



# %%
# Center location
#site = 'S-MODE'
site = 'S-MODE IOP1'
#site = 'S-MODE IOP2 zoom'
#site = 'SPURS-1'

if site == 'S-MODE':
    lon0 = -123.5
    lat0 = 37.5 
    dlon = 2.5 # half of box width in lon
    dlat = 1.5 # half of box width in lat
elif site == 'S-MODE IOP1':
    lon0 = -124.5
    lat0 = 36.75
    dlon = 1.5 # half of box width in lon
    dlat = 1.0 # half of box width in lat
elif site == 'S-MODE IOP2':
    lon0 = -124.5
    lat0 = 36.5
    dlon = 2.5 # half of box width in lon
    dlat = 2.0 # half of box width in lat
elif site == 'S-MODE IOP2 zoom':
    lon0 = -124.9
    lat0 = 37.1
    dlon = 1.5 # half of box width in lon
    dlat = 1.0 # half of box width in lat

# Define the max/min lon
lon_min = lon0 - dlon
lon_max = lon0 + dlon
lat_min = lat0 - dlat
lat_max = lat0 + dlat

# Define a box where we want to check for good sst data (may be different than larger analysis domain)
# Still centered on lon0,lat0
dlon = 0.5 # half of box width in lon
dlat = 0.5 # half of box width in lat

# Define the max/min lon
x1 = lon0 - dlon
x2 = lon0 + dlon
y1 = lat0 - dlat
y2 = lat0 + dlat

# %%
ship_position


# %%
# Function to plot WG position and vel at time of SST image
def plot_ship_positions(ax,t0,skip, **kwargs):
    # List of WGs
    # Read in cleaned ADCP files from all WG
    tmin = t0 - np.timedelta64(12,'h')#np.datetime64('now')
    tmax = t0 + np.timedelta64(12,'h')#np.datetime64('now')
    n=0
    ds = ship_position
    tind = np.flatnonzero(np.logical_and(ds.time>tmin,ds.time<tmax))
    tind=tind[0:-1:skip]
    if tind.size==0:
        print('Skipping ship')
    else:
        #ax.scatter(ds.Longitude[tind[-1]].values,ds.Latitude[tind[-1]].values,s=5,color='k',transform=ccrs.PlateCarree())
        ax.plot(ds.longitude[tind].values,ds.latitude[tind].values,color='m',transform=ccrs.PlateCarree())
            


# %%
# Function to plot WG position and vel at time of SST image
def plot_WG_positions(ax,t0,skip, **kwargs):
    # List of WGs
    # Read in cleaned ADCP files from all WG
    tmin = t0 - np.timedelta64(12,'h')#np.datetime64('now')
    tmax = t0 + np.timedelta64(12,'h')#np.datetime64('now')
    n=0
    for WG in WG_list:
        ds = eval('adcp_'+WG)
        tind = np.flatnonzero(np.logical_and(ds.time>tmin,ds.time<tmax))
        tind=tind[0:-1:skip]
        if tind.size==0:
            print('Skipping '+WG)
            continue
        else:
            #ax.scatter(ds.Longitude[tind[-1]].values,ds.Latitude[tind[-1]].values,s=5,color='k',transform=ccrs.PlateCarree())
            ax.plot(ds.Longitude[tind].values,ds.Latitude[tind].values,color='k',transform=ccrs.PlateCarree())
            

# %%
def plot_SST_map(ds2, mean_SST):
    
    fig = plt.figure()
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=-125))  # Orthographic
    extent = [lon_min, lon_max,lat_min, lat_max]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # This string of the time of the SST selected will be useful
    day_str = np.datetime_as_string(ds2.time,unit='m')


    plt.set_cmap(cmap=plt.get_cmap('turbo')) #try turbo, nipy_spectral and gist_ncar
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title('SST, ' + day_str[0], size = 10.)

    # Does not do what I think it should:
    ax.minorticks_on()
    ax.tick_params(axis='both',which='both',direction='inout',top=True, right=True,zorder=100)

    V = [mean_SST+i for i in [-3,3]] #range for colorbar

    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=3, facecolor=[.6,.6,.6], edgecolor='black')
    cs = ax.pcolormesh(ds2.lon,ds2.lat,ds2.sea_surface_temperature.squeeze()-274.15,vmin=V[0],vmax=V[1],transform=ccrs.PlateCarree())
    # cs = ax.pcolormesh(ds2.lon,ds2.lat,ds2.sea_surface_temperature.squeeze().notnull(),transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs,fraction = 0.022,extend='both')
    cb.set_label('SST [$\circ$C]',fontsize = 10)
    functions.plot_ops_area_IOP1(ax,transform=ccrs.PlateCarree(),color='k')


# %% 
# http://smode.whoi.edu:8080/thredds/catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml
# url for the data (use xml extension instead of html)
ds_name = 'VIIRS_NRT'
# ds_name = 'VIIRS_N20'
server_url = 'http://smode.whoi.edu:8080/thredds/'
request_url = 'catalog/IOP1_2022/satellite/' + ds_name + '/catalog.xml'
url = server_url + request_url
sst_filelist = functions.list_THREDDS(server_url,request_url)
# Cannot use open mf_dataset because number of x-y points changes from pass to pass


# %%
good_files=[]
skip = 1
for n in range(0,len(sst_filelist)):
    ds = xr.open_dataset(sst_filelist[n])
    day_str = ds.time.dt.strftime("%Y-%m-%d-%H%MZ").values
    # print(day_str[0])
    if ds.time<np.datetime64('2022-09-26T09') or ds.time>np.datetime64('2022-11-10T09'):
        continue
    good_data = ds.quality_level.where(np.bitwise_and(np.bitwise_and(ds.lat<y2,ds.lat>y1),np.bitwise_and(ds.lon>x1,ds.lon<x2)))
    # dtime_sub = ds.sst_dtime.where(np.bitwise_and(np.bitwise_and(ds.lat<y2,ds.lat>y1),np.bitwise_and(ds.lon>x1,ds.lon<x2))).squeeze().mean()
    mean_sst = ds.sea_surface_temperature.where(np.bitwise_and(np.bitwise_and(ds.lat<y2,ds.lat>y1),np.bitwise_and(ds.lon>x1,ds.lon<x2))).squeeze().mean()-274.15
    # day_str = np.datetime_as_string(ds.time+dtime_sub,unit='m')
    print(day_str[0])

    # ds.quality_level is =1 for bad data and 5 for good data; Transform good_data values to span 0-1:
    good_data2=(good_data-1)/4
    if good_data2.squeeze().mean()>.65:
        good_files.append(sst_filelist[n])
        plot_SST_map(ds, mean_sst)
        ax = plt.gca()
        t0 = ds.time.values
        plot_WG_positions(ax,t0,skip)
        plot_ship_positions(ax,t0,skip)
        # Export figure
        if savefig:
            plt.savefig(__figdir__+'map_'  + day_str[0] + ds_name + '.' +plotfiletype,**savefig_args)
        # plt.show(block=False)

# %%
for n in range(0,len(good_files)):
    ds = xr.open_dataset(good_files[n])
    day_str = ds.time.dt.strftime("%Y-%m-%d-%H%MZ").values
    print(day_str[0])

    # ds.quality_level is =1 for bad data and 5 for good data; Transform good_data values to span 0-1:
    ds2 = ds
    ds2['sea_surface_temperature']= ds.sea_surface_temperature.where(ds.quality_level > 1)
    mean_sst = ds2.sea_surface_temperature.where(np.bitwise_and(np.bitwise_and(ds.lat<y2,ds.lat>y1),np.bitwise_and(ds.lon>x1,ds.lon<x2))).squeeze().mean()-274.15
 
    plot_SST_map(ds2, mean_sst)
    ax = plt.gca()
    t0 = ds.time.values
    plot_WG_positions(ax,t0,skip)
    plot_ship_positions(ax,t0,skip)
    # Export figure
    if savefig:
        plt.savefig(__figdir__+'map_QC_'  + day_str[0] + ds_name + '.' +plotfiletype,**savefig_args)
    # plt.show(block=False)




# %%
