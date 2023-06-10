# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Preliminary gradient calculations with near-real-time Wave Glider ADCP data-- 12.805 edition

# * Uses cleaned up S-MODE ADCP files made using ./S-MODE_analysis/code_IOP1/WG_realtime_velocity.ipynb
#
# first cut for IOP1 by [@jtomfarrar](https://github.com/jtomfarrar), 10/20/2022  
#
# In thinking about presenting this example for 12.805, I was reminded of the final pages of the course notes, which recount some advice Ken Brink gave to students when he taught 12.805:
# > <em>"There are a few points about this course that I would like you to remember and use for the rest of your oceanographic career. They are all common sense, but it's interesting how often people can forget to consider these once they become engaged in a problem. Or, sometimes people simply put these aside out of wishful thinking."
# >
# > 1. <em>The instruments that we use... are not perfect. It is important to understand how the instrument works, what it actually measures, what kind of accuracy (or errors) you can expect...
# >
# > 2. <em>Always plot your data...
# >
# > 3. <em>When you compute a number based on data, always evaluate the errors and/or statistical significance. Doing this makes your result quantitatively credible.
# >
# > 4. <em>Know the correlation time and length scales relevant to your observations, and take advantage of this information. You use this information, explicitly or implicitly, as you design measurements, for example: 1. How frequent should my time sampling be? 2. How close together should those stations be? 3. When does this survey become non-synoptic?
# >
# > <em>Often you may not know the scales before you do the field work, but you can still make intelligent guesses (such as the internal Rossby radius for a distance scale) or you can make estimates based on analogous observations elsewhere.
#     
# I employed all of those principles in this analysis.  Furthermore, we will also illustrate a couple of the principles of data analysis we outlined in the introduction of the notes:
# 1. Check your analysis method using a synthetic data record.  (Start with the answer and construct a synthetic data record to test your analysis.)
# 1. Simpler methods are usually better.  (We want people to understand what we did.)
#
#

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
from scipy.stats import binned_statistic_2d

plt.close('all')

# + tags=[]
# %matplotlib inline
# %matplotlib widget
# # %matplotlib qt5
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 400

__figdir__ = '../plots/vel_fits/' 
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# + tags=[]
savefig = True
if savefig:
    figdir_contents = __figdir__+'*.*'
    # !rm $figdir_contents
# -

# List of WGs
WG_list = ['WHOI32','WHOI43','STOKES', 'PLANCK', 'PASCAL', 'CARSON'] #leave Kelvin out because of ADCP problem in IOP1
path='../data/raw/WG_NRT/'

# Make a list of the files:
n=0
file_list = []
for WG in WG_list:
    file = path+'adcp_'+WG+'.nc'
    file_list.append(file)

file_list

# Read in cleaned ADCP files from all WG
n=0
for WG in WG_list:
    file = file_list[n]
    varstr = 'adcp_'+WG
    locals()[varstr]=xr.open_dataset(file,decode_times=True) #Time and z already fixed in WG_realtime_cleanup.ipynb
    n=n+1
    print(file)
    
    # attempt crude QC, dropping bins 3 and 7 (numbering from 1), 4 and 9 for Pascal
    #eval('adcp_'+WG).current_east.where(eval('adcp_'+WG).current_east!=eval('adcp_'+WG).current_east[6][:], drop=True)
    #eval('adcp_'+WG).current_east.where(eval('adcp_'+WG).current_east!=eval('adcp_'+WG).current_east[2][:], drop=True)
    #eval('adcp_'+WG).current_north.where(eval('adcp_'+WG).current_north!=eval('adcp_'+WG).current_north[6][:], drop=True)
    #eval('adcp_'+WG).current_north.where(eval('adcp_'+WG).current_north!=eval('adcp_'+WG).current_north[2][:], drop=True)
    #eval('adcp_'+WG).z_matrix.where(eval('adcp_'+WG).z_matrix!=eval('adcp_'+WG).z_matrix[6][:], drop=True)
    #eval('adcp_'+WG).z_matrix.where(eval('adcp_'+WG).z_matrix!=eval('adcp_'+WG).z_matrix[2][:], drop=True)
    #eval('adcp_'+WG).depth.where(eval('adcp_'+WG).depth!=eval('adcp_'+WG).depth[6], drop=True)
    #eval('adcp_'+WG).depth.where(eval('adcp_'+WG).depth!=eval('adcp_'+WG).depth[2], drop=True)


# + tags=[]
# Now we can access these in a loop using syntax like:
# eval('adcp_'+WG_list[7])
# -

ds = eval('adcp_'+WG_list[2])

ds

tmin = np.datetime64('2022-10-03T00:00:00')
tmax = np.datetime64('2022-11-02T00:00:00')
vmin = -.50 
vmax = .50
levels=np.arange(vmin,vmax,.05)

plt.figure()
plt.set_cmap(cmap=plt.get_cmap('turbo'))
n = 0
ax1 = plt.subplot(len(WG_list),1,len(WG_list))
ax1.set_xlim(tmin,tmax)
zmax=-60
for WG in WG_list:
    n=n+1
    ds = eval('adcp_'+WG)
    ax = plt.subplot(len(WG_list),1,n,sharex=ax1)
    im = plt.pcolor(ds.time.values,ds.depth,ds.current_east,vmin=vmin,vmax=vmax)
    # plt.contourf(ds.time.values,ds.z_matrix[:,1],ds.current_east,levels)
    plt.ylim(zmax, 0)
    plt.text(tmin,zmax+5,WG)
    if n==1: plt.title('East vel')
fig=plt.gcf()
fig.autofmt_xdate()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.875, 0.1, 0.025, 0.8])
fig.colorbar(im, cax=cbar_ax)

plt.figure()
plt.set_cmap(cmap=plt.get_cmap('turbo'))
n = 0
ax1 = plt.subplot(len(WG_list),1,len(WG_list))
ax1.set_xlim(tmin,tmax)
for WG in WG_list:
    n=n+1
    ds = eval('adcp_'+WG)
    ax = plt.subplot(len(WG_list),1,n,sharex=ax1)
    im = plt.pcolor(ds.time.values,ds.z_matrix,ds.current_north,vmin=vmin,vmax=vmax)
    # plt.contourf(ds.time.values,ds.z_matrix[:,1],ds.current_east,levels)
    plt.ylim(-60, 0)
    plt.text(tmin,zmax+5,WG)
    if n==1: plt.title('North vel')
fig=plt.gcf()
fig.autofmt_xdate()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.875, 0.1, 0.025, 0.8])
fig.colorbar(im, cax=cbar_ax)

# Do the same plots as the last two above, but showing only the data from the triangle.

# Plot locations
t0 = np.datetime64('2022-10-03T00:00:00')
t1 = np.datetime64('2022-10-21T00:00:00')
plt.figure()
n = 0
zmax=-60
for WG in WG_list:
    n=n+1
    ds = eval('adcp_'+WG)
    tind = np.flatnonzero(np.logical_and(ds.time>t0,ds.time<t1))
    plt.plot(ds.Longitude[tind], ds.Latitude[tind])
    # plt.contourf(ds.time.values,ds.z_matrix[:,1],ds.current_east,levels)
    plt.axis('square')
    plt.axis([-124.9, -124.0, 36.5, 37.5])
plt.legend(WG_list,loc='best')

# Locations of triangles:
# determined from google earth file (shift-ctrl-c copies google earth coordinates to clipboard)
#   36.846128° -124.427452°
triangle_coords = np.array([[36.890103, -124.621356],
                #[36.947069, -124.625868],
                [36.967925, -124.657486],
                [36.991903, -124.634525],
                [37.002247, -124.636256],
                #[37.045228, -124.575364],
                [36.84, -124.427]]) #36.846128, -124.427452

triangle_lat=triangle_coords[:,0]
triangle_lon=triangle_coords[:,1]

triangle_ind = 1
tol = .023


def plot_triangle_vel(triangle_ind,variable):
    plt.figure()
    plt.set_cmap(cmap=plt.get_cmap('turbo'))
    n = 0
    ax1 = plt.subplot(len(WG_list),1,len(WG_list))
    ax1.set_xlim(tmin,tmax)
    lon0 = triangle_lon[triangle_ind]
    lat0 = triangle_lat[triangle_ind]
    for WG in WG_list:
        n=n+1
        ds = eval('adcp_'+WG)
        # ds = ds.where(np.logical_and(np.abs(ds.Latitude.values-lat0)<tol, np.abs(ds.Longitude.values-lon0)<tol))
        tind = np.flatnonzero(np.logical_and(np.abs(ds.Latitude.values-lat0)<tol, np.abs(ds.Longitude.values-lon0)<tol))
        ax = plt.subplot(len(WG_list),1,n,sharex=ax1)
        if tind.size == 0: 
            im = plt.axis()
        else:
            im = plt.pcolor(ds.time[tind].values,ds.depth,eval('ds.'+variable)[:,tind],vmin=vmin,vmax=vmax)
            # plt.contourf(ds.time.values,ds.z_matrix[:,1],ds.current_east,levels)
        plt.ylim(-60, 0)
        plt.text(tmin,zmax+5,WG)
        if n==1: plt.title(variable + ', triangle ' + str(triangle_ind))
    fig=plt.gcf()
    fig.autofmt_xdate()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.1, 0.025, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    if savefig:
        plt.savefig(__figdir__+'WG_triangle_'+str(triangle_ind)+'_'+variable+'.'+plotfiletype,**savefig_args)


for triangle_ind in range(len(triangle_lat)):
    variable = 'current_east'
    plot_triangle_vel(triangle_ind,variable)

# # We want to estimate vorticity and divergence from these triangle arrays.  How can we do this?
#
# To estimate vorticity and divergence we need estimates of $\frac{\partial u}{\partial x}$, $\frac{\partial u}{\partial y}$, $\frac{\partial v}{\partial x}$, and $\frac{\partial v}{\partial y}$. 
#
# We _could_ just try to compute finite differences from the measured data.
#
# We will fit a plane to the measured velocity components.  Conceptually, we are thinking of the velocity as being,
# \begin{equation}
#     u(x,y) = u_0 + \frac{\partial u}{\partial x}\Delta x + \frac{\partial u}{\partial y}\Delta y
# \end{equation}
#
# We could write this system of equations in vector form as:
# \begin{eqnarray}
#   \begin{bmatrix} u_{n=1} \\ u_{n=2} \\ u_{n=3} \end{bmatrix} = u_0 \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} +\frac{\partial u}{\partial x} \begin{bmatrix} \Delta x_{n=1} \\ \Delta x_{n=2} \\ \Delta x_{n=3} \end{bmatrix} +\frac{\partial u}{\partial y} \begin{bmatrix} \Delta y_{n=1} \\ \Delta y_{n=2} \\ \Delta y_{n=3} \end{bmatrix}
# \end{eqnarray}
# To simplify the notation, let $\frac{\partial u}{\partial x}\equiv u_x$, $\frac{\partial u}{\partial y}\equiv u_y$, $\Delta x\equiv x$, and $\Delta y \equiv y$.  Then the above equation is the same as
# \begin{eqnarray}
#   \begin{bmatrix} u_{n=1} \\ u_{n=2} \\ u_{n=3} \end{bmatrix} = \begin{bmatrix} 1 & x_{n=1} & y_{n=1} \\ 1 & x_{n=2} & y_{n=2} \\ 1 & x_{n=3} & y_{n=3} \end{bmatrix} \begin{bmatrix} u_0 \\ u_x \\u_y  \end{bmatrix} \\
#   \mathbf{y} \ \ \  = \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathbf{E} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathbf{x} \ \ \ \ \
# \end{eqnarray}
#
#
# The ordinary least squares solution is (Eqns 1.127 or 1.208 of class notes):
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\mathbf{E}^T\mathbf{E}\right)^{-1}\mathbf{E}^T\mathbf{y}.
#  \label{TW_LS2}
# \end{equation}
#
# The solution uncertainty is given by the diagonals of:
# \begin{equation}
# {\bf P} = {\bf C}_{{\tilde {\bf x}}{\tilde {\bf x}}} + {\bf bb}^T.
# \label{P_TWLS}
# \end{equation}
# where $\mathbf{b}$ is the bias of the estimate and
# \begin{equation}
# \label{eq:Cxx_TWLS}
# {\bf C}_{{\tilde{\bf x}}{\tilde{\bf x}}} = ({\bf E}^T{\bf E})^{-1} {\bf E}^T {\bf C}_{nn} {\bf E}  ({\bf E}^T {\bf E})^{-1}
# \end{equation}
# is the solution covariance (the diagonals of which are the variance of the estimate).  I am going to focus on the variance of the estimate for now, and ignore the bias.
#

# # How to organize code for the fits?
#
# The whole situation is kind of a mess.  We have 6 vehicles that went in and out of the triangle formations.  We had to turn their instruments on and off to save power when solar charging was weak. 
# <br>
#
# We're going to need $u$ and $v$ written as column vectors.  (We don't have to do that, but I find it conceptually easier for matching up with the $\mathbf{y=Ex}$ framework.)  We're also going to want to be able to handle each "time" (time window) separately.
#
# One way to do it would be to dump all of the U, V, lon, lat, time values into column vectors and then select the data points that are within some lon/lat/time distance of a given target.  That should work.

# + active=""
#

# +
# dump all of the U, V, lon, lat, time values into vectors and then select the data points that are within some lon/lat/time distance of a given target
# I guess this will have to be Ntimes x Nz, where N = len(time)*len(WG_list)

# This will only work if all files have the same depths-- check that
Ntimes=0
Nz_check = []
z_check = []
for WG in WG_list:
    ds = eval('adcp_'+WG)
    Ntimes=Ntimes+len(ds.time)
    Nz_check.append(len(ds.depth)) # doing this to verify that all files have the same number of depths so I can concatenate variables like U(z,t)
    z_check.append(np.min(ds.depth)) # doing this to verify that a single depth vector can work for all files

if np.diff(Nz_check).all()==0 and np.diff(z_check).all()==0 :
    Nz=Nz_check[0]
    z=ds.depth
else:
    raise ValueError('There are different numbers of depth points in the files')
# -

U = np.zeros((Ntimes, Nz))
np.shape(U)

# +
U = []
V = []
lon = []
lat = []
time = []
name = []

for WG in WG_list:
    ds = eval('adcp_'+WG)
    #tind = np.flatnonzero(np.logical_and(np.abs(ds.Latitude.values-lat0)<tol, np.abs(ds.Longitude.values-lon0)<tol))
    U.extend(ds.current_east.transpose())
    V.extend(ds.current_north.transpose())
    time.extend(ds.time.values)
    lon.extend(ds.Longitude)
    lat.extend(ds.Latitude)
    name.extend(np.repeat(WG,len(ds.time.values)))

U = np.array(U)
V = np.array(V)
lon = np.array(lon)
lat = np.array(lat)
time = np.array(time)
name = np.array(name)
# -

print(np.shape(lat),np.shape(lon),np.shape(time),np.shape(U),np.shape(V))

# ## OK, we have vectors of lon, lat, time, and matrices of U and V.  Let's write a function to find all the velocity measurements at a given depth and time that are within a speficied distance of one of the triangles.

# +
z0 = -15
zind = np.flatnonzero(np.abs(z-z0)<1)

lon0 = -124.66
lat0 = 36.96
lon_lat_tol = .023
t0 = np.datetime64('2022-10-13T12:00:00')
time_tol = np.timedelta64(1,'h')


# -

def subset_vel(lon0, lat0, t0, lon_lat_tol, time_tol, zind):
    tind = np.flatnonzero(np.logical_and(np.logical_and(np.abs(lat-lat0)<lon_lat_tol, np.abs(lon-lon0)<lon_lat_tol),np.abs(time-t0)<time_tol))
    lonsub = lon[tind]
    latsub = lat[tind]
    timesub = time[tind]
    Usub = U[tind,zind]
    Vsub = V[tind,zind]
    namesub = name[tind]
    xsub = []
    for n in range(len(lonsub)):
        loni=lonsub[n]
        xsub.append(np.sign(loni-lon0)*gsw.geostrophy.distance((loni,lon0), (lat0,lat0)))
    ysub = []
    for lati in latsub:
        ysub.append(np.sign(lati-lat0)*gsw.geostrophy.distance((lon0,lon0), (lati,lat0)))
    
    return namesub, lonsub, latsub, xsub, ysub, timesub, Usub, Vsub


namesub, lonsub, latsub, xsub, ysub, timesub, Usub, Vsub = subset_vel(lon0,lat0,t0, lon_lat_tol, time_tol, zind)

plt.figure()
# plt.scatter(xsub, ysub,c=Vsub)
plt.quiver(xsub, ysub,Usub, Vsub)
plt.title(', '.join(np.unique(namesub)) + ' data, ' + str(abs(lon0)) + 'W, ' + str(lat0) + 'N, time = ' + str(t0) )
plt.axis('equal')

# We want to estimates of the partial derivatives $u_x$, $u_y$, $v_x$, and $v_y$.  We're going to make these estimates by fitting planes to the data, as expressed by this equation (and an analogous one for $v$):
# \begin{eqnarray}
#   \begin{bmatrix} u_{n=1} \\ u_{n=2} \\ u_{n=3} \end{bmatrix} = \begin{bmatrix} 1 & x_{n=1} & y_{n=1} \\ 1 & x_{n=2} & y_{n=2} \\ 1 & x_{n=3} & y_{n=3} \end{bmatrix} \begin{bmatrix} u_0 \\ u_x \\u_y  \end{bmatrix} \\
#   \mathbf{y} \ \ \  = \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathbf{E} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathbf{x} \ \ \ \ \
# \end{eqnarray}
#
# For easy reference, I am just repeating the equations from above. The ordinary least squares solution is (Eqns 1.127 or 1.208 of class notes):
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\mathbf{E}^T\mathbf{E}\right)^{-1}\mathbf{E}^T\mathbf{y}.
#  \label{TW_LS2}
# \end{equation}
#
# To think about doing this in python using the numpy.linalg library, it might help to rewrite it as:
# \begin{equation}
#   \left(\mathbf{E}^T\mathbf{E}\right)\tilde{\mathbf{x}}=\mathbf{E}^T\mathbf{y}.
# \end{equation}
#
# The solution uncertainty is given by the diagonals of:
# \begin{equation}
# {\bf P} = {\bf C}_{{\tilde {\bf x}}{\tilde {\bf x}}} + {\bf bb}^T.
# \label{P_TWLS}
# \end{equation}
# where $\mathbf{b}$ is the bias of the estimate and
# \begin{equation}
# \label{eq:Cxx_TWLS}
# {\bf C}_{{\tilde{\bf x}}{\tilde{\bf x}}} = ({\bf E}^T{\bf E})^{-1} {\bf E}^T {\bf C}_{nn} {\bf E}  ({\bf E}^T {\bf E})^{-1}
# \end{equation}
# is the solution covariance (the diagonals of which are the variance of the estimate).  I am going to focus on the variance of the estimate for now, and ignore the bias.
#
#

# So, let's define E and y
# first row of E is 1 xsub[0] ysub[0]
E=[]
for n in range(len(xsub)):
    E.append([1,xsub[n].item(),ysub[n].item()])

E = np.asarray(E)
y = Usub

np.shape(E)

# ### I was wondering if explicitly computing the inverse for the LS solution would be needlessly expensive
#
# Specifically, I was wondering about whether one of these would be better:
# The ordinary least squares solution is (Eqns 1.127 or 1.208 of class notes):
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\mathbf{E}^T\mathbf{E}\right)^{-1}\mathbf{E}^T\mathbf{y}.
#  \label{TW_LS2}
# \end{equation}
#
# To think about doing this in python using the numpy.linalg library, it might help to rewrite it as:
# \begin{equation}
#   \left(\mathbf{E}^T\mathbf{E}\right)\tilde{\mathbf{x}}=\mathbf{E}^T\mathbf{y}.
# \end{equation}

A = E.transpose()@E
B = E.transpose()@y

# %%timeit
xtilde1 = np.linalg.solve(A, B) # solves A x = B

# %%timeit
part1 = np.linalg.inv(A)
xtilde2 = part1@B # solves A x = B

# Wow-- I am surprised.  It must be that python does not evaluate $\mathbf{A}^{-1}$.  When I have done similar things with Matlab, I thought computing the inverse explicitly really slowed things down.  Let's see if the two methods give the same result.

xtilde1 = np.linalg.solve(A, B) # solves A x = B
part1 = np.linalg.inv(A)
xtilde2 = part1@B # solves A x = B

u0, ux, uy = xtilde1
print('xtilde1: u0 = ' + str(u0) + ', ux = ' + str(ux) + ', uy = ' + str(uy))
u0, ux, uy = xtilde2
print('xtilde2: u0 = ' + str(u0) + ', ux = ' + str(ux) + ', uy = ' + str(uy))


# It looks like they give the same result to something like the 13th decimal place.  I prefer the explicit inverse for conceptual clarity in the code.

# ## Make a function to do the plane fit

def plane_fit_2D(Usub,xsub,ysub,sig):
    ''' 
    Inputs: 
    Usub = scalar field to fit
    xsub = x locations
    ysub = y locations
    sig = velocity noise standard deviation (single number)
    
    Outputs:
    Estimates of:
    u0 (value at x=0,y=0)
    ux (x slope of input field)
    uy (y slope of input field)
    sig_u0 (standard error of the estimate of u0)
    sig_ux (standard error of the estimate of ux)
    sig_uy (standard error of the estimate of uy)
    
    '''
    
    E=[]
    for n in range(len(xsub)):
        E.append([1,xsub[n].item(),ysub[n].item()])
    E = np.asarray(E)
    y = Usub
    ETE = E.transpose()@E
    B = E.transpose()@y
    u0, ux, uy = np.linalg.inv(ETE)@B # np.linalg.solve(ETE, B) # solves ETE@x = B
    
    # error estimate
    # I am assuming that each ADCP has the same error and no bias
    Cnn = (sig**2)*(np.eye(len(Usub)))
    Cxx = np.linalg.inv(E.transpose()@E)@E.transpose()@Cnn@E@np.linalg.inv(E.transpose()@E)
    sig_u0, sig_ux, sig_uy = np.sqrt(np.diag(Cxx))
    
    return u0, ux, uy, sig_u0, sig_ux, sig_uy


sig = 0.01 #noise standard deviation in m/s
u0, ux, uy, sig_u0, sig_ux, sig_uy = plane_fit_2D(Usub,xsub,ysub,sig)
print('xtilde2: u0 = ' + str(u0) + ', ux = ' + str(ux) + ', uy = ' + str(uy))
print('estimated errors: sig_u0 = ' + str(sig_u0) + ', sig_ux = ' + str(sig_ux) + ', sig_uy = ' + str(sig_uy))

# # Generate a test dataset and make sure this works as expected.  This is an easy and VERY IMPORTANT step!
#
# The basic idea is to test your numerical solution with something that you know the answer for.  This is easy to do by doing the _forward problem_.  
#
# I will just assume values for $u_0$, $u_x$ and $u_y$ and generate synthetic values for $u(x,y)$ using the $(x,y)$ values of my measurements:
# \begin{eqnarray}
#   \begin{bmatrix} u_{n=1} \\ u_{n=2} \\ u_{n=3} \end{bmatrix} = \begin{bmatrix} 1 & x_{n=1} & y_{n=1} \\ 1 & x_{n=2} & y_{n=2} \\ 1 & x_{n=3} & y_{n=3} \end{bmatrix} \begin{bmatrix} u_0 \\ u_x \\u_y  \end{bmatrix} \\
#   \mathbf{y} \ \ \  = \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathbf{E} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathbf{x} \ \ \ \ \
# \end{eqnarray}
# Then, we can easily check that inverse estimate is producing a sensible answer.  I am also adding measurement noise, which allows me to check the error estimate.
#
# ### This step is super important!  In fact, I had a coding/conceptual error in my error estimate that I found because my estimated errors were much bigger than my actual errors.

# +
# Generate a test dataset
u0 = np.array(0.2)
ux = np.array(1e-5)
uy = np.array(-2.3e-5)
sig = 0.01 #standard deviation for some added noise, imagining 1 cm/s ADCP error

# Here, I am just using the real values of xsub and ysub to generate a synthetic realization of Usub, with noise:
Usub = u0+ ux*xsub + uy*ysub +np.random.default_rng().normal(0, sig, size=np.shape(xsub))
# -

u0, ux, uy, sig_u0, sig_ux, sig_uy = plane_fit_2D(Usub,xsub,ysub,sig)
print('u0 = ' + str(u0) + ', ux = ' + str(ux) + ', uy = ' + str(uy))
print('estimated errors: sig_u0 = ' + str(sig_u0) + ', sig_ux = ' + str(sig_ux) + ', sig_uy = ' + str(sig_uy))

# ### Notice below what happens with the error estimates when one component of the gradient is poorly constrained.  We will place three observations almost in a line.

# +
# 3 data points with nearly the same x coordinate
xsub = np.asarray([500, 501, 500]) # x coordinates almost the same-- not a very good triangle
ysub = np.asarray([-500, 500, 0]) 

u0 = np.array(0.2)
ux = np.array(1e-5)
uy = np.array(-2.3e-5)
sig = 0.01 #standard deviation for some added noise, imagining 1 cm/s ADCP error

# Here, I am just using the real values of xsub and ysub to generate a synthetic realization of Usub, with noise:
Usub = u0+ ux*xsub + uy*ysub +np.random.default_rng().normal(0, sig, size=np.shape(xsub))
# -

u0, ux, uy, sig_u0, sig_ux, sig_uy = plane_fit_2D(Usub,xsub,ysub,sig)
print('u0 = ' + str(u0) + ', ux = ' + str(ux) + ', uy = ' + str(uy))
print('estimated errors: sig_u0 = ' + str(sig_u0) + ', sig_ux = ' + str(sig_ux) + ', sig_uy = ' + str(sig_uy))

# ### $u_y$ is constrained much better than $u_x$, because we had three observations at nearly the same $x$ value.  Also, $u_0$ is poorly constrained because the line of measurements was far from $x=0$. 

# ## OK, the fit looks like it is working, and now we are set up to fit the whole time series!

# + tags=[]
z0 = -4 # choose a target depth
zind = np.flatnonzero(np.abs(z-z0)<1) # find the depth that corresponds to target depth

triangle_ind = 4
lon_lat_tol = .023
time_tol = np.timedelta64(1,'h')
#t0 = np.datetime64('2022-10-13T12:00:00')
# -

t = np.arange('2022-10-09T12:00:00', 'now',15, dtype='datetime64[m]')


def do_triangle_fit_z0(z0,triangle_ind,plot_flag = False):
    zind = np.flatnonzero(np.abs(z-z0)<1)
    lon0 = triangle_lon[triangle_ind]
    lat0 = triangle_lat[triangle_ind]

    if plot_flag: plt.figure()
    ax = [-2000, 2000, -2000, 2000]
    u0 = []; ux = []; uy = []; v0 = []; vx = []; vy = []; sig_u0 = []; sig_ux = []; sig_uy = []; sig_v0 = []; sig_vx = []; sig_vy = []
    n=0
    for t0 in t:
        namesub, lonsub, latsub, xsub, ysub, timesub, Usub, Vsub = subset_vel(lon0,lat0,t0, lon_lat_tol, time_tol, zind)
        # Do Least squares fits
        if len(xsub)>9:
            u0i, uxi, uyi, sig_u0i, sig_uxi, sig_uyi = plane_fit_2D(Usub,xsub,ysub,sig)
            v0i, vxi, vyi, sig_v0i, sig_vxi, sig_vyi = plane_fit_2D(Vsub,xsub,ysub,sig)
            if plot_flag:# Plot
                n=n+1
                plt.clf()
                plt.quiver(xsub, ysub,Usub, Vsub)
                plt.quiver(0,0,u0i,v0i,color='r')    
                plt.axis('square')
                plt.title(str(t0))
                plt.text(-1000,1000,', '.join(np.unique(namesub)))
                plt.axis(ax)
                if savefig:
                    plt.savefig(__figdir__+'triangle_vel_plots_'+str(n)+'.'+plotfiletype,**savefig_args)
        else:
            u0i, uxi, uyi, sig_u0i, sig_uxi, sig_uyi = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            v0i, vxi, vyi, sig_v0i, sig_vxi, sig_vyi = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        u0.append(u0i);ux.append(uxi);uy.append(uyi)
        v0.append(v0i);vx.append(vxi);vy.append(vyi)
        sig_u0.append(sig_u0i);sig_ux.append(sig_uxi);sig_uy.append(sig_uyi)
        sig_v0.append(sig_v0i);sig_vx.append(sig_vxi);sig_vy.append(sig_vyi)

    u0 = np.array(u0)
    ux = np.array(ux)
    uy = np.array(uy)
    v0 = np.array(v0)
    vx = np.array(vx)
    vy = np.array(vy)
    sig_u0 = np.array(sig_u0)
    sig_ux = np.array(sig_ux)
    sig_uy = np.array(sig_uy)
    sig_v0 = np.array(sig_v0)
    sig_vx = np.array(sig_vx)
    sig_vy = np.array(sig_vy)
    
    return u0, v0, ux, vx, uy, vy, sig_u0, sig_ux, sig_uy, sig_v0, sig_vx, sig_vy


sig = 0.02

# + tags=[]
triangle_ind=4
z0=-4
plot_flag = False
u0, v0, ux, vx, uy, vy, sig_u0, sig_ux, sig_uy, sig_v0, sig_vx, sig_vy = do_triangle_fit_z0(z0=-4,triangle_ind=triangle_ind,plot_flag = plot_flag)
# -

plot_names = __figdir__+'triangle_vel_plots_%d.png'
movie_name = __figdir__ + 'triangle_vel_movie.avi'

if plot_flag:
    # !ffmpeg -y -framerate 8 -i $plot_names $movie_name
    print('OK')



f = gsw.geostrophy.f(37)
inertial_period = 2*np.pi/f/3600
print('Inertial period is '+ f'{inertial_period:.1f}' + ' hours')

# ## OK, we did the fits.  Now form vorticity, divergence, strain rate magnitude
# Vorticity: $\frac{\zeta}{f} = \frac{v_x-u_y}{f}$  
# Divergence: $\frac{\delta}{f} = \frac{u_x+v_y}{f}$  
# Strain rate magnitude: $\frac{\sigma}{f} = \frac{\sqrt{(u_x-v_y)^2+(v_x+u_y)^2}}{f}$  
#

# +
zeta=vx-uy
sig_zeta=np.sqrt(sig_vx**2 + sig_uy**2)
strain = np.sqrt((ux-vy)**2+(vx+uy)**2)
sig_strain=np.sqrt(sig_vx**2 + sig_uy**2 + sig_ux**2 + sig_vy**2) # Note that "sig" refers to error estimate here
div = ux+vy
sig_div=np.sqrt(sig_ux**2 + sig_vy**2)
feff_f=np.sqrt(1+zeta/f)

plt.figure(figsize=(5,6))
capsize = 3
skip = 4

ax = plt.subplot(4,1,1)
plt.errorbar(t,u0,yerr=sig_u0,capsize=capsize,errorevery=skip)
plt.errorbar(t,v0,yerr=sig_v0,capsize=capsize,errorevery=skip)
plt.ylabel('u,v')
plt.title('Triangle \"' + str(triangle_ind) +'\", depth =' + str(-z0))

ax2 = plt.subplot(4,1,2,sharex=ax)
# plt.plot(t,(div)/f)
plt.errorbar(t,div/f,yerr=sig_div/f,capsize=capsize,errorevery=skip)
plt.plot(t,0*div,'--',color='k')
plt.ylabel('$\delta$/f')
plt.ylim(-2, 2)

ax3 = plt.subplot(4,1,3,sharex=ax)
# plt.plot(t,(strain)/f)
plt.errorbar(t,strain/f,yerr=sig_strain/f,capsize=capsize,errorevery=skip)
plt.plot(t,0*div,'--',color='k')
plt.ylabel('$\sigma$/f')
plt.ylim(-0.5,3)

ax3 = plt.subplot(4,1,4,sharex=ax)
# plt.plot(t,(zeta)/f)
hz = plt.errorbar(t,zeta/f,yerr=sig_zeta/f,capsize=capsize,errorevery=skip)
hfeff, = plt.plot(t,feff_f)
plt.plot(t,0*div,'--',color='k')
plt.plot(t,1+0*div,'--',color='grey')
plt.ylabel('$\zeta$/f')
plt.ylim(-2, 2)
ax3.legend([hz, hfeff],['$\zeta$/f', '$f_{effective}$'])

fig = plt.gcf()
fig.autofmt_xdate()

if savefig:
    plt.savefig(__figdir__+'WG_triangle_fit_'+str(-z0)+'m_triangle_'+str(triangle_ind)+'.'+plotfiletype,**savefig_args)

# +
zeta=vx-uy
sig_zeta=np.sqrt(sig_vx**2 + sig_uy**2)
strain = np.sqrt((ux-vy)**2+(vx+uy)**2)
sig_strain=np.sqrt(sig_vx**2 + sig_uy**2 + sig_ux**2 + sig_vy**2) # Note that "sig" refers to error estimate here
div = ux+vy
sig_div=np.sqrt(sig_ux**2 + sig_vy**2)
feff_f=np.sqrt(1+zeta/f)

plt.figure(figsize=(5,6))
capsize = 3
skip = 4

ax = plt.subplot(3,1,1)
plt.errorbar(t,u0,yerr=sig_u0,capsize=capsize,errorevery=skip)
plt.errorbar(t,v0,yerr=sig_v0,capsize=capsize,errorevery=skip)
plt.ylabel('u,v')
plt.title('Triangle \"' + str(triangle_ind) +'\", depth =' + str(-z0))

ax2 = plt.subplot(3,1,2,sharex=ax)
# plt.plot(t,(div)/f)
plt.errorbar(t,div/f,yerr=sig_div/f,capsize=capsize,errorevery=skip)
plt.plot(t,0*div,'--',color='k')
plt.ylabel('$\delta$/f')
plt.ylim(-2, 2)

ax3 = plt.subplot(3,1,3,sharex=ax)
# plt.plot(t,(zeta)/f)
hz = plt.errorbar(t,zeta/f,yerr=sig_zeta/f,capsize=capsize,errorevery=skip)
# hfeff, = plt.plot(t,feff_f)
plt.plot(t,0*div,'--',color='k')
# plt.plot(t,1+0*div,'--',color='grey')
plt.ylabel('$\zeta$/f')
plt.ylim(-2, 2)
# ax3.legend([hz, hfeff],['$\zeta$/f', '$f_{effective}$'])

fig = plt.gcf()
fig.autofmt_xdate()

if savefig:
    plt.savefig(__figdir__+'WG_triangle_fit_simple_'+str(-z0)+'m_triangle_'+str(triangle_ind)+'.'+plotfiletype,**savefig_args)


# -

def do_fit_all_depths(zgrid,triangle_ind):
    n=0
    for z0 in zgrid:
        n=n+1
        u0, v0, ux, vx, uy, vy, sig_u0, sig_ux, sig_uy, sig_v0, sig_vx, sig_vy = do_triangle_fit_z0(z0,triangle_ind,plot_flag = False)
        if n==1:
            u0z=[u0]
            v0z=[v0]
            uxz=[ux]
            vxz=[vx]
            uyz=[uy]
            vyz=[vy]
        else:
            u0z=np.append(u0z,[u0],axis=0)
            v0z=np.append(v0z,[v0],axis=0)
            uxz=np.append(uxz,[ux],axis=0)
            vxz=np.append(vxz,[vx],axis=0)
            uyz=np.append(uyz,[uy],axis=0)
            vyz=np.append(vyz,[vy],axis=0)
    return u0z, v0z, uxz, vxz, uyz, vyz



def plot_fit_UV(t, zgrid, u0z, v0z,triangle_ind):
    plt.figure(figsize=(7, 4))
    levels=np.linspace(vmin,vmax,21)
    ax = plt.subplot(2,1,1)
    plt.title('Triangle \"' +str(triangle_ind) + '\"')
    plt.pcolor(t,zgrid,u0z,vmin=0, vmax=.75)
    #ax.xaxis.set_ticklabels([])
    plt.colorbar(orientation='vertical',label='U (m/s)')
    plt.subplot(2,1,2,sharex=ax)
    plt.pcolor(t,zgrid,v0z,vmin=-.25, vmax=.25)
    plt.colorbar(orientation='vertical',label='V (m/s)')
    fig = plt.gcf()
    fig.autofmt_xdate()
    if savefig:
        plt.savefig(__figdir__+'WG_triangle_fit_UV_triangle_'+str(triangle_ind)+'.'+plotfiletype,**savefig_args)



def plot_fit_div_zeta(t, zgrid, zeta, div, strain,triangle_ind):
    plt.figure(figsize=(7, 6))
    levels=np.linspace(-2, 2,21)
    ax = plt.subplot(3,1,1)
    plt.title('Triangle \"' +str(triangle_ind) + '\"')
    plt.pcolor(t,zgrid,zeta,vmin=-2,vmax=2)
    plt.colorbar(orientation='vertical',label='$\zeta$/f')
    plt.subplot(3,1,2,sharex=ax)
    plt.pcolor(t,zgrid,div,vmin=-2,vmax=2)
    plt.colorbar(orientation='vertical',label='$\delta$/f')
    plt.subplot(3,1,3,sharex=ax)
    plt.pcolor(t,zgrid,strain,vmin=-2,vmax=2)
    plt.colorbar(orientation='vertical',label='$\sigma$/f')
    fig = plt.gcf()
    fig.autofmt_xdate()
    if savefig:
        plt.savefig(__figdir__+'WG_triangle_fit_zeta_triangle_'+str(triangle_ind)+'.'+plotfiletype,**savefig_args)



t = np.arange('2022-10-09T12:00:00', 'now',15, dtype='datetime64[m]')
triangle_ind = 4
zgrid = np.arange(-4,-50,-2)
# zgrid = np.delete(zgrid,[2,3,6])
u0z, v0z, uxz, vxz, uyz, vyz = do_fit_all_depths(zgrid,triangle_ind)
tind = np.flatnonzero(~np.isnan(u0z[0,:]))
u0z=u0z[:,tind]
v0z=v0z[:,tind]
uxz=uxz[:,tind]
vxz=vxz[:,tind]
uyz=uyz[:,tind]
vyz=vyz[:,tind]
t=t[tind]

# +
zeta = (vxz-uyz)/f #vx-uy
div = (uxz+vyz)/f
strain = np.sqrt((uxz-vyz)**2+(vxz+uyz)**2)/f # This is strain magnitude
plot_fit_UV(t, zgrid, u0z, v0z,triangle_ind)
plot_fit_div_zeta(t, zgrid, zeta, div, strain,triangle_ind)

outfile = '../data/processed/vorticity_fits/triangle_' + str(triangle_ind) + '.npz'
np.savez(outfile, t=t, zgrid=zgrid, u0z=u0z, v0z=v0z, uxz=uxz, vxz=vxz, uyz=uyz, vyz=vyz, zeta=zeta, div=div, strain=strain)
# -

zeta1 =[]
div1=[]
strain1=[]
for triangle_ind in [1, 4]:
    outfile = '../data/processed/vorticity_fits/triangle_' + str(triangle_ind) + '.npz'
    file_contents = np.load(outfile)
    zeta1 = np.append(zeta1,file_contents['zeta'])
    div1 = np.append(div1,file_contents['div'])
    strain1 = np.append(strain1,file_contents['strain'])
    #div = (uxz+vyz)/f
    #strain = (uyz+vxz)/f


# +
# Do a binned vorticity-strain plot

x_bins = np.linspace(-4, 4, 160)
y_bins = np.linspace(0, 4, 80)
# for PDF
ret = binned_statistic_2d(zeta1.flatten(), np.abs(strain1.flatten()), np.ones(np.shape(div1)).flatten(), statistic=np.sum, bins=[x_bins, y_bins])
#for bin avg
#ret = binned_statistic_2d(zeta1.flatten(), np.abs(strain1.flatten()), div1.flatten(), statistic=np.mean, bins=[x_bins, y_bins])

plt.figure(figsize=(7, 3))
#plt.pcolor(x_bins,y_bins,ret.statistic.T,vmin=-2,vmax=2)
plt.pcolor(x_bins,y_bins,np.log10(ret.statistic.T))
plt.plot([0,4],[0,4],'k--')
plt.plot([-4,0],[4,0],'k--')
plt.colorbar()
# -

# ##  One way we might do this better is with a weighted LS solution, or a TWLS solution
# Consider the tapered and weighted least-squares solution (Equation 1.125 in the course notes),
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\mathbf{E}^T\mathbf{W}^{-1}\mathbf{E}+\mathbf{S}^{-1}\right)^{-1}\left(\mathbf{E}^T\mathbf{W}^{-1}\mathbf{y}+\mathbf{S}^{-1}\mathbf{x_0}\right).
#  \label{TW_LS}
# \end{equation}
# Recall that $\mathbf{W}^{-1}$ is a ''weight matrix'', $\mathbf{S}^{-1}$ is a ''taper matrix'' (which can be thought of as another weight matrix, as we shall see soon), and $\mathbf{x_0}$ is the first guess solution.  Just to simplify the notation and discussion a little bit, we will assume that $\mathbf{x_0}=0$, which would be the case if we know or think that the expectation value $<\mathbf{x}>=0$.  In that case
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\mathbf{E}^T\mathbf{W}^{-1}\mathbf{E}+\mathbf{S}^{-1}\right)^{-1}\mathbf{E}^T\mathbf{W}^{-1}\mathbf{y}.
#  \label{TW_LS2}
# \end{equation}
#
# Again assuming $\mathbf{x_0}=0$, the cost function that led to Equation \ref{TW_LS2} was (Equation 1.193 in the notes):
# \begin{equation}
#   J=\mathbf{n}^T \mathbf{W}^{-1}\mathbf{n}+\mathbf{x}^T\mathbf{S}^{-1}\mathbf{x}.
#  \label{J_TW_LS2}
# \end{equation}
#
# Like most complicated equations, we can get a better feel for what the equation means by considering some special cases.  A common special case to consider in matrix problems is one where some matrices are diagonal and square, because these matrices can easily be inverted.  If $\mathbf{W}=a \mathbf{I}$, then $\mathbf{W}^{-1}=\frac{1}{a} \mathbf{I}$. So, let's try letting $\mathbf{W}^{-1}=\frac{1}{\sigma_n^2} \mathbf{I}$ and letting $\mathbf{S}^{-1}=\frac{1}{\Delta_x^2} \mathbf{I}$.  Then, the cost function in Equation \ref{J_TW_LS2} becomes
# \begin{equation}
#   J=\frac{1}{\sigma_n^2}\mathbf{n}^T \mathbf{n}+\frac{1}{\Delta_x^2}\mathbf{x}^T \mathbf{x},
#  \label{J_TW_LS2_simple}
# \end{equation}
# and Equation \ref{TW_LS2} becomes:
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\frac{1}{\sigma_n^2}\mathbf{E}^T \mathbf{E}+\frac{1}{\Delta_x^2}\mathbf{I}\right)^{-1}\frac{1}{\sigma_n^2}\mathbf{E}^T\ \mathbf{y},
#   \nonumber
# \end{equation}
# or,
# \begin{equation}
#   \tilde{\mathbf{x}}=\left(\mathbf{E}^T\mathbf{E}+\frac{\sigma_n^2}{\Delta_x^2}\mathbf{I}\right)^{-1}\mathbf{E}^T \mathbf{y}.
#  \label{TW_LS2_simple}
# \end{equation}
# If $\sigma_n^2$ is the expected noise variance and $\Delta_x^2$ is the expected solution variance, then we can interpret Equation \ref{J_TW_LS2_simple} as a cost function where we equally penalize (in a normalized sense) the estimated noise variance and the estimated solution variance.  We are simultaneously minimizing the model-data misfit and the solution variance.
#
#
#
#
#
# The tapering parameter $\sigma_n^2/\Delta_x^2$ can be considered to be an inverse signal-to-noise ratio (SNR), expressing our expectation about the relative variance of the measurement noise and the solution.  In the limit that the tapering parameter is very small (meaning the SNR is high), Equation \ref{TW_LS2_simple} is just the ordinary least squares solution.  If the tapering parameter is small, the tapered least squares solution could also be viewed as a mere computational trick-- by adding a small value to the diagonal of $\mathbf{E}^T\mathbf{E}$, we have guaranteed that the inverse $\left(\mathbf{E}^T\mathbf{E}+\frac{\sigma_n^2}{\Delta_x^2}\mathbf{I}\right)^{-1}$ exists.
#




