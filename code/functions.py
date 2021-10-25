# -*- coding: utf-8 -*-
"""
Some function definitions for S-MODE analysis tools

Created on Sun Oct 24 10:37:33 2021

@author: jtomfarrar
"""



def plot_ops_area(ax,**kwargs):
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

