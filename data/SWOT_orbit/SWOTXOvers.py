from pykml import parser
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt

def ParseKMLCoordinates(filename):
    """
        Parse geographic coordinates from a kml file

        ---
        Input: 
              - filename [string]

        Returns:
              - lon [array]
              - lat [array]
    """

    with open(filename) as f:
        folder = parser.parse(f).getroot().Document.Folder

    plnm=[]
    cordi=[]

    for pm in folder.Placemark:
        plnm1=pm.name
        plcs1=pm.Point.coordinates
        plnm.append(plnm1.text)
        cordi.append(plcs1.text)

    lat, lon = [], []
    for i in range(len(cordi)):
        dummy_coords = cordi[i].split(',')
        lon.append(float(dummy_coords[0]))
        lat.append(float(dummy_coords[1]))
    
    return np.array(lon), np.array(lat)

def GetOrbit(filename):
    """
        Get orbit geographic coordinates from
        a txt file.

        ---
        Input: 
              - filename [string]

        Returns:
              - lon [array]
              - lat [array]

    """
    orbit = np.loadtxt(filename)
    lon, lat = orbit[:,1], orbit[:,2]
    lon[lon>180] = lon[lon>180]-360
    return lon, lat

if __name__ == "__main__":

    # Get geographic coordinates of SWOT's fast-sampling (CalVal) orbit, 
    #   CalVal's cross-over points and Science orbits

    calval_xover_filename   = 'swot_calval_orbite_xover.kml'
    calval_orbit_filename   = 'ephem_calval_june2015_ell-v2.txt'
    science_orbit_filename = 'ephem_science_sept2015_ell-v2.txt'

    lonx,latx = ParseKMLCoordinates(calval_xover_filename)
    lon_calval, lat_calval = GetOrbit(calval_orbit_filename)
    lon_science, lat_science = GetOrbit(science_orbit_filename)

    # Sanity-check plot
    fig = plt.figure()
    plt.plot(lon_calval,lat_calval,'o',markersize=3)
    plt.plot(lonx,latx,'ro')
    plt.savefig('SWOTCalValOrbit')

    # Save matfile
    mat_dict = {'lon': lon_science,'lat':lat_science,'lon_calval':lon_calval,
                'lat_calval': lat_calval ,'lon_xover': lonx, 'lat_xover': latx}
    savemat("SWOTOrbit.mat",mat_dict)
