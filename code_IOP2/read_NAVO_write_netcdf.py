#  Ben Greenwood's code to decode NAVO float BUFR files and write them to netcdf
#


import glob
import json
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatJsonRenderer
import datetime
import netCDF4

incoming = '/Library/Webserver/Documents/SMODE/navo_float/BUFR'
json_dir = '/Library/Webserver/Documents/SMODE/navo_float/json'
THREDDS  = '/Volumes/raid1/thredds/public/SMODE/IOP2_2023/navo_float'

floats = [4903089,4903092,4903095,4903097,4903099,4903101,4903089,4903093,4903096,4903098,4903100]

for file in sorted(glob.glob('%s/A_IOPX*.txt' % incoming)):
  print(file)
  decoder = Decoder()
  msg = decoder.process(open(file,'rb').read())
  d = FlatJsonRenderer().render(msg)
  t = d[4][2][0][9:14]
  time = datetime.datetime(t[0],t[1],t[2],t[3],t[4],0)
  type = d[4][2][0][5]
  if type == 15:
    type = 'APEX'
  cycle = d[4][2][0][6]
  id = d[4][2][0][0]
  lat = d[4][2][0][14]
  lon = d[4][2][0][15]

  rec = []
  i0 = 19 # offset of first CTD
  while i0 < len(d[4][2][0])-9:
    i0 = i0+9
    pres = float(d[4][2][0][i0])/10000.0;               # convert Pa to dbar
    temp  = round(float(d[4][2][0][i0+3]) - 273.15,3);  # degC
    psal  = float(d[4][2][0][i0+6]);                    # pracical salinity
    r = {'pres':pres,'temp': temp, 'psal': psal}
    rec.append(r)

  # Write out JSON file
  fname = '%s/APEX_%s_%03d.json' % (json_dir,id,cycle)
  json.dump({'wmo_id':id,'type':type,'cycle':cycle,'time':time.strftime('%Y-%m-%dT%H:%M:%SZ'),'lat':lat,'lon':lon,'scans':rec},open(fname,'w'),indent=2)

# Create NetCDF
for apex in floats:
  d = netCDF4.Dataset('%s/APEX_%s.nc' % (THREDDS,apex),'w',format='NETCDF4')
  d.createDimension('dive')
  d.createDimension('pressure')
  d.title = 'S-MODE IOP2 NAVO APEX Float WMO %s' % apex
  d.summary = 'NAVO APEX Float WMO %s was deployed from R/V Sally Ride and is participating in S-MODE IOP2 Campaign' % apex
  d.wmo_id = apex
  d.history = '%s Ben Greenwood - decode raw APEX float BUFR data, concatenate profiles into NetCDF' % datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
  d.createVariable('dive','i4',('dive'))
  d['dive'].long_name = 'APEX float profile number'
  d['dive'].units = 1
  d.createVariable('latitude','f8',('dive'))
  d['latitude'].long_name = 'latitude recorded by APEX float'
  d['latitude'].standard_name = 'latitude'
  d['latitude'].units = 'degrees_north'
  d['latitude'].axis = 'Y'
  d.createVariable('longitude','f8',('dive'))
  d['longitude'].long_name = 'longitude recorded by APEX float'
  d['longitude'].standard_name = 'longitude'
  d['longitude'].units = 'degrees_east'
  d['longitude'].axis = 'X'
  d.createVariable('time','f8',('dive'))
  d['time'].long_name = 'time recorded by APEX float'
  d['time'].standard_name = 'time'
  d['time'].units = 'seconds since 1970-01-01T00:00:00Z'
  d['time'].axis = 'T'
  d.createVariable('pressure','f4',('pressure'))
  d['pressure'].long_name = 'pressure of APEX float CTD measurement'
  d['pressure'].units = 'dbar'
  d.createVariable('temperature','f4',('pressure'))
  d['temperature'].long_name = 'temperature of APEX float CTD measurement'
  d['temperature'].standard_name = 'sea_water_temperature'
  d['temperature'].units = 'degrees_C'
  d.createVariable('salinity','f4',('pressure'))
  d['salinity'].long_name = 'APEX float practical salinity CTD measurement'
  d['salinity'].standard_name = 'sea_water_practical_salinity'
  d['salinity'].units = '1'

  idive = 0
  iscan = 0
  for file in sorted(glob.glob('%s/APEX_%s_*.json' % (json_dir,apex))):
    print(file)
    c = json.load(open(file))
    time = datetime.datetime.strptime(c['time'],'%Y-%m-%dT%H:%M:%SZ').timestamp()
    d['dive'][idive] = c['cycle']
    d['latitude'][idive] = c['lat']
    d['longitude'][idive] = c['lon']
    d['time'][idive] = time
    idive = idive + 1
    for scan in c['scans']:
      d['pressure'][iscan] = scan['pres']
      d['temperature'][iscan] = scan['temp']
      d['salinity'][iscan] = scan['psal']
      iscan = iscan + 1
  d.close()
