#!/usr/bin/env python
# Script to download all .nc files from a THREDDS catalog directory
# This script was obtained from an OOI webpage:
# https://oceanobservatories.org/knowledgebase/how-can-i-download-all-files-at-once-from-a-data-request/
# Written by Sage 4/5/2016, revised 5/31/2018
  
from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve

# The THREDDS catalog I am trying to read is:
# http://smode.whoi.edu:8080/thredds/catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml

# Divide the url you get from the data portal into two parts
# Everything before "catalog/"
server_url = 'http://smode.whoi.edu:8080/thredds/'
# Everything after "catalog/"
request_url = 'catalog/IOP2_2023/satellite/VIIRS_NPP/catalog.xml'

def get_elements(url, tag_name, attribute_name):
  """Get elements from an XML file"""
  # usock = urllib2.urlopen(url)
  usock = urlopen(url)
  xmldoc = minidom.parse(usock)
  usock.close()
  tags = xmldoc.getElementsByTagName(tag_name)
  attributes=[]
  for tag in tags:
    attribute = tag.getAttribute(attribute_name)
    attributes.append(attribute)
  return attributes
 
def main():
  url = server_url + request_url
  print(url)
  catalog = get_elements(url,'dataset','urlPath')
  files=[]
  for citem in catalog:
    if (citem[-3:]=='.nc'):
      files.append(citem)
  count = 0
  for f in files:
    count +=1
    file_url = server_url + 'fileServer/' + f
    file_prefix = file_url.split('/')[-1][:-3]
    file_name = file_prefix + '_' + str(count) + '.nc'
    print('Downloaing file %d of %d' % (count,len(files)))
    print(file_name)
    a = urlretrieve(file_url,file_name)
    print(a)
 
# Run main function when in comand line mode        
if __name__ == '__main__':
  main()