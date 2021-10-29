# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:22:34 2021

@author: jtomf
"""


import urllib.request
import ssl
import certifi


ssl._create_default_https_context = ssl._create_unverified_context
testfile = urllib.request.urlopen("http://webcam.oregonstate.edu/cam/oceanus/live/live.jpg?1635357871568",cafile=None, capath=None,context=None)


# testfile = urllib.request.urlopen("https://webcam.oregonstate.edu/cam/oceanus/live/live.jpg?1635357871568")
#testfile = urllib.URLopener()
foo = testfile.read("https://webcam.oregonstate.edu/cam/oceanus/live/live.jpg?1635357871568", "file.jpg")
