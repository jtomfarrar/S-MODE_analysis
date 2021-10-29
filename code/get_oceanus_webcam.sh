# -*- coding: utf-8 -*-
#
# Bash script for reading and saving Oceanus webcam images during S-MODE cruise
# Created on Thu Oct 28 11:50:54 2021
# @author: jtomf



#!/bin/bash

while :
do
  curl  -k -o "$(date +"%Y_%m_%d_%I_%M_%p").jpg" https://webcam.oregonstate.edu/cam/oceanus/live/live.jpg 
	sleep 300
done

