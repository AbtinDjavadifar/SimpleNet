#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:36:26 2019

@author: aeroclub
"""

from PIL import Image
#import glob
import os


LoadPath="/home/aeroclub/Abtin/CASE/Simple_Net/Images/def_ply011"
SavePath="/home/aeroclub/Abtin/CASE/Simple_Net/Images/chunks/"
k=0
height=81
width=108

#for img in glob.glob(LoadPath + "/*.png"):
for filename in [f for f in os.listdir(LoadPath) if f.endswith(".jpg")]:
    filepath=os.path.join(LoadPath, filename)
    im = Image.open(filepath)
    imgwidth, imgheight = im.size
    for i in range(0,imgwidth,width):
        for j in range(0,imgheight,height):
            box = (i, j, i+width, j+height)
            a = im.crop(box)
            a.save(os.path.join(SavePath, filename[:-4] + "-X-%s-Y-%s" % (i, j) + ".jpg"))         
            k +=1

