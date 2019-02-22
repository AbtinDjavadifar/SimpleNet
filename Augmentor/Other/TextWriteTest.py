#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:33:45 2019

@author: aeroclub
"""

Path="/home/aeroclub/Abtin/Augmentor/figure_skating_templates/"
a=2
b=3
c=4.5
d=6.47
f=open(Path + str(30).zfill(6) + ".txt", "w+")
f.write("1\n")
# 1 is the number of class
f.write(str(a) + " " + str(b) + " " + str(c) + " " + str(d))
f.close()