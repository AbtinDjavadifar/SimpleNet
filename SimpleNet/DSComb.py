#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:10:33 2019

@author: aeroclub
"""

from pyds import MassFunction

v1 = MassFunction({'w':0.6, 'f':0.3, 'g':0.1, 'b':0.0})
v2 = MassFunction({'w':0.5, 'f':0.4, 'g':0.1, 'b':0.0})
v3 = MassFunction({'w':0.7, 'f':0.2, 'g':0.1, 'b':0.0})
v4 = MassFunction({'w':0.3, 'f':0.3, 'g':0.4, 'b':0.0})

print('Dempster\'s combination rule for m_1 and m_2 =', v1 & v2 & v3 & v4)