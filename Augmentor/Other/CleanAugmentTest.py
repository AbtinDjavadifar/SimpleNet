#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:57:26 2019

@author: aeroclub
"""

import Augmentor
import invoke
from invoke import run

# =============================================================================
# def Augment(inputpath, cmdspath):
#     cmds = open(cmdspath).readlines()
#     for i in range(len(cmds)):
#         p = Augmentor.Pipeline(inputpath)
#         run(cmds[i])
#         p.process()
#         
# inpath="/home/aeroclub/Abtin/Augmentor/figure_skating_templates"      
# cmpath="/home/aeroclub/Abtin/Augmentor/cmds.txt"
# Augment(inpath, cmpath)
# =============================================================================


p = Augmentor.Pipeline("/home/aeroclub/Abtin/Augmentor/figure_skating_templates")
cmds="p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)"
run(cmds, hide=True, warn=True)
p.process()
