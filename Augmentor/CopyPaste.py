# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 19:54:28 2019

@author: djava
"""
from PIL import Image
from random import randint
import cv2
import glob


#im1=Image.open("C:/Users/djava/OneDrive/Desktop/Donkey.jpg")
#im2=Image.open("C:/Users/djava/OneDrive/Desktop/rabbit.jpg")
#print(im1.size)
#print(im2.size)

temp_dir="/home/aeroclub/Abtin/CASE/Wrinkle_templates/output"
bckg_dir="/home/aeroclub/Abtin/CASE/Fabric_templates"
dtst_dir="/home/aeroclub/Abtin/CASE/Generated_files/Images/"
labls_dir="/home/aeroclub/Abtin/CASE/Generated_files/Labels/xywh/"
num=0

for bckg_img in glob.glob(bckg_dir + "/*.JPG"):
    bckg = Image.open(bckg_img)
    bckg_w , bckg_h=bckg.size
    for tmp_img in glob.glob(temp_dir + "/*.JPG"):
        tmp=Image.open(tmp_img)
        tmp_w , tmp_h=tmp.size
        
        #print(bckg_w, bckg_h, tmp_w, tmp_h)
        
        for i in range(2):
            new=bckg.copy()
            if bckg_w-tmp_w > 1 and bckg_h-tmp_h > 1:
                num+=1
                x=randint(1,bckg_w-tmp_w)
                y=randint(1,bckg_h-tmp_h)
                offset=(x, y)
                new.paste(tmp, offset)
                #new.show()
                new.save(dtst_dir + str(num).zfill(6) + ".jpg")
                f=open(labls_dir + str(num).zfill(6) + ".txt", "w+")
                f.write("1\n")
                # 1 is the number of class
                f.write(str(x) + " " + str(y) + " " + str(tmp_w) + " " + str(tmp_h))
                f.close()
