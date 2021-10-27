import cv2
import numpy as np
import glob
import os
 
img_array = []
for filename in os.scandir(r"D:\Work\Data\steps"):
    img = cv2.imread(filename.path)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('try1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
