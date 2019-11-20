# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:20:22 2019

@author: Manish
"""
#Code to extract nth frame from a Video : source _ OpenCV
import cv2
 
# Opens the Video file
cap= cv2.VideoCapture(r'D:\duh\project\data\p10.avi')
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%15 == 0:
        cv2.imwrite('project'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()
