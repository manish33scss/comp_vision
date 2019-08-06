# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:18:25 2019

@author: Manish
"""

# organize imports 
import numpy as np 
import cv2 
  
# This will return video from the first webcam on your computer. 
cap = cv2.VideoCapture(r"D:\duh\project\data\video1.avi")   
  
# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('output.avi', fourcc, 60, (640, 480)) # 60 fps
  
# loop runs if capturing has been initialized.  
while(True): 
    # reads frames from a camera  
    # ret checks return at each frame 
    ret, frame = cap.read()  
    resize = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR) 
    # Converts to HSV color space, OCV reads colors as BGR 
    # frame is converted to hsv 
    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV) 
      
    # output the frame 
    out.write(hsv)  
      
    # The original input frame is shown in the window  
    cv2.imshow('Original', hsv) 
  
    # The window showing the operated video stream  
    #cv2.imshow('frame', hsv) 
  
      
    # Wait for 'a' key to stop the program  
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break
  
# Close the window / Release webcam 
cap.release() 
  
# After we release our webcam, we also release the output 
out.release()  
  
# De-allocate any associated memory usage  
cv2.destroyAllWindows() 
