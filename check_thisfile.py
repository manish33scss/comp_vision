# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:59:49 2019

@author: Manish
"""


import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from centroidtracker import CentroidTracker
import csv
from collections import deque


# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = (r"D:\Manish_Mtech\New folder\yolo\pytorch-yolo-v3\coco.names")
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = (r"D:\Manish_Mtech\New folder\yolo\pytorch-yolo-v3\cfg\yolov3.cfg")
modelWeights = (r"D:\Manish_Mtech\New folder\yolo\pytorch-yolo-v3\yolov3.weights")

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    if classId==2 or classId==3 or classId==5:
        
        cv.rectangle(frame, (left, top), (right, bottom), (0,150, 250),thickness=1)
        a,b=[(left+right)/2, (top+bottom)/2]
        centre=(int(a),int(b))
        #for x,y in centre:
            #if  :
        #print("centroid",a,b)
        label = '%.2f' % conf
            
        # Get the label for the class name and its confidence
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        #cv.circle(frame, centre, 1, (0,240,30), thickness=1, lineType=8, shift=0)
        #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 255, 255), thickness=1,lineType=8)
        cv.putText(frame, label, (left, top),  cv.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        #print(i)
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        
        
        
#__________________ Tracking ___________________
        
ct = CentroidTracker()
(H, W) = (None, None)
fourcc = cv.VideoWriter_fourcc(*'XVID') 
    
cap = cv.VideoCapture(r"D:\Manish_Mtech\New folder\yolo\pytorch-yolo-v3\video.mp4")

import collections as css
pts = css.deque(maxlen=100000)
frames = 0
import time
start = time.time()    
past_c= css.deque()#empty list 
cent= css.deque()
# Get the video writer initialized to save the output video

output = cv.VideoWriter('_ssysolo_track11.avi', fourcc,9, (640, 480))
while cv.waitKey(1) < 0:
    key=cv.waitKey(1)
    if key & 0xFF == ord('q'):
                break
    # get frame from the video
    hasFrame, frame = cap.read()
    frames+=1
    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        #print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)
    detections = net.forward(getOutputsNames(net))
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    
 
    conf_threshold = 0.6
    nms_threshold = 0.4
    things = []
    car=[]
    people = []
    confidences_ppl = []
    confidences_car=[]
    confidences_things = []
    class_ids = []
    for out in detections:
    	    for detection in out:
    	        scores = detection[5:]
    	        class_id = np.argmax(scores)
    	        confidence = scores[class_id]
    	        if confidence > 0.5:
    	            center_x = int(detection[0] * Width)
    	            center_y = int(detection[1] * Height)
    	            w = int(detection[2] * Width)
    	            h = int(detection[3] * Height)
    	            x = center_x - w / 2
    	            y = center_y - h / 2
    	            if class_id == 2:
    	            	confidences_car.append(float(confidence))
    	            	car.append([round(x), round(y), round(w), round(h)])
    	            
    indices_p = cv.dnn.NMSBoxes(car, confidences_car, conf_threshold, nms_threshold)
    
    rects_f = []
    for i in indices_p:
    	    i = i[0]
    	    box = car[i]
    	    x = box[0]
    	    y = box[1]
    	    w = box[2]
    	    h = box[3]
           #cv.cirle(frame,(round(x),round(y)),(0,255,0),thickness=1,lineType=8)
    	    rects_f.append((round(x),round(y),round(x+w),round(y+h)))
    	    #cv.rectangle(frame,(round(x),round(y)), (round(x+w),round(y+h)), (0, 50, 50),thickness=1)
            
    objects = ct.update(rects_f)
    
    
    for(objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        if ct.disappeared[objectID] == 0:
            cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), lineType=cv.LINE_AA)
            cv.circle(frame, (centroid[0], centroid[1]),1, (0, 255, 0), -1)
            cv.line(img=frame, pt1=(60, 330), pt2=(600, 330), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
            #cv.line(img=frame,pt1=(centroid[0]),pt2=(centroid[1]),color=(255,0,0),thickness=1,lineType=8,shift=0)
            
            for point in past_c:
                
                if text == point[0]:
                    cv.circle(frame, (point[1], point[2]),1, (0, 255, 0), 2)
                    #cv.line(frame,pt1=())
                    point.append(cent)
            #print(type(center))
            '''for i in range(1,len(point)):
                
                print()
                if(point[i-1] is None or point[i] is None):
                    continue
                cv.line(frame, point[i - 1], point[i], (0, 0, 255), thickness=1,lineType=8,shift=0)
                '''
                 
                 
            
            past_c.appendleft([text,centroid[0],centroid[1]])
            pts.appendleft(centroid) 
            print(type(pts))
    
    
   
    #for i in range(1, len(pts)):
     #   if pts[i - 1] is None or pts[i] is None:
      #      continue
       # cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness=1,lineType=8,shift=0)
        #break
        
    fps = cap.get(cv.CAP_PROP_FPS)      
    for (objectID,centroid) in objects.items():
        
            a=np.array(centroid[0])
            b=np.array(centroid[1])
            
            cv.putText(frame,"fps is :"+format(fps)+" count of cars per frame : "+str(len(objects)), (50,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 255, 255), lineType=cv.LINE_AA)
            
                # show the output frame
    resize = cv.resize(frame, (640, 480), interpolation = cv.INTER_LINEAR)
    
    cv.imshow("output", resize)
    output.write(resize)
    key = cv.waitKey(1) & 0xFF
     
    	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
    		break
 
# do a bit of cleanup
cap.release()
cv.destroyAllWindows()

