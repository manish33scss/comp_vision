# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:12:28 2019

@author: Manish
"""

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from centroidtracker import CentroidTracker


# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "E:\Manish_mtech\yolo\pytorch-yolo-v3\coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "E:\Manish_mtech\yolo\pytorch-yolo-v3\cfg\yolov3.cfg"
modelWeights = "E:\Manish_mtech\yolo\pytorch-yolo-v3\yolov3.weights"

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
    if classId==2 or classId==3:
        
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
        cv.circle(frame, centre, 1, (0,240,30), thickness=1, lineType=8, shift=0)
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
        #a,b=[(left+right)/2, (top+bottom)/2]
        #entre=(int(a),int(b))
# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
ct = CentroidTracker()
(H, W) = (None, None)
outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    
    cap = cv.VideoCapture(r"E:\Manish_mtech\yolo\pytorch-yolo-v3\video.mp4")

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    key=cv.waitKey(1)
    if key & 0xFF == ord('q'):
                break
    # get frame from the video
    hasFrame, frame = cap.read()
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
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
    '''label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)
cap.release()
cv.destroyAllWindows()'''
 
    conf_threshold = 0.5
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
    	            #else:
    	            	#confidences_things.append(float(confidence))
    	            	#things.append([round(x), round(y), round(w), round(h)])
    	            	#class_ids.append(class_id)
    	# apply non-max suppression
    #indices_t = cv.dnn.NMSBoxes(things, confidences_things, conf_threshold, nms_threshold)
    indices_p = cv.dnn.NMSBoxes(car, confidences_car, conf_threshold, nms_threshold)
    #print(indices_p)
    '''for i in indices_t:
    	    i = i[0]
    	    box = things[i]
    	    x = box[0]
    	    y = box[1]
    	    w = box[2]
    	    h = box[3]
    	    cv.rectangle(frame,(round(x),round(y)), (round(x+w),round(y+h)), ( 255,0, 0), 2)
    	    cv.putText(frame,classes[class_ids[i]], (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)'''
    
    rects_f = []
    for i in indices_p:
    	    i = i[0]
    	    box = car[i]
    	    x = box[0]
    	    y = box[1]
    	    w = box[2]
    	    h = box[3]
    	    rects_f.append((round(x),round(y),round(x+w),round(y+h)))
    	    cv.rectangle(frame,(round(x),round(y)), (round(x+w),round(y+h)), (0, 50, 50),thickness=1)
    
    objects = ct.update(rects_f)
    #print(objects)
    	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
    		# draw both the ID of the object and the centroid of the
    		# object on the output frame
    		text = "ID {}".format(objectID)
    		if ct.disappeared[objectID] == 0:
    			cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType=cv.LINE_AA)
    			cv.circle(frame, (centroid[0], centroid[1]),1, (0, 255, 0), -1)
                
                # show the output frame
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
     
    	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
    		break
 
# do a bit of cleanup
cap.release()
cv.destroyAllWindows()
