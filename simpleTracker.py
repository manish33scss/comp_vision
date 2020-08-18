import cv2

video = cv2.VideoCapture(0)

#create trakerobject

tracker=cv2.TrackerMOSSE_create()
dim = (640, 480)
def drawBox(img, bbox):
    x,y,w,h = int(bbox[0]), int(bbox[1]) , int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y),((x+w),(y+h)), (255,0,0),1)
    cv2.putText(img , "tracking Object" , (100,76),cv2.FONT_HERSHEY_TRIPLEX ,0.7,(255,0,255),2)


while True:
    ret, img  = video.read()
    #img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("1st window >?" , img)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break

#create boundary box

bbox = cv2.selectROI(img , False)
tracker.init(img,bbox)
cv2.destroyWindow("roi selected")

while True:
    ret, img = video.read()
    #update the tracker
    ret , bbox = tracker.update(img)
    if ret:
        drawBox (img, bbox)

    cv2.imshow("Tracking" , img)


    k = cv2.waitKey(1)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
