import numpy as np
import cv2
# Create our body classifier
body_classifier = cv2.CascadeClassifier('/media/ritik/Data/Personal Work/ComputerVision-master/haarcascade-files/haarcascade_fullbody.xml')
# Initiate video capture for video file
cap = cv2.VideoCapture('/home/ritik/townset dataset/input_video_s2_l1_06.mp4')
# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    # Pass frame to our body classifier

    bodies = body_classifier.detectMultiScale(gray, 1.3, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)
    if cv2.waitKey(1) == 27:
    	break
cap.release()
cv2.destroyAllWindows()
