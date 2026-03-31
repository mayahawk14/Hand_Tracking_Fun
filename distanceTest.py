import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np


#options available directly from mediapipe documentation, you can adjust them as needed
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, 
                                       num_hands=1, 
                                       min_hand_detection_confidence=0.8, 
                                       min_hand_presence_confidence=0.8, 
                                       min_tracking_confidence=0.8)

# Create the hand landmarker/detector
detector = vision.HandLandmarker.create_from_options(options)
# Start video capture (default camera is 0)
cap = cv2.VideoCapture(0)


while cap.isOpened():
    #read a frame from the webcam
    ret, frame = cap.read()
    #if no frame is read, break the loop
    if not ret:
        break
    
    #flip the frame horizontally for a mirror image
    frame = cv2.flip(frame, 1)
    h,w,_ = frame.shape
    #convert the frame from BGR to RGB format as mediapipe expects it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #create a mediapipe image object from the RGB frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    #add in the detector results to the mp_image object
    results = detector.detect(mp_image)
    
    #if hand landmarks are detected, draw a circle at the tip of the index finger (landmark 8)
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            index_tip = hand_landmarks[8]
            x,y = int(index_tip.x * w), int(index_tip.y * h)
            
            
            THUMB_FINGER_MCP = hand_landmarks[4]
            a,b = int(THUMB_FINGER_MCP.x * w), int(THUMB_FINGER_MCP.y * h)
            
            
            x1,y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2,y2 = int(THUMB_FINGER_MCP.x * w), int(THUMB_FINGER_MCP.y * h)
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distance = int(distance)
            cv2.circle(frame, (a,b), distance, (255,0,255), -1)
            cv2.circle(frame, (x,y), distance, (255,0,255), -1)
            
    # if hand landmarks are detected, calculate the distance between the tip of the index finger and the MCP joint of the thumb, and display it on the frame
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            index_tip = hand_landmarks[8]
            thumb_mcp = hand_landmarks[4]
            x1,y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2,y2 = int(thumb_mcp.x * w), int(thumb_mcp.y * h)
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            cv2.putText(frame, f'Distance: {int(distance)}', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

  

    
    cv2.imshow("Hand Tracking", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
