import cv2
import mediapipe as mp
import time
import numpy as np

# 1. Setup MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
model_path = 'hand_landmarker.task'  # Download this file!
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Hand Connection Map (to draw lines manually)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky/Palm
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)
pTime = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, img = cap.read()
        if not success: break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # Process detection
        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 2. MANUAL DRAWING (Bypasses 'framework' errors)
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                # Get pixel coordinates
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                # Draw Connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(img, coords[start_idx], coords[end_idx], (0, 255, 0), 2)

                # Draw Fingertips & Points
                for i, (cx, cy) in enumerate(coords):
                    # Highlight fingertip IDs from original code
                    if i in [4, 8, 12, 16, 20]:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    else:
                        cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

        # FPS Calc
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Hand Tracker (Tasks API)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
