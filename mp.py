import numpy as np
import cv2 
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

FOREHEAD_POINTS = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69]

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0,0,255)
thickness              = 3
lineType               = 2

text = 'Calculating HR...'

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  start = time.time()

  arr = []
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    end = time.time()
    
    fps = 1/ (end-start )
    start = end 
    arr.append(image)
    gap = (300 - len(arr)) / 30
    # if len(mean_list) < WINDOW_SIZE:

    text = 'Calculating HR, wait {:.1f} s'.format(gap)
    print(text)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    contour = {"forehead": []}
    height, width, _ = image.shape
    
    
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for i in FOREHEAD_POINTS:
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                contour["forehead"].append([x, y])
                
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None
        )
        cv2.drawContours(image, [np.array(contour["forehead"])], 0, (0,0,255), 2)
                
    
    cv2.putText(image, "Calculating HR...", (contour["forehead"][0][0] -500, contour["forehead"][0][1] -200), 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()