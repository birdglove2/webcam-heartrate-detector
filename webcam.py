import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

focus_point = [69, 108, 151, 337, 299, 296, 295, 285, 8, 55, 65, 66]

arr = []
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:

    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed Video Capture")
            break

        arr.append(frame)

        gap = 30 - (900 - len(arr)) / 30
        print(gap)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame)

        # Draw the face detection annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                # print(type(detection))

                x = int(
                    detection.location_data.relative_bounding_box.xmin
                    * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                y = int(
                    detection.location_data.relative_bounding_box.ymin
                    * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                w = int(
                    detection.location_data.relative_bounding_box.width
                    * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                h = int(
                    detection.location_data.relative_bounding_box.height
                    * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                faces = frame[y : y + h, x : x + w]
                forehead = frame[y - 200 : y - 500, x - 200 : x - 500]
                # print(faces)
                mp_drawing.draw_detection(frame, detection)

        # # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Face Detection", cv2.flip(forehead, 1))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
