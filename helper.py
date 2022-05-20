import mediapipe as mp
import numpy as np
from scipy.fft import fftfreq
import cv2
import random

# =============== Config ===============

USE_HSV = True
USE_WEBCAM = False
USE_CHEEK = True
VIDEO_FILENAME = "video/test.mp4"
FPS = 30
DURATION = 10

# =============== Global vars ===============

FOREHEAD_POINTS = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69]
CHEEK1_POINTS = [329, 348, 347, 280, 411, 427, 436, 426, 266, 329, 348]
CHEEK2_POINTS = [100, 119, 118, 50, 187, 207, 216, 206, 36]


WINDOW_SIZE = FPS * DURATION

MIN_HR_BPM, MAX_HR_BMP = 48, 144

FREQS = fftfreq(WINDOW_SIZE, 1 / FPS)

INVALID_IDX = np.where((FREQS < 50) | (FREQS > 180))

# =============== =============== ===============


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# get detected face from the webcam
# return False if failed, otherwise
# return True, frame of the camera, and the face
def detect_face(cap):
    if USE_WEBCAM:
        success, frame = cap.read(cv2.IMREAD_UNCHANGED)
        if not success:
            print("Ignoring empty camera frame.")
            return False, None, None
    else:
        if not cap.more():
            return False, None, None
        frame = cap.read()

    # To improve performance, optionally mark
    # the frame as not writeable to pass by reference.
    frame.flags.writeable = False

    face = mp_face_mesh.process(frame)
    return True, frame, face


def moving_avg(signal, w_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, "valid")
    return moving_avg


# find FOREHEAD_POINTS position in the frame
# mask the points by 1, other by False value
# return the masked frame or ROI
def extract_roi(frame, face):
    contour = {"forehead": [], "cheek1": [], "cheek2": []}
    height, width, _ = frame.shape

    if face.multi_face_landmarks:
        for face_landmarks in face.multi_face_landmarks:
            for i in FOREHEAD_POINTS:
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                contour["forehead"].append([x, y])

            if USE_CHEEK:
                for i in CHEEK1_POINTS:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    contour["cheek1"].append([x, y])
                for i in CHEEK2_POINTS:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    contour["cheek2"].append([x, y])

                cv2.drawContours(
                    frame, [np.array(contour["cheek1"])], 0, (0, 0, 255), 2
                )
                pos = (contour["cheek1"][0][0] - 200, contour["cheek1"][0][1] - 20)

                cv2.drawContours(
                    frame, [np.array(contour["cheek2"])], 0, (0, 0, 255), 2
                )
                pos = (contour["cheek2"][0][0] - 200, contour["cheek2"][0][1] - 20)

            # draw contour
            cv2.drawContours(frame, [np.array(contour["forehead"])], 0, (0, 0, 255), 2)
            pos = (contour["forehead"][0][0] - 200, contour["forehead"][0][1] - 20)

            # getting 2D array with same width and height
            mask = np.zeros((height, width))

            # fill 1 only the forehead, 0 for others
            cv2.fillConvexPoly(mask, np.array(contour["forehead"]), 1)

            if USE_CHEEK:
                cv2.fillConvexPoly(mask, np.array(contour["cheek1"]), 1)
                cv2.fillConvexPoly(mask, np.array(contour["cheek2"]), 1)

            # 0 value will be converted to False
            mask = mask.astype(bool)

            # init 3D array shaped-like frame with all 0 values
            roi = np.zeros_like(frame)

            # put the mask on returning other point in black frame
            roi[mask] = frame[mask]

            return roi, pos
    else:
        return None, None


def calculate_mean_from_roi(roi):
    if USE_HSV:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        HSVChannels = roi.reshape(-1, roi.shape[-1])
        HSVChannels = HSVChannels.astype(float)

        HSVChannels = HSVChannels[:, 0] / 180
        HSVChannels = HSVChannels[HSVChannels > 0]
        HSVChannels = HSVChannels[HSVChannels < 0.05]
        return np.mean(HSVChannels)
    else:
        roi = roi.astype(float)
        greenChannel = roi[:, :, 1]
        greenChannel[greenChannel == False] = np.nan
        meanGreen = np.nanmean(greenChannel)
        return meanGreen


def calculate_hr(raw):
    selfFft = np.abs(raw)
    selfFreqs = float(FPS) / WINDOW_SIZE * np.arange(WINDOW_SIZE / 2 + 1)

    freqs = 60 * selfFreqs
    valid_idx = np.where((freqs > 50) & (freqs < 180))

    valid_fft = selfFft[valid_idx]

    freqs = freqs[valid_idx]

    maxIdx = np.argmax(valid_fft)

    bpm = freqs[maxIdx]
    return bpm


