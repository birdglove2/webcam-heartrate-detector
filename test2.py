import cv2
import mediapipe as mp
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fftpack import rfft
from getHR import getHeartRate

# from matplotlib.pyplot import plot, ion, show
from global_vars import FOREHEAD_POINTS, WINDOW_SIZE, FPS, FREQS
import time

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
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        return False, None, None

    # To improve performance, optionally mark
    # the frame as not writeable to pass by reference.
    frame.flags.writeable = False

    face = mp_face_mesh.process(frame)
    return True, frame, face


# find FOREHEAD_POINTS position in the frame
# mask the points by 1, other by False value
# return the masked frame or ROI
def extract_roi(frame, face):
    contour = {"forehead": []}
    height, width, _ = frame.shape

    if face.multi_face_landmarks:
        for face_landmarks in face.multi_face_landmarks:
            for i in FOREHEAD_POINTS:
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                contour["forehead"].append([x, y])

            # draw contour
            # cv2.drawContours(frame, [np.array(contour["forehead"])], 0, (0, 0, 255), 2)
            pos = (contour["forehead"][0][0] - 200, contour["forehead"][0][1] - 20)

            # getting 2D array with same width and height
            mask = np.zeros((height, width))

            # fill 1 only the forehead, 0 for others
            cv2.fillConvexPoly(mask, np.array(contour["forehead"]), 1)

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
    roi = roi.astype(float)
    greenChannel = roi[:, :, 1]
    greenChannel[greenChannel == False] = np.nan
    meanGreen = np.nanmean(greenChannel)
    return meanGreen


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
thickness = 3
lineType = 2


def main():
    cap = cv2.VideoCapture(0)
    # mean_list = []
    # heart_rates = []
    # text = f"Calculating HR..."
    # text2 = "HR: ... bpm"
    # inds = np.where((FREQS < 0.8) | (FREQS > 2.4))
    # l = 0
    # times = []

    # used to record the time at which we processed current frame
    start = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return False, None, None
        # get frame and face
        # success, frame, face = detect_face(cap)
        # if not success:
        # break

        # # # extract roi
        # roi, pos = extract_roi(frame, face)
        # if roi is None:
        #     print("cannot detect face")
        #     continue

        # # # # calculate mean of green channel from roi
        # meanGreen = calculate_mean_from_roi(roi)

        # # # wait until mean_list >= WINDOW_SIZE
        # mean_list.append(meanGreen)
        # times.append(time.time() - start)

        # gap = (WINDOW_SIZE - len(mean_list)) / FPS
        # # if len(mean_list) < WINDOW_SIZE:
        # # print(len(mean_list))
        # text = "Calculating HR, wait {:.1f} s".format(gap)
        # # print(len(mean_list),text)
        # if (len(mean_list) >= WINDOW_SIZE) and (len(mean_list) % FPS == 0):
        #     end = time.time()
        #     times = times[-WINDOW_SIZE:]
        #     print("time: ", end - start)
        #     start = end
        #     windowStart = len(mean_list) - WINDOW_SIZE
        #     window = mean_list[windowStart : windowStart + WINDOW_SIZE]

        #     even_times = np.linspace(times[0], times[-1], WINDOW_SIZE)
        #     print(len(even_times), len(times), len(window))
        #     interpolated = np.interp(even_times, times, np.array(window))
        #     interpolated = np.hamming(WINDOW_SIZE) * interpolated
        #     interpolated = interpolated - np.mean(interpolated)

        #     raw = rfft(interpolated)

        #     phase = np.angle(raw)
        #     selfFft = np.abs(raw)
        #     selfFreqs = float(FPS) / WINDOW_SIZE * np.arange(WINDOW_SIZE / 2 + 1)

        #     freqs = 60 * selfFreqs
        #     idx = np.where((freqs > 50) & (freqs < 180))

        #     pruned = selfFft[idx]
        #     phase = phase[idx]

        #     pfreq = freqs[idx]
        #     selfFreqs = pfreq
        #     selfFft = pruned
        #     idx2 = np.argmax(pruned)

        #     bpm = selfFreqs[idx2]

        #     # # # calculate HR
        #     # hr2 = getHeartRate(window)
        #     # hr2 = 0

        #     text2 = f"HR: {bpm}  bpm"
        #     # print("HR=",bpm , "not inter= ", hr2)

        # cv2.putText(frame, text, pos, font, fontScale, fontColor, thickness, lineType)
        # (x, y) = pos
        # cv2.putText(
        #     frame, text2, (x + 500, y), font, fontScale, fontColor, thickness, lineType
        # )

        cv2.imshow("HR Detector", frame)
        # cv2.imshow("ROI", cv2.flip(green_channel, 1))

        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    cap.release()

    cv2.destroyAllWindows()


main()
