import cv2
import mediapipe as mp
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fftpack import rfft
from getHR import getHeartRate
from sklearn.decomposition import FastICA

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
    success, frame = cap.read(cv2.IMREAD_UNCHANGED)
    if not success:
        print("Ignoring empty camera frame.")
        return False, None, None

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
    contour = {"forehead": []}
    height, width, _ = frame.shape

    if face.multi_face_landmarks:
        for face_landmarks in face.multi_face_landmarks:
            for i in FOREHEAD_POINTS:
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                contour["forehead"].append([x, y])

            # draw contour
            cv2.drawContours(frame, [np.array(contour["forehead"])], 0, (0, 0, 255), 2)
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


def calculate_mean_from_roi2(roi):
    colorChannels = roi.reshape(-1, roi.shape[-1])
    avgColor = colorChannels.mean(axis=0)
    return avgColor


def calculate_mean_from_roi(roi):
    # if USE_HSV:
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    HSVChannels = roi.reshape(-1, roi.shape[-1])
    HSVChannels = HSVChannels.astype(float)

    HSVChannels = HSVChannels[:, 0] / 180
    HSVChannels = HSVChannels[HSVChannels > 0]
    HSVChannels = HSVChannels[HSVChannels < 0.05]
    return np.mean(HSVChannels)

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


def main():
    ica = FastICA()

    cap = cv2.VideoCapture(0)
    mean_list = []
    mean_list2 = []
    heart_rates = []
    heart_rates1 = []
    heart_rates2 = []
    heart_rates3 = []
    heart_rates4 = []
    text = f"Calculating HR..."
    text1 = "HR1: ... bpm"
    text2 = "HR2: ... bpm"
    text3 = "HR3: ... bpm"
    text4 = "HR4: ... bpm"
    inds = np.where((FREQS < 0.8) | (FREQS > 2.4))
    l = 0
    times = []

    # used to record the time at which we processed current frame
    start = time.time()
    while True:

        # get frame and face
        success, frame, face = detect_face(cap)
        if not success:
            break

        # # extract roi
        roi, pos = extract_roi(frame, face)
        if roi is None:
            print("cannot detect face")
            continue

        # # # calculate mean of green channel from roi
        meanGreen = calculate_mean_from_roi(roi)

        # meanGreen2 = calculate_mean_from_roi2(roi)

        # # wait until mean_list >= WINDOW_SIZE
        mean_list.append(meanGreen)
        # mean_list2.append(meanGreen2)
        times.append(time.time() - start)

        gap = (WINDOW_SIZE - len(mean_list)) / FPS
        # if len(mean_list) < WINDOW_SIZE:
        # print(len(mean_list))
        text = "Calculating HR, wait {:.1f} s".format(gap)
        # print(len(mean_list),text)
        if (len(mean_list) >= WINDOW_SIZE) and (len(mean_list) % FPS == 0):
            times = times[-WINDOW_SIZE:]
            windowStart = len(mean_list) - WINDOW_SIZE
            window = mean_list[windowStart : windowStart + WINDOW_SIZE]

            even_times = np.linspace(times[0], times[-1], WINDOW_SIZE)

            # print(len(even_times), len(times), len(window))

            # interpolate
            interpolated = np.interp(even_times, times, np.array(window))
            interpolated = np.hamming(WINDOW_SIZE) * interpolated
            interpolated = interpolated - np.mean(interpolated)

            # normalized
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            normalized = (window - mean) / std
            print(normalized == window)

            # mva 6
            mva = moving_avg(window, 6)

            # raws
            raw1 = rfft(window)
            raw2 = rfft(interpolated)
            raw3 = rfft(normalized)
            raw4 = rfft(mva)

            # calculate hr from raw
            hr1 = calculate_hr(raw1)
            hr2 = calculate_hr(raw2)
            hr3 = calculate_hr(raw3)
            hr4 = calculate_hr(raw4)

            heart_rates1.append(hr1)
            heart_rates2.append(hr2)
            heart_rates3.append(hr3)
            heart_rates4.append(hr4)

            # heart_rates.append(bpm)

            # if len(heart_rates) > 5:
            #     last5hr = sum(heart_rates[-5:]) / 5
            #     text2 = f"HR2: {last5hr}"
            #     file_object = open("last5.txt", "a")
            #     file_object.write(str(last5hr) + "\n")

            # hr_mean_all = sum(heart_rates) / len(heart_rates)
            # file_object = open("meanAll.txt", "a")
            # file_object.write(str(hr_mean_all) + "\n")

            meanHr1 = sum(heart_rates1) / len(heart_rates1)
            meanHr2 = sum(heart_rates2) / len(heart_rates2)
            meanHr3 = sum(heart_rates3) / len(heart_rates3)
            meanHr4 = sum(heart_rates4) / len(heart_rates4)

            text1 = "HR1: {:.2f}, {:.2f}".format(hr1, meanHr1)
            text2 = "HR2 interpolate: {:.2f}, {:.2f}".format(hr2, meanHr2)
            text3 = "HR3 normalized: {:.2f}, {:.2f}".format(hr3, meanHr3)
            text4 = "HR4 mva: {:.2f}, {:.2f}".format(hr4, meanHr4)

            # file_object = open("data1.txt", "a")
            # file_object.write(str(bpm) + "\n")

        (x, y) = pos
        cv2.putText(frame, text, pos, font, fontScale, fontColor, thickness, lineType)
        cv2.putText(
            frame,
            text1,
            (x - 600, y + 50),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv2.putText(
            frame,
            text2,
            (x - 600, y + 100),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        cv2.putText(
            frame,
            text3,
            (x - 600, y + 150),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        cv2.putText(
            frame,
            text4,
            (x - 600, y + 200),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv2.imshow("HR Detector", frame)
        # cv2.imshow("ROI", cv2.flip(green_channel, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


main()
