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

PRE_STEP_ASF = False
PRE_STEP_CDF = False


def get_pulse(mean_rgb):
    seg_t = 3.2
    l = int(FPS * seg_t)
    H = np.zeros(WINDOW_SIZE)

    B = [int(0.8 // (FPS / l)), int(4 // (FPS / l))]

    for t in range(0, (WINDOW_SIZE - l + 1)):
        # pre processing steps
        C = mean_rgb[t : t + l, :].T

        # if PRE_STEP_CDF:
        #     C = CDF(C, B)

        # if PRE_STEP_ASF:
        #     C = ASF(C)

        # post processing steps
        # Signal processing techniques
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv, C)
        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        S = np.matmul(projection_matrix, Cn)
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S)
        H[t : t + l] = H[t : t + l] + (P - np.mean(P))
    return H


def moving_avg(signal, w_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, "valid")
    return moving_avg


# To estimate heart rate we compute power spectral density PSD
# applying fast fourier transformation (FFT) on rPPG signal
def get_rfft_hr(signal):
    signal = signal.flatten()
    fft_data = np.fft.rfft(signal)  # FFT
    fft_data = np.abs(fft_data)

    freq = np.fft.rfftfreq(WINDOW_SIZE, 1.0 / FPS)  # Frequency data
    # validIdx = np.where((freqs >= 50 / 60) & (freqs <= 180 / 60))

    inds = np.where((freq < 50) | (freq > 180))[0]
    print(len(signal), len(fft_data), signal, fft_data)
    fft_data[inds] = 0

    # then band pass filtered to analyse only frequencies of interest.
    # The maximum power spectrum represents the frequency of instant heart rate.
    bps_freq = 60.0 * freq
    max_index = np.argmax(fft_data)
    fft_data[max_index] = fft_data[max_index] ** 2
    HR = bps_freq[max_index]
    return HR


def main():
    ica = FastICA()

    cap = cv2.VideoCapture(0)
    mean_list = []
    mean_list2 = []
    heart_rates = []
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

        meanGreen = calculate_mean_from_roi(roi)
        mean_list.append(meanGreen)

        # if len(mean_list) < WINDOW_SIZE:
        print(len(mean_list))

        if (len(mean_list) >= WINDOW_SIZE) and (len(mean_list) % FPS == 0):
            p = get_pulse(mean_list)
            print("p1", p)
            # moving average filter of order 6 to remove outliers from signal
            p = moving_avg(p, 3)
            hr = get_rfft_hr(p)
            print("hr=", hr)

        cv2.imshow("HR Detector", frame)
        # cv2.imshow("ROI", cv2.flip(green_channel, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


main()
