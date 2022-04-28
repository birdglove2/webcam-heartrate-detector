import cv2
import mediapipe as mp
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fftpack import rfft

# from getHR import getHeartRate
from matplotlib.pyplot import plot, ion, show

MIN_HR_BPM, MAX_HR_BMP = 45, 240
WINDOW_SIZE = 10
FPS = 30
SEC_PER_MIN = 60


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


FOREHEAD_POINTS = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69]

cap = cv2.VideoCapture(0)

SAMPLE_RATE = 1
DURATION = 10
N = SAMPLE_RATE * DURATION

xf = fftfreq(N, 1 / SAMPLE_RATE)
bps_freq = 30 * xf


mean_list = []
heartRates = []

while True:
    success, image = cap.read(cv2.IMREAD_UNCHANGED)
    if not success:
        print("Ignoring empty camera frame.")
        break

    # To improve performance, optionally mark
    # the image as not writeable to pass by reference.
    image.flags.writeable = False

    contour = {"forehead": []}

    results = mp_face_mesh.process(image)
    height, width, _ = image.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in FOREHEAD_POINTS:
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                contour["forehead"].append([x, y])

            mask = np.zeros((image.shape[0], image.shape[1]))
            cv2.fillConvexPoly(mask, np.array(contour["forehead"]), 1)
            # print("contour", contour["forehead"])

            # focus only forehead, set others to black color
            mask = mask.astype(bool)
            roi = np.zeros_like(image)
            roi[mask] = image[mask]

            # print('mask', mask.shape)
            # print('roi 0 ', roi.shape)

            roi = roi.astype(float)
            roi[roi == False] = np.nan
            colorChannels = roi.reshape(-1, roi.shape[-1])
            # print(colorChannels.shape)
            # colorChannels[colorChannels == False] = np.nan
            means = np.nanmean(colorChannels, axis=0)

            # meanGreen = means[1]
            meanGreen = means

            # # normalized
            # norm = np.linalg.norm(green_channel)
            # print('norm',norm)
            # norm_mean_green = green_channel/norm
            # # print("green channel",green_channel)
            # # print("norm green chanel",norm_mean_green)

            # print("mean before norm green chan",mean_green)

            # mean_list.append(np.mean(GreenChannels))
            mean_list.append(meanGreen)

            # print(mean_list)

            # print(len(mean_list))

            if (len(mean_list) >= WINDOW_SIZE) and (len(mean_list) % np.ceil(FPS) == 0):
                # print(mean_list)
                with open("output.txt", "w") as txt_file:
                    for line in mean_list:
                        txt_file.write(
                            " ".join(str(line)) + "\n"
                        )  # works with any number of elements in a line
                yf = rfft(mean_list)

                #     print(yf)

                inds = np.where((xf < 0.8) | (xf > 2.4))
                yf[inds] = 0
                print("yf", yf)

                max_index = np.argmax(yf)
                # print(max_index)
                yf[max_index] = yf[max_index] ** 2
                # print(bps_freq)
                HR = bps_freq[max_index]
                print("HR= ", HR)

                # windowStart = len(mean_list) - WINDOW_SIZE
                # window = mean_list[windowStart : windowStart + WINDOW_SIZE]
                # lastHR = heartRates[-1] if len(heartRates) > 0 else None

                # hr = getHeartRate(window)
                # print("hr =", hr)
                # heartRates.append(hr)
                # print(heartRates)

            # if len(mean_list) == N:

            #     yf = rfft(mean_list)

            #     print(yf)

            #     inds= np.where((xf < 0.8) | (xf > 2.4))
            #     yf[inds] = 0
            #     max_index = np.argmax(yf)
            #     print(max_index)
            #     # yf[max_index] = yf[max_index]**2
            #     # print(bps_freq)
            #     HR = bps_freq[max_index]
            #     print("HR=",HR)

            #     plot(xf,np.abs(yf))
            #     mean_list = mean_list[450:]
        # green_img = np.zeros(image.shape)
        # green_img[:,:,1] = green_channel

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Face Mesh", roi)
        # cv2.imshow("ROI", cv2.flip(green_channel,1))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
