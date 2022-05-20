from lib.fvs import filevideostream as fvs
import cv2
import numpy as np
from scipy.fftpack import rfft
from utils import writeToFile, deleteDataFromFile
from helper import (
    USE_WEBCAM,
    WINDOW_SIZE,
    VIDEO_FILENAME,
    FPS,
    calculate_mean_from_roi,
    detect_face,
    calculate_hr2,
    extract_roi,
    calculate_hr,
    moving_avg,
)


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
thickness = 3
lineType = 2


def main():
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
    else:
        cap = fvs.FileVideoStream(VIDEO_FILENAME).start()

    count = 0
    mean_list = []
    heart_rates = []
    text = ""
    text1 = ""
    text2 = ""
    text3 = ""

    deleteDataFromFile("output.txt")
    writeToFile("output.txt", "second, prediction")

    while True:
        # get frame and face
        success, frame, face = detect_face(cap)
        if not success:
            break

        # extract roi
        roi, pos = extract_roi(frame, face)
        if roi is None:
            print("cannot detect face")
            continue

        # calculate mean of green channel from roi
        meanGreen = calculate_mean_from_roi(roi)

        # wait until mean_list >= WINDOW_SIZE
        mean_list.append(meanGreen)
        gap = (WINDOW_SIZE - len(mean_list)) / FPS
        if gap > 0:
            text = "Calculating HR, wait {:.1f} s".format(gap)
        else:
            text = ""

        if (len(mean_list) >= WINDOW_SIZE) and (len(mean_list) % FPS == 0):
            count += 1

            # get array of WINDOW_SIZE
            windowStart = len(mean_list) - WINDOW_SIZE
            window = mean_list[windowStart : windowStart + WINDOW_SIZE]

            # mva 5
            mva = moving_avg(window, 5)

            # raws
            raw = rfft(mva)

            # calculate hr from raw
            hr = calculate_hr(raw)

            heart_rates.append(hr)

            hr = sum(heart_rates) / len(heart_rates)
            text = ""
            text2 = "HR: {:.2f} bpm".format(hr)

            writeToFile(
                "output.txt",
                "{}, {:.2f}".format(
                    count,
                    hr,
                ),
            )

            if count == 60:
                break

        (x, y) = pos
        cv2.putText(frame, text, pos, font, fontScale, fontColor, thickness, lineType)
        cv2.putText(
            frame,
            text1,
            (x, y),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        cv2.putText(
            frame,
            text2,
            (x, y - 50),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        cv2.imshow("HR Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if USE_WEBCAM:
        cap.release()

    cv2.destroyAllWindows()


main()
