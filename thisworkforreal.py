import mediapipe as mp
import numpy as np
import cv2
import copy


def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstructFrame(pyramid, index, levels, videoHeight, videoWidth):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


webcam = cv2.VideoCapture(0)
realWidth = 960
realHeight = 540

videoChannels = 3
videoFrameRate = 30
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 0.8
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (160 // 2 + 5, 30)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
videoGauss = []
tmpVideoGauss = []
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

calculationCount = 0
x = 0
y = 0
w = 0
h = 0

while True:
    ret, frame = webcam.read()
    face = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ).process(frame)

    if face.detections:
        for detection in face.detections:
            x = int(detection.location_data.relative_bounding_box.xmin * realWidth)
            y = int(detection.location_data.relative_bounding_box.ymin * realHeight)

            # w = int(detection.location_data.relative_bounding_box.width * realWidth)
            # h = int(detection.location_data.relative_bounding_box.height * realHeight)
            w = 50
            h = 50

        break
    else:
        continue

while True:
    ret, frame = webcam.read()
    if ret == False:
        break

    detectionFrame = frame[y : y + h, x : x + w, 0:]

    # green
    detectionFrame = detectionFrame[:, :, 1]

    # Construct Gaussian Pyramid

    if len(tmpVideoGauss) < 150:
        tmpVideoGauss.append(buildGauss(detectionFrame, levels + 1)[levels])
        videoGauss = copy.copy(tmpVideoGauss)

        for i in range(150 - len(tmpVideoGauss)):
            videoGauss.append(np.zeros_like(videoGauss[0]))
    else:
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]

    try:
        fourierTransform = np.fft.fft(videoGauss, axis=0)
    except:
        continue

    # Bandpass Filter
    fourierTransform[mask == False] = 0

    # Grab a Pulse
    if bufferIndex % bpmCalculationFrequency == 0:
        calculationCount = calculationCount + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    # Reconstruct Resulting Frame
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels, w, h)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    frame[y : y + h, x : x + w, :] = outputFrame

    c1 = (x, y)
    c2 = (x + w, y + h)
    cv2.rectangle(frame, c1, c2, boxColor, boxWeight)

    if calculationCount > bpmBufferSize:
        cv2.putText(
            frame,
            "BPM: %d" % bpmBuffer.mean(),
            bpmTextLocation,
            font,
            fontScale,
            fontColor,
            lineType,
        )
    else:
        cv2.putText(
            frame,
            "Calculating BPM...",
            loadingTextLocation,
            font,
            fontScale,
            fontColor,
            lineType,
        )

    cv2.imshow("Webcam Heart Rate Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
