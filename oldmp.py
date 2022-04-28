
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft,fftfreq

# ------------------- Parameter -----------------------

drawMesh = False

# ------------------- Helper function -----------------------

def _createSpectrum(data, nyquistFreq):
    """
    Calculate interpolated power spectrum density from the given data.
    Return 2-tuple of freqs and spectrum arrays.
    """
    data = data * np.hanning(len(data))
    print("data",data)
    # fft = np.fft.rfft(data, n=8 * len(data))
    yf = fft(data)
    spectrum = np.abs(fft)
    freqs = np.linspace(0, nyquistFreq * 60, len(spectrum))
    idx = np.where((freqs >= 40) & (freqs <= 120))
    freqs = freqs[idx]
    spectrum = spectrum[idx]
    spectrum /= np.max(spectrum)
    spectrum **= 2
    return freqs, spectrum

def _findPeak(x, y):
    """
    Find interpolated location of the highest peak.
    """
    peak = 0
    maxBin = np.argmax(y)
    threshold = y.max() / 2
    if 0 < maxBin < len(y) - 1:
        # find bins around peak that are at least half the peak hight
        leftBin, rightBin = -1, -1
        for leftBin in range(maxBin, 0, -1):
            if y[leftBin - 1] < threshold:
                break
        for rightBin in range(maxBin, len(y) - 1):
            if y[rightBin + 1] < threshold:
                break
        # parabolic fit of peak
        if leftBin >= 0 and rightBin >= 0:
            s = np.arange(leftBin, rightBin + 1)
            a, b, c = np.polyfit(x[s], y[s], 2)
            peak = -0.5 * b / a if a != 0 else -2
            if peak < x[maxBin - 1] or peak > x[maxBin + 1]:
                # parabolic fit failed
                peak = x[maxBin]
    else:
        peak = x[maxBin]
    return peak

# ------------------- Code -----------------------

mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


FOREHEAD_POINTS = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69]

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

cap = cv2.VideoCapture(0)


# ----------------------------------------------

meanList = []
hrList = []

def moving_avg(signal, w_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, 'valid')
    return moving_avg

def get_rfft_hr(signal):
    signal_size = len(signal)
    signal = signal.flatten()
    fft_data = np.fft.rfft(signal) # FFT
    fft_data = np.abs(fft_data)

    freq = np.fft.rfftfreq(signal_size, 1./30) # Frequency data

    inds= np.where((freq < 0.8) | (freq > 1.2) )[0]
    fft_data[inds] = 0
    bps_freq=60.0*freq
    max_index = np.argmax(fft_data)
    fft_data[max_index] = fft_data[max_index]**2
    HR =  bps_freq[max_index]
    return HR

# ----------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contour = {'forehead': []}

    results = faceMesh.process(frame)

    height, width, _ = frame.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if drawMesh:
                mp_drawing.draw_landmarks(frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

            for i in FOREHEAD_POINTS:
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                contour['forehead'].append([x, y])


            mask = np.zeros((frame.shape[0], frame.shape[1]))

            cv2.fillConvexPoly(mask, np.array(contour['forehead']), 1)

            mask = mask.astype(bool)

            roi = np.zeros_like(frame)
            roi[mask] = frame[mask]

            mean_rgb = roi.reshape(-1, roi.shape[-1])
            mean_rgb = mean_rgb[~np.all(mean_rgb == 0, axis=1)]
            mean_rgb = np.mean(mean_rgb, axis=0)
        
            meanList.append(mean_rgb)
            print(meanList)
            if len(meanList) >= 270 and len(meanList) % 30 == 0:
                l = int(30*3.2)
                H = np.zeros(270)

                for t in range(0, (270 - l + 1)):
                    B = np.array(meanList)
                    C = B[t:t+l,:].T

                    mean_color = np.mean(C, axis=1)
                    diag_mean_color = np.diag(mean_color)
                    diag_mean_color_inv = np.linalg.inv(diag_mean_color)
                    Cn = np.matmul(diag_mean_color_inv,C)
                    projection_matrix = np.array([[0,1,-1],[-2,1,1]])
                    S = np.matmul(projection_matrix,Cn)
                    std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
                    P = np.matmul(std,S)
                    H[t:t+l] = H[t:t+l] +  (P-np.mean(P))

                p = moving_avg(H, 6)
                hr = get_rfft_hr(p)
                hrList.append(hr)

                hr_fft = moving_avg(hrList, 3)[-1] if len(hrList) > 5 else hrList[-1]
                print(f'\rHr: {round(hr_fft, 0)}')
            
            '''
            colorChannels = roi.reshape(-1, roi.shape[-1])
            colorChannels = colorChannels.astype(float)
            colorChannels[colorChannels == 0] = np.nan

            greenChannel = np.nanmean(colorChannels, axis=0, dtype=np.float64)[1]

            count += 1
            times.append(count)
            sgn.append(greenChannel)

            if len(times) >= 22:
              t0 = times[0]
              t1 = times[-1]
              timeSpace = np.linspace(t0, t1, len(times))
              interpolated = np.interp(timeSpace, times, sgn)

              r = [min(1, bpm / 60 / (0.5*30)) for bpm in (40, 120)]
              b, a = butter(3, r, btype='bandpass')
              filtered = filtfilt(b, a, interpolated)

              freqs, spectrum = _createSpectrum(filtered, 0.5*30)
              bpm = _findPeak(freqs, spectrum)
              bpmList.append(bpm)

              if len(bpmList) > 10:
                print(sum(bpmList[-10:-1])/10)
              else:
                print(bpm)
            '''

        cv2.imshow("ROI", roi)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
