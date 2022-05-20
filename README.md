# webcam-heartrate-detector

# Requirements

```
python3
```

# Installation

```
pip3 install -r requirements.txt
```

# How to Run

```
python3 main.py
```

After time passed for `duration` the estimated HR will be shown above the user's face

The estimated HR also be written onto the file `output.txt`

# Config

At the `helper.py` there are 6 configs you can adjust

- `USE_HSV`: set to `True` to use HSV otherwise RGB
- `USE_WEBCAM`: set to `True` if user want to use webcam, `False` for using video
- `USE_CHEEK`: set to `True` to include cheek in the ROI selected area.
- `VIDEO_FILENAME`: filepath for testing video, this will be used only if `USE_WEBCAM` set to `False`
- `FPS`: the fps of your video
- `DURATION`: the amount of time using to calculate the HR, the more it is, the more accurate it will be.
