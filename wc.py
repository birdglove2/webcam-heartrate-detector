import cv2 


cap = cv2.VideoCapture(1)
arr = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    arr.append(frame)
    print(len(arr))
    
    # gap = (300 - len(arr)) / 30

    # text = 'Calculating HR, wait {:.1f} s'.format(10-gap)
    # print(text)
    
