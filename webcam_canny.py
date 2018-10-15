import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
    # weighted = cv2.addWeighted( frame, 0.5, mask, 0.5, gamma=1)
    gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # _, otsu = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    # sobelx = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    # sobely = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)

    # cv2.imshow("lap", laplacian)
    # cv2.imshow("sobelx", sobelx)
    # cv2.imshow("sobely", sobely)
    edges = cv2.Canny(frame, 150, 200)
    # cv2.imshow("edges", edges)
    cv2.imshow("gaus", edges)


    # lower_red = np.array([150,150,0])
    # upper_red = np.array([180,255,255])

    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame, frame, mask = mask)

    # median = cv2.medianBlur(res, 15)

    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    # cv2.imshow('median', median)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
