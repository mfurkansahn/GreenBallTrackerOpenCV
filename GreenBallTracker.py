import cv2
import numpy as np
from abc import ABC, abstractmethod
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class tennis_ball_detect(ABC):
    @abstractmethod
    def getNumberOfBall(self):
        pass

    def getCoordinates(self):
        pass


class region_number(tennis_ball_detect):
    def __init__(self, nob=None, c=None, co=None):
        self.__numberOfBall = nob
        self.__centroids = c
        self.__coordinates = co

    def getNumberOfBall(self):
        return self.__numberOfBall

    def setNumberOfBall(self, nob):
        self.__numberOfBall = nob

    def centroid(self):
        return self.__centroids

    def setCentroids(self, c):
        self.__centroids = c

    def getCoordinates(self):
        return self.__coordinates

    def setCoordinates(self, c):
        self.__coordinates = c


class general_control(tennis_ball_detect):
    def __init__(self, c=None):
        self.__centroids = c

    def getNumberOfBall(self):
        return len(self.__centroids)

    def centroid(self):
        return self.__centroids


def track(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = (29, 86, 6)
    upper = (64, 255, 255)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]

    blank_image = np.zeros((height+1, 900, 3), np.uint8)
    blank_image[:, :] = (255, 255, 255)

    l_img = blank_image.copy()

    x_offset = y_offset = 1
    l_img[y_offset:y_offset+height, x_offset:x_offset+width] = frame.copy()

    identifiedElementsPixels = []
    identifiedElementsCoordinates = []
    identifiedElementsCentroids = []
    counter = 0
    center = (-1, -1)
    for contour in contours:
        xCoor, yCoor, wCoor, hCoor = cv2.boundingRect(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:
            counter = counter+1
            cv2.circle(l_img, center, radius, (0, 0, 255),
                       2, cv2.LINE_AA)  # tennis ball outline
            cv2.putText(l_img, str(counter), (center), font,
                        0.75, (0, 0, 0), 2, cv2.LINE_4)

            crop_img = l_img[yCoor:yCoor+hCoor, xCoor:xCoor+wCoor]
            croppedBlur = cv2.GaussianBlur(crop_img, (5, 5), 0)
            croppedHsv = cv2.cvtColor(croppedBlur, cv2.COLOR_BGR2HSV)
            whiteblackmask = cv2.inRange(croppedHsv, lower, upper)
            whiteblackmask = cv2.erode(whiteblackmask, None, iterations=3)
            whiteblackmask = cv2.dilate(whiteblackmask, None, iterations=3)
            n_white_pix = np.sum(whiteblackmask == 255)

            identifiedElementsCoordinates.append((xCoor, yCoor, wCoor, hCoor))
            identifiedElementsPixels.append(n_white_pix)
            identifiedElementsCentroids.append(center)

    rNumber = region_number(
        counter, identifiedElementsCentroids, identifiedElementsCoordinates)
    gControl = general_control(identifiedElementsCentroids)
    numberOfElem1 = "Number of identified"
    numberOfElem2 = "region: "+str(rNumber.getNumberOfBall())

    cv2.putText(l_img, numberOfElem1, (width+10, 50),
                font, 0.75, (0, 0, 0), 2, cv2.LINE_4)
    cv2.putText(l_img, numberOfElem2, (width+10, 75),
                font, 0.75, (0, 0, 0), 2, cv2.LINE_4)

    pixelCountText1 = "Number of green pixels"
    cv2.putText(l_img, pixelCountText1, (width+10, 125),
                font, 0.75, (0, 0, 0), 2, cv2.LINE_4)

    lastYCoordinateForText = 125
    pixelCounter = 0
    for nPixels in identifiedElementsPixels:
        lastYCoordinateForText = lastYCoordinateForText+25
        pixelCounter = pixelCounter+1
        cv2.putText(l_img, str(pixelCounter)+":"+str(nPixels), (width+10,
                                                                lastYCoordinateForText), font, 0.75, (0, 0, 0), 2, cv2.LINE_4)

    lastYCoordinateForText = lastYCoordinateForText+50
    centroidText1 = "Centroid Coordinate"
    cv2.putText(l_img, centroidText1, (width+10, lastYCoordinateForText),
                font, 0.75, (0, 0, 0), 2, cv2.LINE_4)

    centroidCounter = 0
    for nCentroid in gControl.centroid():
        lastYCoordinateForText = lastYCoordinateForText+25
        centroidCounter = centroidCounter+1
        (nX, nY) = nCentroid
        cv2.putText(l_img, str(centroidCounter)+":("+str(nX)+","+str(nY)+")",
                    (width+10, lastYCoordinateForText), font, 0.75, (0, 0, 0), 2, cv2.LINE_4)
    cv2.imshow('Tennis Ball Tracking', l_img)
    return center


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    while True:
        okay, image = capture.read()
        if okay:
            if not track(image):
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
           print('Capture failed')
           break