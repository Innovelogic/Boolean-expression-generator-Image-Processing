import cv2
import numpy as np

class LinesRecognition :

    @staticmethod
    def grab_lines():

        img = cv2.imread("test2.jpg")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        thresh1 = cv2.bitwise_not(thresh1)

        edges = cv2.Canny(thresh1, threshold1=50, threshold2=200, apertureSize=3)

        lines = cv2.HoughLinesP(thresh1, rho=1, theta=np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=30)

        return lines