import cv2
import numpy as np

def grayScaling(frame):
    
    h, w, c = frame.shape
    if c == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        gray = frame
    
    return gray