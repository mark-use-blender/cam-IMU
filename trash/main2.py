import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
vid = cv.VideoCapture(5) 
ret, frame = vid.read() 
# Initiate ORB detector
orb = cv.ORB_create()


while (True):
    lframe = frame
    ret, frame = vid.read()
 
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(lframe,None)
    kp2, des2 = orb.detectAndCompute(frame,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 10 matches.
    img3 = cv.drawMatches(lframe,kp1,frame,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv.imshow("ing",img3)
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 