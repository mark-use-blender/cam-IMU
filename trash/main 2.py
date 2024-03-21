import cv2
import numpy as np
import time

def closest_node(node, nodes,max):
    dist = max
    pnt = []
    if (len(nodes)<200):
        for no in nodes:
            dis =  ((no[0] - node[0]) ** 2 + (no[1] - node[1]) ** 2) ** 0.5
            if (dis<dist):
                dist=dis
                pnt = no
        if (dist == max):
            dist = 1000
            pnt = None
    else:
        dist = 1000
        pnt = None

            

    return (pnt,dist)
#'''
def find_dots2(frame):
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(frame, 100, 255, 0)    
    thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY,3,3)   
    Cont, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cps = []
    try:
        for cont in Cont:
            
            ((x, y), (width, height), angle) = cv2.minAreaRect(cont)
            if (angle < 14):
                width, height= height,width
            asp=height/width
            #print (asp)
            if (2<asp<3):
                cps.append((x,y))
                cv2.circle(frame,(np.intp(x),np.intp(y)), 5, (0,0,0), -1)
                #print(x,y)

            #cv2.drawContours(thresh,[box],0,(0,255,255),2)
            #print (box)
    except:
        pass
    return cps, frame
#'''
from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)
    if (type(lines)=='NoneType'):
        pass
    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        x0, y0 = np.linalg.solve(A, b)
    except:
        return []
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    if(len(lines[0])<100 and len(lines[1])<100 ):
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        it = intersection(line1, line2)
                        if (it==[]):
                            continue
                        intersections.append(it) 



    return intersections
#'''
def find_dots1(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
    rho, theta, thresh = 2, np.pi/180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    if (type(lines)==type(None)):
        cps = []
        return cps, bin_img
    segmented = segment_by_angle_kmeans(lines)
    if (len(lines)<=1):
        cps = []
        return cps, bin_img
    tcps = segmented_intersections(segmented)
    tframe = frame.copy()
    tframe = tframe-tframe
    for p in tcps:
        cv2.circle(tframe,(np.intp(p[0]),np.intp(p[1])), 5, (255,255,255), -1)
    contours, _ = cv2.findContours(tframe, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cps = []
    for con in contours:
        M = cv2.moments(con)
        X = int(M["m10"] / M["m00"])
        Y = int(M["m01"] / M["m00"])
        cps.append([X,Y])
        
    cv2.drawKeypoints(frame, cps,None,(0,0,0))
    return cps, frame
#'''
def find_dots(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dest = cv2.cornerHarris(gray,2,3,0.04)
    dest = cv2.dilate(dest, None) 
    dest = cv2.dilate(dest, None) 
    dest = cv2.dilate(dest, None) 
    dest = cv2.dilate(dest, None) 
    im = gray - gray
    im[dest > 0.01 * dest.max()]=[255]
    contours, _ = cv2.findContours(im, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cps = []
    for con in contours:
        M = cv2.moments(con)
        X = int(M["m10"] / M["m00"])
        Y = int(M["m01"] / M["m00"])
        cps.append([X,Y])
        cv2.circle(frame,(np.intp(X),np.intp(Y)), 5, (0,0,0), -1)
    return cps, frame
fast = cv2.FastFeatureDetector.create()

def find_dots3(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cps = fast.detect(gray,None)
    
    
    cv2.drawKeypoints(frame, cps,None,(0,0,0))
    return cps, frame
agast = cv2.AgastFeatureDetector.create()
def find_dots4(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cps = agast.detect(gray,None)
    
    
    cv2.drawKeypoints(frame, cps,None,(0,0,0))
    return cps, frame

lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

feature_params = dict( maxCorners = 500, 
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

vid = cv2.VideoCapture(5) 
ret, frame = vid.read() 
frame0 = frame.copy()
frame1 = frame.copy()
frame2 = frame.copy()
frame3 = frame.copy()
frame4 = frame.copy()
poi0, frame0 = find_dots(frame0)
poi1, frame1 = find_dots1(frame1)
poi2, frame2 = find_dots2(frame2)
poi3, frame3 = find_dots3(frame3)
poi4, frame4 = find_dots4(frame4)
while(True): 
      
    # Capture the video frame 
    # by frame 
    dists0 =[]
    dists1 =[]
    dists2 =[]
    dists3 =[]
    dists4 =[]
    lpoi0 = poi0
    lpoi1 = poi1
    lpoi2 = poi2
    lpoi3 = poi3
    lpoi4 = poi4
    ret, frame = vid.read() 
    frame
    frame0 = frame.copy()
    frame1 = frame.copy()
    frame2 = frame.copy()
    frame3 = frame.copy()
    frame4 = frame.copy()
    poi0, frame0 = find_dots(frame0)
    poi1, frame1 = find_dots1(frame1)
    poi2, frame2 = find_dots2(frame2)
    poi3, frame3 = find_dots3(frame3)
    poi4, frame4 = find_dots4(frame4)
    if (len(poi0)>7 and len(lpoi0)>7):


        for point0 in poi0:
            no0,dis0 = closest_node(point0,lpoi0,100)
            if (dis0 < 10):
                dists0.append(dis0)
    mean0=(np.mean(dists0))

    if (len(poi1)>7 and len(lpoi1)>7):


        for point1 in poi1:
            no1,dis1 = closest_node(point1,lpoi1,100)
            if (dis1 < 10):
                dists1.append(dis1)
    mean1=(np.mean(dists1))
                
    if (len(poi2)>7 and len(lpoi2)>7):


        for point2 in poi2:
            no2,dis2 = closest_node(point2,lpoi2,100)
            if (dis2 < 10):
                dists2.append(dis2)
    mean2=(np.mean(dists2))

    if (len(poi3)>7 and len(lpoi3)>7):


        for point3 in poi3:
            no3,dis3 = closest_node(point3,lpoi3,100)
            if (dis3 < 10):
                dists3.append(dis3)
    mean3=(np.mean(dists3))

    if (len(poi4)>7 and len(lpoi4)>7):


        for point4 in poi4:
            no4,dis4 = closest_node(point4,lpoi4,100)
            if (dis4 < 10):
                dists4.append(dis4)
    mean4=(np.mean(dists4))
    #print(cont)
    
    cv2.putText(frame0,str(mean0),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame1,str(mean1),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame2,str(mean2),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame3,str(mean3),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame4,str(mean4),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)

    cv2.imshow("frame0", frame0)
    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)
    cv2.imshow("frame3", frame3)
    cv2.imshow("frame4", frame4)
    time.sleep(0.1)
    #print (cont)
    
    
    # Display the resulting frame 
    
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 