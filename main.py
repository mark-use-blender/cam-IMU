import cv2
import numpy as np


def closest_node(node, nodes,max):
    dist = max
    pnt = []
    for no in nodes:
        dis =  ((no[0] - node[0]) ** 2 + (no[1] - node[1]) ** 2) ** 0.5
        if (dis<dist):
            dist=dis
            pnt = no
    if (dist == max):
        dist = -1
        pnt = None

            

    return (pnt,dist)
'''
def find_dots(frame):
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
                cv2.circle(frame,(np.int0(x),np.int0(y)), 5, (0,0,0), -1)
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
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    it = intersection(line1, line2)
                    intersections.append(it) 
                    cv2.circle(frame,(np.int0(it[0]),np.int0(it[1])), 5, (0,0,0), -1)


    return intersections
'''
def find_dots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
    rho, theta, thresh = 2, np.pi/180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    segmented = segment_by_angle_kmeans(lines)
    cps = segmented_intersections(segmented)

    return cps, frame
'''
def find_dots(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dest = cv2.cornerHarris(gray,2,3,0.04)
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
        cv2.circle(frame,(np.int0(X),np.int0(Y)), 5, (0,0,0), -1)
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

poi,frame = find_dots(frame)

while(True): 
      
    # Capture the video frame 
    # by frame 
    dists =[]
    lpoi = poi
    ret, frame = vid.read() 
    poi, frame = find_dots(frame)
    if (len(poi)>7 and len(lpoi)>7):


        for point in poi:
            no,dis = closest_node(point,lpoi,100)
            if (dis < 10):
                dists.append(dis)
    mean=(np.mean(dists))
            
                

    #print(cont)
    
    print("frame",mean)
    cv2.imshow("frame", frame)
    #time.sleep(1)
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