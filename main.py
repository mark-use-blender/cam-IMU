import cv2
import numpy as np
import time
from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2):    
    if (type(lines)=='NoneType'):
        pass
    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]for angle in angles], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1) 
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0[0])), int(np.round(y0[0]))
    except:
        return [0,0]
    
    return [x0, y0]


def segmented_intersections(lines):
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
def closest_node(node, nodes,max):
    dist = max
    pnt = []
    dx=0
    dy=0
    if (len(nodes)<200):
        for no in nodes:            
            ddx = (no[0] - node[0])
            ddy = (no[1] - node[1])
            dis =  (ddx ** 2 + ddy ** 2) ** 0.5
            if (dis<dist):
                dx=ddx
                dy=ddy
                dist=dis
                pnt = no
                
        if (dist == max):
            dist = 1000
            pnt = None
    else:
        dist = 1000
        pnt = None
    
            

    return (pnt,dist,dx,dy)
def closest_nodekp(node, nodes,max):
    dist = max
    pnt = []
    dx=0
    dy=0
    if (len(nodes)<200):
        for no in nodes:            
            ddx = (no.pt[0] - node.pt[0])
            ddy = (no.pt[1] - node.pt[1])
            dis =  (ddx ** 2 + ddy ** 2) ** 0.5
            if (dis<dist):
                dx=ddx
                dy=ddy
                dist=dis
                pnt = no
                
        if (dist == max):
            dist = 1000
            pnt = None
    else:
        dist = 1000
        pnt = None

            

    return (pnt,dist,dx,dy)
#'''
def find_dots0(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 1)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 21, 7)
    dest = cv2.cornerHarris(bin_img,2,3,0.04)
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

def find_dots2(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 1)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 21, 7)
    Cont, hierarchy = cv2.findContours(bin_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cps = []
    try:
        for cont in Cont:
            ((x, y), (width, height), angle) = cv2.minAreaRect(cont)
            if (angle < 14):
                width, height= height,width
            asp=height/width
            if (2<asp<3):
                cps.append((x,y))
                cv2.circle(frame,(np.intp(x),np.intp(y)), 5, (0,0,0), -1)
    except:
        pass
    return cps, frame
#'''

def find_dots1(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 1)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(gray, 255, adapt_type, thresh_type, 21, 7)
    rho, theta, thresh = 2, np.pi/180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    if (type(lines)==type(None)):
        cps = []
        return cps, gray
    segmented = segment_by_angle_kmeans(lines)
    if (len(lines)<=1):
        cps = []
        return cps, gray
    tcps = segmented_intersections(segmented)
    tframe = gray.copy()
    tframe = tframe-tframe
    for p in tcps:
        cv2.circle(tframe,(np.intp(p[0]),np.intp(p[1])), 5, (255,255,255), -1)
    contours, _ = cv2.findContours(tframe, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cps = []
    for con in contours:
        try:
            M = cv2.moments(con)
            X = int(M["m10"] / M["m00"])
            Y = int(M["m01"] / M["m00"])
            cps.append((X,Y))
            cv2.circle(frame,(np.intp(X),np.intp(Y)), 5, (0,0,0), -1)
        except:
            continue
    return cps, frame

fast = cv2.FastFeatureDetector.create(2)

def find_dots3(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 1)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 21, 7)

    cps = fast.detect(bin_img,None)
    frame=cv2.drawKeypoints(frame, cps,None,(0,0,0))
    return cps, frame

agast = cv2.AgastFeatureDetector.create()

def find_dots4(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 1)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 21, 7)
    cps = agast.detect(bin_img,None)
    cv2.drawKeypoints(bin_img, cps,None,(0,0,255))
    return cps, bin_img

def compare(poi,lpoi, minpnt,maxpnt,maxdis,frame):
    dists =[]
    dxs =[]
    dys =[]
    if (len(poi)>minpnt and len(lpoi)>minpnt):
        for point in poi:
            no,dis,dx,dy  = closest_node(point,lpoi,maxpnt)
            if (dis < maxdis):
                dists.append(dis)
                dxs.append(dx)
                dys.append(dy)
                frame=cv2.line(frame,point,no,(0,255,0),3)
    if (dists!=[]and dxs!=[]and  dys!=[]):
        mean=(np.mean(dists))
        meanx=(np.mean(dxs))
        meany=(np.mean(dys))
        
    else:
        mean=0
        meanx=0
        meany=0
    return mean,meanx,meany

def comparekp(poi,lpoi, minpnt,maxpnt,maxdis,frame):
    dists =[]
    dxs =[]
    dys =[]
    if (len(poi)>minpnt and len(lpoi)>minpnt):
        for point in poi:
            no,dis,dx,dy  = closest_nodekp(point,lpoi,maxpnt)
            if (dis < maxdis):
                dists.append(dis)
                dxs.append(dx)
                dys.append(dy)
                frame=cv2.line(frame,point.pt,no.pt,(0,255,0),3)
    mean=(np.mean(dists))
    meanx=(np.mean(dxs))
    meany=(np.mean(dys))
    return mean,meanx,meany


vod = cv2.VideoCapture(5) 
ret, frame = vod.read() 
# frame0 = frame.copy()
frame1 = frame.copy()
# frame2 = frame.copy()
# frame3 = frame.copy()
# frame4 = frame.copy()
# poi0, frame0 = find_dots0(frame0)
poi1, frame1 = find_dots1(frame1)
# poi2, frame2 = find_dots2(frame2)
# poi3, frame3 = find_dots3(frame3)
# poi4, frame4 = find_dots4(frame4)
dx1 = 0
dy1 = 0

while(True): 
      
    # Capture the video frame 
    # by frame 
    maxdis = 20

    #lpoi0 = poi0
    lpoi1 = poi1
    # lpoi2 = poi2
    # lpoi3 = poi3
    # lpoi4 = poi4
    ret, frame = vod.read() 
    
    # frame0 = frame.copy()
    frame1 = frame.copy()
    # frame2 = frame.copy()
    # frame3 = frame.copy()
    # frame4 = frame.copy()
    # poi0, frame0 = find_dots0(frame0)
    poi1, frame1 = find_dots1(frame1)
    # poi2, frame2 = find_dots2(frame2)
    # poi3, frame3 = find_dots3(frame3)
    # poi4, frame4 = find_dots4(frame4)
    
                
    # mean0, meanx0, meany0 = compare(poi0, lpoi0,7,100,maxdis)
    mean1, meanx1, meany1 = compare(poi1, lpoi1,7,100,maxdis,frame1)
    # mean2, meanx2, meany2 = compare(poi2, lpoi2,7,100,maxdis)
    # mean3, meanx3, meany3 = comparekp(poi3, lpoi3,7,100,maxdis)
    # mean4, meanx4, meany4 = comparekp(poi4, lpoi4,7,100,maxdis)

    
    #print(cont)
    
    # cv2.putText(frame0,str(mean0),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame0,str(meanx0),(0,150),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame0,str(meany0),(0,200),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)

    cv2.putText(frame1,"speed:"+str(mean1),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame1,"linear x speed:"+str(meanx1),(0,150),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame1,"linear  y speed:"+str(meany1),(0,200),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)   
    dx1= dx1+meany1
    dy1= dy1+meanx1
    cv2.putText(frame1,"linear x displacement:"+str(dy1),(0,250),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    cv2.putText(frame1,"linear y displacement:"+str(dx1),(0,300),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)




    # cv2.putText(frame2,str(mean2),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame2,str(meanx2),(0,150),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame2,str(meany2),(0,200),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)

    # cv2.putText(frame3,str(mean3),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame3,str(meanx3),(0,150),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame3,str(meany3),(0,200),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)

    # cv2.putText(frame4,str(mean4),(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame4,str(meanx4),(0,150),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)
    # cv2.putText(frame4,str(meany4),(0,200),cv2.FONT_HERSHEY_SIMPLEX ,1,(255, 0, 0) ,2,cv2.LINE_AA)


    # cv2.imshow("frame0", frame0)
    cv2.imshow("frame1", frame1)
    # cv2.imshow("frame2", frame2)
    # cv2.imshow("frame3", frame3)
    # cv2.imshow("frame4", frame4)
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