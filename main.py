import cv2
import numpy as np
import time
import math
from collections import defaultdict
# pip install git+https://github.com/nelsond/gridfit
'''
from ros2 import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#'''
#find the dots in the frame
def find_dots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 1)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(gray, 255, adapt_type, thresh_type, 21, 7)
    rho, theta, thresh = 2, np.pi/180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    lcps = []
    cps = []
    rcps=[]
    if (type(lines)==type(None)):
        cps = []
        return cps, gray, rcps
    segmented = segment_by_angle_kmeans(lines)
    if (len(lines)<=1):
        cps = []
        return cps, gray, rcps
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
    #
    return cps, frame, rcps
#find the intersection between two lines
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

#segment the lines by angle
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
#segment the lines by angle
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
#find the closest point to a point
def compare(poi,lpoi, minpnt,maxpnt,maxdis):
    dists =[]
    dxs =[]
    dys =[]
    #find the closest point to each point in the current frame
    if (len(poi)>minpnt and len(lpoi)>minpnt):
        for point in poi:
            no,dis,dx,dy  = closest_node(point,lpoi,maxpnt)
            if (dis < maxdis):
                dists.append(dis)
                dxs.append(dx)
                dys.append(dy)
                
    #return the mean of the distances scaled and rotated
    if (dists!=[]and dxs!=[]and  dys!=[]):
        dists=np.sort(dists)
        dxs=np.sort(dxs)
        dys=np.sort(dys)
        mean=(dists[np.intp(len(dists)/2)])
        meanx=(dxs[np.intp(len(dxs)/2)])
        meany=(dys[np.intp(len(dys)/2)])
        
    else:
        mean=0
        meanx=0
        meany=0

    
    return mean,meanx,meany

#calculate rotation and scale
def unScRo(lcps):
    lcps=lcps[lcps[:, 2].argsort()]
    sca,ang=lcps[np.intp(len(lcps))][2],lcps[np.intp(len(lcps))][3]
    return sca,ang

#perfornm a rotation on a set of points
def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

#perform a scaling on a set of points
def scaling(sca, cps, center_point):
    scaled_pts = []
    for p in cps:
        tr_pointx, tr_pointy = p[0]-center_point[0], p[1]-center_point[1]
        sc_pointx, sc_pointy = tr_pointx * sca, tr_pointy * sca
        scaled_pt = [sc_pointx + center_point[0], sc_pointy + center_point[1]]
            # draw the pt  
        scaled_pts.append(scaled_pt)
    return scaled_pts

#find the closest node to a point
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

#calculate the altitude using median vertical spacing of the dots
def calcAlt(poi,margin,refconvratio):
    poi = np.array(poi)
    poi=poi[poi[:, 1].argsort()]
    dists = []
    for i in range(1,len(poi)):
        dis = poi[i][1]-poi[i-1][1]
        if (abs(dis)>margin):
            dists.append(dis)
    dists = np.array(dists)
    median = np.median(dists)
    alt = median*refconvratio
    return alt



#draw the lines between the points
def compareD(poi,lpoi, minpnt,maxpnt,maxdis,frame):
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
    return frame
#calculate relative scale and rotation between frames by comparing previous and current calculated rotation and scale accounting for loop closure
def calcRel(lsca,lang,sca,ang):
    
    #calculate relative scale
    
    sca=sca/lsca
    #calculate relative rotation
    if (ang-lang>180):
        ang=ang-360
    ang=ang-lang
    return sca,ang
'''
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,#std_msgs.msg.String,
            '/camera/image_raw',#'/chatter',
            self.listener_callback,#self.chatter_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()#CvBridge()
        self.prev_frame = None  #previous frame
        self.prev_dots = None   #previous dots
        self.prev_rdots = None  #previous dots rotated and scaled
        self.prev_angle = 0     #previous angle
        self.prev_scale = 1     #previous scale
        self.accu_scale = 1     #accumulated scale
        self.accu_angle = 0     #accumulated angle
        self.dx1 = 0            #displacement x
        self.dy1 = 0            #displacement y
        self.maxdis = 50        #maximum distance between points
        self.margin = 10        #margin for calculating altitude
        self.refconvratio = 0.1 #conversion ratio for altitude
        self.alt = 0            #altitude
        self.refalt = 0         #reference altitude
    # the loop
    def listener_callback(self, msg):
        #convert the image to a frame
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        
        #prevent frame from being modified
        disframe = frame.copy()
        #find dots in the frame
        poi1, disframe, rpoi1 = find_dots(disframe)
        #calculate relative scale and rotation
        sca,ang=unScRo(poi1)
        resc, reang = calcRel(self.prev_scale,self.prev_angle,sca,ang)
        #update the previous scale and rotation
        self.prev_angle = ang
        self.prev_scale = sca
        #update the accumulated scale and rotation
        self.accu_angle = self.accu_angle + reang
        self.accu_scale = self.accu_scale * resc
        #scale and rotate the current points
        spoi1 = scaling(self.accu_scale, poi1, [np.intp(len(poi1)/2),np.intp(len(poi1)/2)])
        rpoi1 = rotate(spoi1, [np.intp(len(poi1)/2),np.intp(len(poi1)/2)], self.accu_angle)
        
        #check if the previous frame is not None
        if self.prev_frame is not None:
            #calculate the altitude
            self.alt = self.refalt * self.accu_scale
            #compare the current points with the previous points
            mean1, meanx1, meany1 = compare(rpoi1, self.prev_rdots, 7, 100, self.maxdis)
            #draw the lines between the points

            #update the displacement
            self.dx1 = self.dx1 + meany1
            self.dy1 = self.dy1 + meanx1
            #display the results
            disframe = compareD(poi1, self.prev_dots, 7, 100, self.maxdis, disframe)
            cv2.putText(disframe, "speed:" + str(mean1), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(disframe, "linear x speed:" + str(meanx1), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(disframe, "linear  y speed:" + str(meany1), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(disframe, "linear x displacement:" + str(self.dy1), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(disframe, "linear y displacement:" + str(self.dx1), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            #calculate initial altitude
            self.refalt = calcAlt(poi1,self.margin,self.refconvratio)
        #update the previous frame and points
        self.prev_frame = frame
        self.prev_dots = poi1
        self.prev_rdots = rpoi1
        #self.get_logger().info('I heard: "%s"' % msg.data)
        #cv2.imshow("frame1", disframe)
        cv2.waitKey(1)
    # send result to main node
    def send_result(self):
        return self.dx1, self.dy1, self.alt, self.accu_angle
#'''    
#test the code with webcam
class webcam:
    def __init__(self):
        self.video = cv2.VideoCapture("http://localhost:8001/?action=stream")
        self.prev_frame = None
        self.prev_dots = None
        self.prev_rdots = None
        self.prev_angle = 0
        self.prev_scale = 1
        self.accu_scale = 1
        self.accu_angle = 0
        self.dx1 = 0
        self.dy1 = 0
        self.maxdis = 50
        self.margin = 10
        self.refconvratio = 0.1
        self.alt = 0
    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            #prevent frame from being modified
            disframe = frame.copy()
            #find dots in the frame
            poi1, disframe, rpoi1 = find_dots(disframe)
            #calculate relative scale and rotation
            sca,ang=unScRo(poi1)
            resc, reang = calcRel(self.prev_scale,self.prev_angle,sca,ang)
            #update the previous scale and rotation
            self.prev_angle = ang
            self.prev_scale = sca
            #update the accumulated scale and rotation
            self.accu_angle = self.accu_angle + reang
            self.accu_scale = self.accu_scale * resc
            #scale and rotate the current points
            spoi1 = scaling(self.accu_scale, poi1, [np.intp(len(poi1)/2),np.intp(len(poi1)/2)])
            rpoi1 = rotate(spoi1, [np.intp(len(poi1)/2),np.intp(len(poi1)/2)], self.accu_angle)
            #calculate altitude
            if self.prev_frame is not None:
                #calculate the altitude
                self.alt = self.alt * self.accu_scale
                #compare the current points with the previous points
                mean1, meanx1, meany1 = compare(rpoi1, self.prev_rdots, 7, 100, self.maxdis)
                #draw the lines between the points
                disframe = compareD(poi1, self.prev_dots, 7, 100, self.maxdis, disframe)
                #update the displacement
                self.dx1 = self.dx1 + meany1
                self.dy1 = self.dy1 + meanx1
                #draw the results
                cv2.putText(disframe, "speed:" + str(mean1), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disframe, "linear x speed:" + str(meanx1), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disframe, "linear  y speed:" + str(meany1), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disframe, "linear x displacement:" + str(self.dy1), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disframe, "linear y displacement:" + str(self.dx1), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                #calculate initial altitude
                self.alt = calcAlt(poi1,self.margin,self.refconvratio)
            #update the previous frame and points
            self.prev_frame = frame
            self.prev_dots = poi1
            self.prev_rdots = rpoi1
            #display the results
            cv2.imshow("frame1", disframe)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("exiting...")
                break
        
        self.video.release()
        cv2.destroyAllWindows()
    def send_result(self):
        print(self.dx1, self.dy1, self.accu_scale, self.accu_angle, self.alt)
        return self.dx1, self.dy1, self.accu_scale, self.accu_angle, self.alt
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

#'''
#test the code with webcam
webcam().run()
#'''

        
