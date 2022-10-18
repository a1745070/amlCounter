# Applied Machine Learning
# Archie Verma(a174070)

import numpy as np
import cv2 as cv
import Number
import time


# people.txt stores the data of number of people walking in a frame
try:
    log = open('people.txt',"w")

# if unable to open file print can't open file
except:
    print("Can't open files")

Capture = cv.VideoCapture('video.avi')

# setting up an empty counter for input and output(people going north and south within the frame) 
north = 0
south = 0


# Printing capture properties in console
for a in range(19):
    print(a, Capture.get(a))

h = 480
w = 640
frameArea = h*w
areaTH = frameArea/250
print( 'Area Threshold', areaTH)

line_up = int(2*(h/5))
line_down   = int(3*(h/5))

up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))

print( "Red line y:",str(line_down))
print( "Blue line y:", str(line_up))
line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))


fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)


kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while(Capture.isOpened()):
    ret, frame = Capture.read()

    for a in persons:
        a.age_one() 
    
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    try:
        ret,imBin= cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
        ret,imBin2 = cv.threshold(fgmask2,200,255,cv.THRESH_BINARY)
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        mask2 = cv.morphologyEx(imBin2, cv.MORPH_OPEN, kernelOp)
        mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernelCl)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print( 'UP:',north)
        print ('DOWN:',south)
        break
 
    contours0, hierarchy = cv.findContours(mask2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:
        
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for a in persons:
                    if abs(x-a.getX()) <= w and abs(y-a.getY()) <= h:
                        new = False
                        a.updateCoords(cx,cy)  
                        if a.going_UP(line_down,line_up) == True:
                            north += 1;
                            print( "ID:",a.getId(),' walking north at ',time.strftime("%c"))
                            log.write("ID: "+str(a.getId())+' walking north at ' + time.strftime("%c") + '\n')
                        elif a.going_DOWN(line_down,line_up) == True:
                            south += 1;
                            print( "ID:",a.getId(),' walking south at ',time.strftime("%c"))
                            log.write("ID: " + str(a.getId()) + ' walking south at ' + time.strftime("%c") + '\n')
                        break
                    if a.getState() == '1':
                        if a.getDir() == 'down' and a.getY() > down_limit:
                            a.setDone()
                        elif a.getDir() == 'up' and a.getY() < up_limit:
                            a.setDone()
                    if a.timedOut():
                        index = persons.index(a)
                        persons.pop(index)
                        del a    
                if new == True:
                    p = Number.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     
         
            cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
           
    for a in persons:

        cv.putText(frame, str(a.getId()),(a.getX(),a.getY()),font,0.3,a.getRGB(),1,cv.LINE_AA)
    str_up = 'UP: '+ str(north)
    str_down = 'DOWN: '+ str(south)
    frame = cv.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv.LINE_AA)
    cv.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv.LINE_AA)

    cv.imshow('Frame',frame)
    cv.imshow('Mask',mask)    
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
log.flush()
log.close()
Capture.release()
cv.destroyAllWindows()
