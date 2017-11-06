import cv2
import numpy as np

lowerBound = np.array([33,80,40])
upperBound = np.array([102,255,255])

cap = cv2.VideoCapture(0)
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
while True:
    #ret = False
    #while ret == False:
    ret, frame = cap.read()
    mirror = cv2.flip(frame,flipCode=1)
    canny = cv2.Canny(frame,100,200)
    #hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if ret:
        hsv_img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #cv2.imshow('frame',frame)
        #cv2.imshow('hsv',hsv_img)
        mask = cv2.inRange(hsv_img,lowerBound,upperBound)
        cv2.imshow('mask',mask)
        kernelOpen = np.ones((5,5))
        kernelClose = np.ones((20,20))
        maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
        cv2.imshow("maskClose",maskClose)
        cv2.imshow("maskOpen",maskOpen)
        maskFinal=maskClose
        conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame,conts,-1,(255,0,0),3)
        for i in range(len(conts)):
            x,y,w,h=cv2.boundingRect(conts[i])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 2)
            cv2.cv.PutText(cv2.cv.fromarray(frame), str(i+1),(x,y+h),font,(0,255,255))
        cv2.imshow('frame',frame)
#if cv2.waitkey(1) & 0xFF == ord('q'):
    #break
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

# To exit, type ctrl+C into the terminal


'''
cap = cv2.VideoCapture(0)

#ont = cv2.cv.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX,2,.5,0,3,1)

ret,img = cap.read()

#img=cv2.resize(img,(340,220))
#imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask = cv2.inRange(imgHSV,lowerBound,upperBound)

cv2.imshow('mask',mask)
cv2.imshow('cap',img)
cv2.waitKey(10)

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelClose)

cv2.imshow('maskClose',maskClose)
cv2.imshow('maskOpen',maskOpen)
cv2.waitKey(10)
'''
