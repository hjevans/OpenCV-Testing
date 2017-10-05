import cv2
cap = cv2.VideoCapture(0)

while True:
    ret = False
    while ret == False:
        ret, frame = cap.read()
        mirror = cv2.flip(frame,flipCode=1)
    #cv2.imshow('frame',frame)
    cv2.imshow('frame',mirror)
    #if cv2.waitkey(1) & 0xFF == ord('q'):
        #break
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

# To exit, type ctrl+C into the terminal
