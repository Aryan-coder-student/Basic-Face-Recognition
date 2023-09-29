import cv2 as cv
from simple_facerec import SimpleFacerec
cap = cv.VideoCapture("Video Practice/video1.mp4")
# r_cap = cv.resize(cap , (500,500) ,interpolation=cv.INTER_AREA)
sfr = SimpleFacerec()
sfr.load_encoding_images("Images Practice/")
while True :
    isframe , frame = cap.read()
    r_frame = cv.resize(frame , (700,500) ,interpolation=cv.INTER_AREA)
    img_loc , img_name = sfr.detect_known_faces(r_frame)
    for f_l , f_n in zip(img_loc, img_name):
        y1 , x2 ,y2 , x1 = f_l[0],f_l[1],f_l[2],f_l[3]
        cv.putText(r_frame,f_n,(x1,y1-8),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        cv.rectangle(r_frame , (x1,y1), (x2,y2) , (225,0,223) , 2)
    if cv.waitKey(25) and 0xFF == ord('d'):
        break
    cv.imshow("video",r_frame)
cap.release()
cv.destroyAllWindows()