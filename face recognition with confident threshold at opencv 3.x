import numpy as np
import cv2

cam=cv2.VideoCapture(0);
faceDetect=cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml');
rec=cv2.face.createLBPHFaceRecognizer();
rec.load('recognizer/trainingData.yml')
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.1,5);
        for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                result=cv2.face.MinDistancePredictCollector()
                rec.predict(gray[y: y+h, x: x+w],result,0)
                #conf=rec.predict(gray[y: y+h, x: x+w])
                nbr_predicted=result.getLabel()
                conf=result.getDist()
                
                id=nbr_predicted
                if(id==1):
                     id="obama"
                     print("obama",",",conf)#print name and confident level on the image
                     #print(conf)
                     cv2.putText(img,str(id),(10,50),font,1,(255,255,255),2);
             
                elif(id==2):
                     id="donald"
                     print("donald",",",conf)
                     cv2.putText(img,str(id),(10,50),font,1,(255,255,255),2);
                elif(id==3):
                     id="aralei"
                     print("arale",",",conf)
                     cv2.putText(img,str(id),(10,50),font,1,(255,255,255),2);
                     
                elif(id==5):
                     id="sinhui"
                     print("sinhui",",",conf)
                     cv2.putText(img,str(id),(10,50),font,1,(255,255,255),2);
        cv2.imshow('img',img);
        if(cv2.waitKey(1)==ord('q')):
                break;
        
cam.release()
cv2.destroyAllWindows()
