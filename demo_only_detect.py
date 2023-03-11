# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:51:27 2020

@author: ChenYixuan
"""


import cv2
import numpy as np
from detection_module import HelmetDetector

detector=HelmetDetector()
detector.load()
#path='41.mp4'
cap = cv2.VideoCapture(0)
num=0
#deleter=Deleter()
while True:
    ret,img=cap.read()
    if ret:       
        #deleter.add_false_box(img[630:649,559:578,:])
        out=detector.detect([img])[0]
        clas,conf,bbox=out[0],out[1],out[2].astype(np.int)
        if len(clas):
            for i in range(len(clas)):
                x1,y1,x2,y2=bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3]
                
                if conf[i]>0.3:
                    num+=1
                    if clas[i] == 1:
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    else:
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv2.imshow("demo", img)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    
#print(num)