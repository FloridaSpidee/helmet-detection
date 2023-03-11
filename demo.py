# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:19:17 2020

@author: ChenYixuan
"""

import cv2
import numpy as np
import tracker
import copy
import time
from detection_module import HelmetDetector
import threading
import queue

detector=HelmetDetector()
detector.load()


class newThread():
    def __init__(self,threadID, name):
        self.threadID = threadID
        self.name = name
        self.imgs_now=None
    def getimgs(self,imgs):
        self.imgs_done=self.imgs_now
        self.imgs_now=imgs
    def detect(self):
        self.out=detector.detect(self.imgs_now)
    def get_result(self):
        return self.out
    def get_img(self):       
        return self.imgs_done[0]
        

if __name__ == "__main__":
    interval=3
    frame=0
    flag=0#用于判断是否已经开始展示图片
    frameID=0
    is_dect=False
    mot_tracker = tracker.Tracker((512,384, 3), min_hits=2, num_classes=2, interval=interval)
    path='41.mp4'
    cap = cv2.VideoCapture(path)
    img_que = queue.Queue()#img_que存放延时图片
    thread1 = newThread(1, "Thread-1")
    #thread1.start()
    total_time=0
    while True:
        
        ret, img = cap.read()
        #print(img_que.qsize())
        #print(img)
        dets,labels=[],[]
        if ret:
            
 
            if not flag and np.mod(frame, interval) == 0: #判断是否为第一帧                
                thread1.getimgs([img])
                thread1.detect()                
                #print(newthread.isAlive())
            elif not flag and np.mod(frame, interval) != 0:#若是未展示部分的间隔帧，不进行检测，放入img_que
                img_que.put(img)  #还未开始展示时放入的图片
            elif flag:
                time_start=time.time()
                if np.mod(frame, interval) == 0:
                    
                    #time_start=time.time()
                    
                    out=thread1.get_result()[0] #取出先前thread1已计算好的结果
                    thread1.getimgs([img])
                    newthread=threading.Thread(target=thread1.detect) #重新扔进一张实际捕捉到的图片进行detect
                    newthread.start()
                    img=thread1.get_img()  #取出展示的图片
                    clas,conf,bbox=out[0],out[1],out[2].astype(np.int)                   
                    #time_end=time.time()
                    #print(time_end-time_start)                    
                    dets_arr=copy.deepcopy(bbox)
                    for num in range(len(dets_arr)):
                        label=[clas[num],is_dect,num]
                        labels.append(label)
                    labels_arr=np.array(labels)
                    is_dect=True
                elif np.mod(frame, interval) != 0:
                    img_que.put(img)#放入实际捕捉到的图片
                    img=img_que.get()#取出展示的图片
                    dets_arr, labels_arr = np.array([]), np.array([])
                    is_dect = False
                time_end=time.time()
                print(time_end-time_start)
                trackers_arr = mot_tracker.update(dets_arr,labels_arr,is_dect=is_dect)#带入tracker进行update
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                for data in trackers_arr: #data形式为[x1,y1,x2,y2,number,is_update(0,1), labelID(clas)]
                    x1,y1,x2,y2,is_helmet=data[0],data[1],data[2],data[3],data[6]
                    if is_helmet:
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)                        
                    else:
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                
                cv2.imshow("demo", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))                
                
                if cv2.waitKey(42) & 0xFF == ord('q'):
                    break
            #print('frame',frameID,' costs ',time_end-time_start,'s')
            #print('cost',time_end-time_start,'s')
            #total_time+=(time_end-time_start)
            frame+=1
            if frame==interval:
                frame-=interval
                if not flag:
                    flag=1#第一次frame值达到interval时就开始展示，此时flag转变为1
            frameID+=1
            
        else:
            break
        
        
        
        #print('frame',frameID,' costs ',time_end-time_start,'s')
        #print('fps is ',frameID/total_time)
    cap.release()
    cv2.destroyAllWindows()
                
