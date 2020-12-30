import os
import cv2
import pandas as pd
import numpy as np

class Recognizer():
    def __init__(self, facerecognition_model = "frozen_graph.pb", 
                        labels_filename="labels.csv", 
                        facedetection_model="haarcascade_frontalface_default.xml"):
        
        path = os.path.join(os.getcwd(), os.path.dirname(__file__))
        
        if os.path.isfile(os.path.join(path, labels_filename)) == False:
            raise Exception("Can't find %s" % os.path.join(path, labels_filename))
            
        self.labels = pd.read_csv(os.path.join(path, labels_filename))['0'].values
        
        self.face_cascade = cv2.CascadeClassifier(os.path.join(path, facedetection_model))
        
        self.net = cv2.dnn.readNet(os.path.join(path, facerecognition_model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layerOutput = self.net.getUnconnectedOutLayersNames()


    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (50, 50))

            blob = cv2.dnn.blobFromImage(face_img, 1.0, (50, 50), (0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(blob)
            output = self.net.forward(self.layerOutput)

            idx = output[0].argmax(axis=1)[0]
            confidence = output[0].max(axis=1)[0]*100

            if confidence > 70:
                label_text = "%s (%.2f %%)" % (self.labels[idx], confidence)
            else :
                label_text = "N/A"
            frame = self.draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
            
        return frame
    
    
    def draw_ped(self, img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img,
                      (x0, y0 + baseline),  
                      (max(xt, x0 + w), yt), 
                      color, 
                      2)
        cv2.rectangle(img,
                      (x0, y0 - h),  
                      (x0 + w, y0 + baseline), 
                      color, 
                      -1)  
        cv2.putText(img, 
                    label, 
                    (x0, y0),                   
                    cv2.FONT_HERSHEY_SIMPLEX,     
                    0.5,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img
    