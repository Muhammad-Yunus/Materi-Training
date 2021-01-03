import os
import cv2
import pandas as pd
import numpy as np
import mtcnn
import datetime

class Recognizer():
    def __init__(self, 
                        socketio,
                        facerecognition_model = "frozen_graph.pb", 
                        labels_filename="labels.csv", 
                        facedetection_model="haarcascade_frontalface_default.xml",
                        use_mtcnn = False,
                        camera_src=0):
        
        self.socketio = socketio

        if os.path.isfile(labels_filename) == False:
            raise Exception("Can't find %s" % labels_filename)
            
        self.labels = pd.read_csv(labels_filename)['0'].values
        self.use_mtcnn = use_mtcnn

        self.camera_src = camera_src
        self.camera = None 

        if self.use_mtcnn :
            self.face_cascade = MTCNN()
        else :
            self.face_cascade = cv2.CascadeClassifier(facedetection_model)
        
        self.net = cv2.dnn.readNet(facerecognition_model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layerOutput = self.net.getUnconnectedOutLayersNames()

        self.curr_frame = None

        self.label_stat = {}
        self.label_count = {}
        self.label_time = {}
        for name in self.labels:
            self.label_stat[name] = False
            self.label_count[name] = 0
            self.label_time[name] = datetime.datetime.now()


    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if self.use_mtcnn :
            faces = detector.detect_faces(img)
            faces = [ face['box'] for face in faces]
        else :
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
                curr_label = self.labels[idx]
                label_text = "%s (%.2f %%)" % (curr_label, confidence)
                if self.label_count[curr_label] > 5:
                    self.socketio.emit("prediction", {
                                                    'frame' :self.get_curr_frame(),
                                                    'label' : curr_label,
                                                    'status' : not self.label_stat[curr_label],
                                                    'time' : self.get_str_datetime()
                                                    })
                    self.socketio.sleep(0.1)
                    self.label_stat[curr_label] = not self.label_stat[curr_label]
                    self.label_time[curr_label] = datetime.datetime.now()
                    self.label_count[curr_label] = 0
                    
                else :
                    if self.check_diff_time(curr_label):
                        self.label_count[curr_label] += 1

            else :
                label_text = "N/A"
            frame = self.draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
            
            
        return frame
    
    
    def draw_ped(self, img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
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
                    0.8,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img
    
    def gen_frames(self):  
        while True:
            if self.camera is None :
                self.open()
            success, frame = self.camera.read()
            if not success:
                break
            else:
                try :
                    self.curr_frame = frame.copy()
                    frame = self.predict(frame)
                except Exception as e:
                    print("[ERROR] ", e)
                    self.camera.release()
                    self.camera = None
                    break
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def close(self):
        if self.camera is not None :
            self.camera.release()
            self.camera = None

    def open(self):
        self.camera = cv2.VideoCapture(self.camera_src)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    def status(self):
        return self.camera is not None

    def get_curr_frame(self):
        frame = cv2.resize(self.curr_frame, (0,0), fx=0.2, fy=0.2)
        ret, buffer = cv2.imencode('.png', frame)
        return buffer.tobytes()

    def get_str_datetime(self):
        return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def check_diff_time(self, label):
        label_time = self.label_time[label]
        now = datetime.datetime.now()

        return now - label_time > datetime.timedelta(seconds=5)