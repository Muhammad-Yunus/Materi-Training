from flask import Flask, render_template, Response
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

app = Flask(__name__)

labels = ['Ariel_Sharon',
          'Colin_Powell',
          'Donald_Rumsfeld',
          'George_W_Bush',
          'Gerhard_Schroeder',
          'Hugo_Chavez',
          'Jacques_Chirac',
          'Jean_Chretien',
          'John_Ashcroft',
          'Junichiro_Koizumi',
          'Serena_Williams',
          'Tony_Blair',
          'Yunus']

facerecognition_model = "frozen_graph.pb"
net = cv2.dnn.readNet(facerecognition_model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layerOutput = net.getUnconnectedOutLayersNames()

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

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

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (50, 50))
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (50, 50), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward(layerOutput)
        
        idx = output[0].argmax(axis=1)[0]
        confidence = output[0].max(axis=1)[0]*100

        if confidence > 70:
            label_text = "%s (%.2f %%)" % (labels[idx], confidence)
        else :
            label_text = "N/A"
        frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
    return frame

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = recognize_face(frame)
                     
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


camera = cv2.VideoCapture(0)
app.run()