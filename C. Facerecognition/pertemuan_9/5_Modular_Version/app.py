import cv2
from flask import Flask, render_template, Response
from facerecognition import Recognizer

app = Flask(__name__)

recognizer = Recognizer()

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try :
                frame = recognizer.predict(frame)
            except :
                print("[ERROR] Can't recognize the face.")
                camera.release()
                break
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