import os
import cv2
from flask import Flask, render_template, Response, request, flash
from flask_socketio import SocketIO

from core_service.facerecognition import Recognizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'

socketio = SocketIO(app)

PATH = '\\'.join(os.path.abspath(__file__).split('\\')[0:-1])
DATASET_PATH = os.path.join(PATH, "..\dataset")

recognizer = Recognizer(
    socketio=socketio,
    facerecognition_model = os.path.join(PATH, "core_Service\\bin\\frozen_graph.pb"), 
    labels_filename=os.path.join(PATH, "core_Service\\labels.csv"), 
    facedetection_model=os.path.join(PATH, "core_Service\\bin\\haarcascade_frontalface_default.xml"),
    use_mtcnn = False,
    camera_src=0
)

@app.route("/")
def index():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off':
        recognizer.close()
        flash("Camera turn off!", "info")
    elif camera is not None and camera == 'on':
        recognizer.open()
        flash("Camera turn on!", "success")
    print("camera status", recognizer.status())
    return render_template("index.html", is_camera = recognizer.status())

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/face_registration")
def face_registration():
    return render_template("face_registration.html")
    
@app.route('/video_feed')
def video_feed():
    return Response(recognizer.gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload_photo", methods=['POST'])
def upload_photo():
    class_name = request.args.get('class_name')
    path_new_class = os.path.join(DATASET_PATH, class_name)

    # create directory label if not exist
    if not os.path.exists(path_new_class):
        os.mkdir(path_new_class) 

    # save uploaded image
    filename = class_name + '%04d.jpg' % (len(os.listdir(path_new_class)) + 1) 
    file = request.files['webcam']
    file.save(os.path.join(path_new_class, filename))

    # resize
    img = cv2.imread(os.path.join(path_new_class, filename))
    img = cv2.resize(img, (250, 250))
    cv2.imwrite(os.path.join(path_new_class, filename), img)

    return '', 200

if __name__ == '__main__':
    socketio.run(app, debug=True)

    