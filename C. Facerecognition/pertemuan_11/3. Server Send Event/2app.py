import time

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'  # This is a secret key that is used by Flask to sign cookies. 

socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('my_event')
def handle_message(message):
    print('received message: ', message)
    send_to_browser()
    
def send_to_browser():
    time.sleep(1)
    emit("server_push", "Hello from server")
        
if __name__ == '__main__':
    socketio.run(app)