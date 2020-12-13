import os
import cv2
import math
import time
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import pandas as pd


def detectColor(img, lower, upper): 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv.copy(), lower, upper)
        res = cv2.bitwise_and(img, img, mask= mask)
        return res
    
def findContourCircle(res): 
        x, y, r = 0, 0, 0
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cnt for cnt in contours if contour_ok(cnt)]
        circle = []
        for cnt in contours:
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 10: 
                circle.append([int(x), int(y), int(radius)])
        if len(circle) > 0 :
            circle = np.array(circle)
            x = int(np.mean(circle[:, 0]))
            y = int(np.mean(circle[:, 1]))
            r = int(np.mean(circle[:, 2]))
        return x, y, r
    
def contour_ok(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    
    return not (w < 30 or h < 30)    


def calc_speed(x, y, t):
    dx = float(x[1]) - float(x[0])
    dy = float(y[1]) - float(y[0])
    ds = math.sqrt( math.pow(dx,2) + math.pow(dy,2))
    dt = float(t[1]) - float(t[0])
    v = float(ds/dt)
    vx = float(dx/dt)
    vy = float(dy/dt)
    return v, vx, vy


def create_custom_button(image_filename, key, disabled=False):
    return sg.Button(key=key, 
                      disabled=disabled, 
                      image_filename=image_filename, 
                      button_color=('#FFF', '#FFF'), 
                      image_size=(40, 40), 
                      image_subsample=3, 
                      border_width=0)

def clean_video(cap):
    if cap is not None:
        cap.release()
        cap = None
    window['-IMAGE-'].update(filename='ready_video.png')
    return cap


def readJson_config(Path, Name, Key):
    with open(Path + Name) as json_config:
        json_object = json.load(json_config)

    return json_object[Key]


def writeJson_config(Path, Name, Data, append):
    mode = 'a+' if append else 'w'
    full_path = Path + Name

    with open(full_path, mode=mode) as json_config:
        json.dump(Data, json.load(json_config) if append else json_config)
    
    return 'success' 

def updateJson_config(Path, Name, Key, Value):
    with open(Path + Name) as json_config:
        json_object = json.load(json_config)

    json_object[Key] = Value
    
    with open(Path + Name, mode='w') as json_config:
        json.dump(json_object, json_config)
        
    return 'success' 

def check_setting():
    return os.path.exists("./setting.json")


def default_setting():
    if not check_setting():
        Data = dict(
            FPS = int((1/25)*1000),  # for 25ms delay
            Format_File = "*.MP4"
        )

        writeJson_config("./", "setting.json", Data, False)
        
        
def create_input_setting(width=25, key='', setting_key=''):
    return sg.In(size=(width, 1), 
            key=key, 
            default_text=readJson_config("./", "setting.json", setting_key))

def readBytesFromDetectedObjectFrame(cap):
    global i, t0, x0, y0 
    frame_byte = None
    v_, vx_, vy_ = 0, 0, 0
    center = [0, 0]
    ret, img = cap.read()
    if ret:
        h, w, c = img.shape
        color = color_ranges[color_input]
        res = detectColor(img, color['lower'], color['upper'])
        
        x, y, r = findContourCircle(res)
        try :
            cv2.circle(img, (x, y), r, (255, 0, 255), 2)

            center = [x, y]
            cv2.putText(img, "(%d, %d)" % (x, y), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA) 
            xs.append(x) 
            ys.append(y) 

            if i % 15 : 
                t = time.time()
                x_ = np.mean(xs[-60:]) 
                y_ = np.mean(h - np.array(ys[-60:])) 
                v, vx, vy = calc_speed([x0, x_], [y0, y_], [t0, t])
                v_, vx_, vy_ = v*ratio, vx*ratio, vy*ratio

                t0, y0, x0 = t, y_, x_

                x, y = x - r, y + r
                cv2.line(img, (x,y), (x, y - r), (200, 200, 200), 2)
                cv2.line(img, (x,y), (x + r, y), (200, 200, 200), 2)
                cv2.arrowedLine(img, (x, y), (x, y - int(vy*scale)), (0,255,255), 2, cv2.LINE_AA) 
                cv2.arrowedLine(img, (x, y), (x - int(vx*scale), y), (0,255,0), 2, cv2.LINE_AA) 
                cv2.arrowedLine(img, (x, y), (x - int(vx*scale), y - int(vy*scale)), (0,0,255), 2, cv2.LINE_AA) 
                if v > 2 :
                    cv2.putText(img, "%.2f cm/s" % v_, (x - int(vx*scale), y - int(vy*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA) 
                trajectory.append([x,y]) 
            
            if len(trajectory) > 60 and is_view_trajectory: 
                vec_trajectory = np.array(trajectory)
                cv2.polylines(img, [vec_trajectory], False, (255,255,0), 1, cv2.LINE_AA)
                
                xc = vec_trajectory[:,0]  
                yc = vec_trajectory[:,1]  
                orde = 2 if len(xc) <= 100 else 3 if len(xc) > 100 and len(xc) < 500 else 5 
                poly_solve = np.unique(yc), np.int0(np.poly1d(np.polyfit(yc, xc, orde))(np.unique(yc)))  
                cv2.polylines(img, [np.array(list(zip(poly_solve[1], poly_solve[0])))], False, (0,0,255), 1, cv2.LINE_AA)  

            (x1, y1), (x2, y2) = (int(w*0.3), int(h*0.8)), (int(w*0.3), int(h*0.9))
            (xm, ym) = (int(w*0.31), int(h*0.85 + 10))
            cv2.line(img, (x1, y1), (x2, y2), (200), 2)
            cv2.putText(img, "%.2f cm" % ((y2-y1)*ratio) , (xm, ym), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e: 
            print("[ERROR] Error : %s" % e)
            pass
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        ret, frame_png = cv2.imencode('.png', img)
        frame_byte = frame_png.tobytes()

        del xs[:-120]
        del ys[:-120]
        i += 1 

    return ret, frame_byte, (v_, vx_, vy_), center


if __name__ == "__main__":
    # ------------------------------------------- COLOR RANGE -------------------------------------
    color_ranges = dict(
        blue = dict(
            lower = np.array([110, 50, 50]),
            upper = np.array([130, 255, 255])
        ),
        orange = dict(
            lower = np.array([10, 50, 50]),
            upper = np.array([20, 255, 255])
        ) 
    )

    availabel_colors = color_ranges.keys()

    #  ------------------------------------------ LAYOUT --------------------------------------------

    image_play = "asset/play.png"
    image_pause = "asset/pause.png"
    image_stop = "asset/stop.png"

    menu_def = [['Home', 'Home'],      
                ['Setting', 'Setting'],      
                ['Help', 'About...']]

    file_list_column = [
        [
            sg.Text("Video Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(values=[], enable_events=True, size=(40, 28), key="-FILE LIST-")
        ],
    ]

    image_viewer_column = [
        [
        create_custom_button(image_play, key='play', disabled=True),
        create_custom_button(image_pause, key='pause', disabled=True),
        create_custom_button(image_stop, key='stop', disabled=True),
        sg.Checkbox('View Trajectory', key='-view_trajectory-', enable_events=True),
        sg.Combo(['blue', 'orange'], default_value='blue', size = (10, 1), key='-color-', enable_events=True)
        ],
        [sg.Image(filename='default_video.png', key="-IMAGE-")],
        [sg.Column([
            [sg.Text("Position \t: x(0), y(0)", key='-position_label-', size=(25, None))],
            [sg.Text("v \t: 0 cm/s", key='-v_label-', size=(25, None))],
            [sg.Text("vx \t: 0 cm/s", key='-vx_label-', size=(25, None))],
            [sg.Text("vy \t: 0 cm/s", key='-vy_label-', size=(25, None))]
        ], size=(350, 100)),
         sg.Column([
             [sg.Button('Save Result')]
         ])
        ]
    ]



    layout_home = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    layout_setting = [
        [sg.Text('FPS \t\t: ', size=(15, None)), create_input_setting(width=25, key='-fps-', setting_key='FPS')],
        [sg.Text('Format File \t: ', size=(15, None)), create_input_setting(width=25, key='-format_file-', setting_key='Format_File')],
        [sg.Text('Position \t\t: ', size=(15, None)), 
             sg.Text('x0 : ', size=(3, None)), create_input_setting(width=5, key='-x0-', setting_key='x0'),
             sg.Text('y0 : ', size=(3, None)), create_input_setting(width=5, key='-y0-', setting_key='y0')],
        [sg.Text('Scale \t\t: ', size=(15, None)), create_input_setting(width=25, key='-scale-', setting_key='Scale')],
        [sg.Text('Ratio (cm/s) \t: ', size=(15, None)), create_input_setting(width=25, key='-ratio-', setting_key='Ratio')],
        [sg.Button('Save')]
    ]

    layout = [[sg.Menu(menu_def, )], 
              [sg.Column(layout_home, key='-home-'), sg.Column(layout_setting, visible=False, key='-setting-')]]


    # --------------------------------------------- MAIN PROGRAM -----------------------------------

    window = sg.Window("Video Viewer + Play List", finalize=True, layout=layout, margins=(10, 10))

    window.set_min_size((700,400))

    cap = None
    video_path = ''
    filename = ''
    stat = ''

    default_setting()
    t0 = time.time()
    x0 = int(readJson_config("./", "setting.json", "x0")) 
    y0 = int(readJson_config("./", "setting.json", "y0"))
    scale = float(readJson_config("./", "setting.json", "Scale"))
    ratio = float(readJson_config("./", "setting.json", "Ratio"))
    fps = int(readJson_config("./", "setting.json", "FPS"))

    color_input = 'blue'
    is_view_trajectory = False

    trajectory = [] 
    xs, ys = [], [] 
    i = 0

    history = []
    while True:
        event, values = window.read(timeout=int(1000/fps))
        if event == sg.WIN_CLOSED:
            break

        if event == "Home" :
            window['-home-'].update(visible=True)
            window['-setting-'].update(visible=False)

        if event == "Setting" :
            window['-home-'].update(visible=False)
            window['-setting-'].update(visible=True)

        if event == "About...":
            sg.popup('About', 'Version 1.0', 'OpenCV Object Tracking')

        if event == "Save" :
            updateJson_config("./", "setting.json", "FPS", values["-fps-"])
            updateJson_config("./", "setting.json", "Format_File", values["-format_file-"])
            updateJson_config("./", "setting.json", "x0", values["-x0-"])
            updateJson_config("./", "setting.json", "y0", values["-y0-"])
            updateJson_config("./", "setting.json", "Ratio", values["-ratio-"])
            updateJson_config("./", "setting.json", "Scale", values["-scale-"])

            #global x0, y0, scale, ratio
            x0, y0 = int(readJson_config("./", "setting.json", "x0")), int(readJson_config("./", "setting.json", "y0"))
            scale = float(readJson_config("./", "setting.json", "Scale"))
            ratio = float(readJson_config("./", "setting.json", "Ratio"))
            fps = int(readJson_config("./", "setting.json", "FPS"))

            sg.popup('Setting saved successfully!')

        if event == "Save Result" :
            df = pd.DataFrame(history, columns=['Time', "V (cm/s)", "Vx (cm/s)", "Vy (cm/s)", "X", "Y"])
            df.to_csv('Result-%s.csv' % filename, index=None)
            sg.popup('Result saved successfully!')

        if event == "-view_trajectory-" :
            is_view_trajectory = values['-view_trajectory-']

        if event == "-color-" :
            color_input = values['-color-']
            cap = clean_video(cap)
            stat = 'stop'
            #trajectory = []
            history = []
            window['play'].update(disabled=False)
            window['pause'].update(disabled=True)
            window['stop'].update(disabled=True) 

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(("video", ".mp4"))
            ]

            window["-FILE LIST-"].update(fnames)
            window['-IMAGE-'].update(filename='ready_video.png')

        elif event == "-FILE LIST-":  
            try:
                folder = values["-FOLDER-"]
                filename = values["-FILE LIST-"][0]
                video_path = os.path.join(folder, filename)

                cap = clean_video(cap)
                window['play'].update(disabled=False)
                window['pause'].update(disabled=True)
                window['stop'].update(disabled=True) 
            except:
                pass

        elif event == 'play' or event == 'pause' or event == 'stop':
            stat = event
            if event == 'play' :
                #trajectory = []
                history = []
                window['play'].update(disabled=True)
                window['pause'].update(disabled=False)
                window['stop'].update(disabled=False)
            elif event == 'pause' or event == 'stop':
                window['play'].update(disabled=False)
                window['pause'].update(disabled=True)
                window['stop'].update(disabled=True) 

        if stat == 'play':
            if cap is None:
                cap = cv2.VideoCapture(video_path)
            ret, frame_byte, (v_, vx_, vy_), (x, y) = readBytesFromDetectedObjectFrame(cap)
            history.append([datetime.datetime.now().strftime("%H:%M:%S.%f"), v_, vx_, vy_, x, y])
            if not ret:
                cap = clean_video(cap)

            window['-IMAGE-'].update(data=frame_byte)
            window['-position_label-'].update(value='Position \t: x(%d), y(%d)' % (x, y))
            window['-v_label-'].update(value='v \t: %.2f cm/s' % v_)
            window['-vx_label-'].update(value='vx \t: %.2f cm/s' % vx_)
            window['-vy_label-'].update(value='vy \t: %.2f cm/s' % vy_)


        elif cap is not None and stat == 'stop':
            cap = clean_video(cap)


    window.close()