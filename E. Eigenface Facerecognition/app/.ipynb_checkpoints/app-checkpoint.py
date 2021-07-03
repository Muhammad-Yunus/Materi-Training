#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import time
import datetime
import shutil
import numpy as np
import pandas as pd
import PySimpleGUI as sg 


# In[14]:


import cv2
import pickle
import matplotlib.pyplot as plt

import pyautogui
from PIL import ImageGrab
from win10toast import ToastNotifier

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[15]:


from user import User
from kehadiran import Kehadiran
from pengaturan import Pengaturan


pengaturan = Pengaturan()
kehadiran = Kehadiran()
user = User()


# - layout definition

# In[38]:


def get_button_style(key, filename="save.png"):
    return sg.Button(image_filename="asset/" + filename, 
                    key = key,
                    button_color=('#FFF', '#FFF'), 
                    image_size=(40, 40), 
                    image_subsample=3, 
                    border_width=0)


# In[39]:


def get_layout_user_item(record):
    Nama = record['Nama']
    NIM  = record['NIM']
    JenisKelamin = record['JenisKelamin']
    JamMasuk = record['JamMasuk']
    filename = record['NamaFoto']
    
    layout_photo = [[sg.Image(filename="photo/" + filename, key="foto-" + filename, size=(100,100))]]
    layout_user = [[sg.Text("Nama \t: " + Nama)],
                  [sg.Text("NIM \t: " + NIM)],
                  [sg.Text("Kelamin \t: " + JenisKelamin)],
                  [sg.Text("Masuk \t: " + JamMasuk)]]
    
    layout_item =[
                    sg.Column(layout_photo), 
                    sg.Column(layout_user)
                  ]
    return layout_item


# In[195]:


def get_class_info():
    curr_date = datetime.datetime.now().strftime("%A, %d %B %Y")
    layout = [
        [sg.Image(filename='asset/logo.png'),
         sg.Text("Absensi Berbasis Facerecognition", size=(15,2), font=("Helvetica", 17, "italic"))],
        [sg.HorizontalSeparator()],
        [sg.Text(curr_date, justification='right', size=(38,1), font=("Helvetica", 10, "italic"))],
        [sg.Text("Kelas \t\t: "), sg.InputText(key="Kelas", size=(24,1))],
        [sg.Text("Jumlah Peserta \t: "), sg.InputText(key="JumlahPeserta", size=(24,1))],
        [sg.Text("JumlahHadir \t: "), sg.Text("-", key="JumlahHadir", size=(24,1))],
        [sg.Text("JamMulai \t: "), sg.Text("-", key="JamMulai", size=(24,1))],
        [sg.Text("Durasi \t\t: "), sg.Text("-", key="Durasi", size=(24,1))],
        [sg.Text("Status \t\t: "), sg.Text("-", key="Status", size=(24,1))],
        [get_button_style("mulai", filename="play.png"), 
         get_button_style("akhiri", filename="stop.png"),
         get_button_style("reset", filename="reset.png")]
    ]
    return layout


# In[196]:


def get_layout_home(records):
    # list user     
    list_user_layout = [get_layout_user_item(record) for record in records]
    
    layout_left = [
        [
            sg.Column(list_user_layout, scrollable=True, vertical_scroll_only=True, size=(320, 400))
        ]
    ]
    
    
    # informasi kelas     
    layout_right = get_class_info()
    
    layout = [
        [
            sg.Column(layout_left),
            sg.VerticalSeparator(), # vertical separator
            sg.Column(layout_right)
        ]
    ]
    return layout


# In[197]:


def get_layout_history(records, headings, listKelas):
    layout = [
              [sg.Combo(listKelas, size=(35,1), key='search_history'), 
               sg.InputText(key="tanggal_history", size=(15,1)), 
               sg.CalendarButton("...", target='tanggal_history', format='%Y-%m-%d'),
               sg.Button('Search', key='button_search_history'),
               sg.Input(key='button_export_history', enable_events=True, visible=False),
               sg.FolderBrowse(button_text='Export', target='button_export_history')],
              [sg.Table(values=records, headings=headings, 
                             key='table_history', 
                             background_color='#f0f3f7',
                             text_color='black',
                             justification='left',
                             num_rows=20
                            )]
             ]
    return layout


# In[198]:


def get_layout_User(records, headings):
    layout = [
              [sg.InputText(size=(45,1), key='search_user'), 
               sg.Button('Search', key='button_search_user'),
               sg.Input(key='button_export_user', enable_events=True, visible=False),
               sg.FolderBrowse(button_text='Export', target='button_export_user'),
               sg.Button('Create New User', key='create_new_user')
              ],
              [sg.Table(values=records, headings=headings, 
                             key='table_user', 
                             background_color='#f0f3f7',
                             text_color='black',
                             justification='left',
                             num_rows=20,
                             enable_events=True
                            )]
             ]
    return layout


# In[199]:


def get_layout_UserDetail(record):
    layout_right = [
        [sg.Text("Nama \t\t :"), 
         sg.InputText(size=(30,1), key="formUser_Nama", default_text=record['Nama'])],
        [sg.Text("NIM \t\t :"), 
         sg.InputText(size=(30,1), key="formUser_NIM", default_text=record['NIM'])],
        [sg.Text("Kelamin \t\t :"), 
         sg.InputText(size=(30,1), key="formUser_Kelamin", default_text=record['JenisKelamin'])],
        [sg.Text("Umur \t\t :"), 
         sg.InputText(size=(30,1), key="formUser_Umur", default_text=record['Umur'])],
        [sg.Text("Alamat \t\t :"), 
         sg.Multiline(size=(30,4), key='formUser_Alamat', default_text=record['Alamat'])],
        [sg.Text("Prediction Id \t :"), 
         sg.InputText(size=(30,1), key="formUser_PredictionId", default_text=record['PredictionId'])],
        [sg.Text("Upload Foto \t :"),
         sg.InputText(size=(20,1), key="formUser_NamaFoto", disabled=True, default_text=record['NamaFoto']),
         sg.FileBrowse(button_text='Browse', target='formUser_NamaFoto', file_types=(('Image', '*.png'),))],
        
        [get_button_style("formUser_save", filename="save.png"), 
         get_button_style("formUser_delete", filename="delete.png"),
         get_button_style("formUser_cancel", filename="cancel.png")]
    ]
    layout_left = [
        [sg.Image(filename='photo/default_user_icon.png', key='formUser_Foto')]
    ]
    layout = [
        [sg.Column(layout_left),
         sg.Column(layout_right),]
    ]
    return layout


# In[200]:


def layout_get_pengaturan():
    layout = [
        [sg.Text("Classification Model \t:"), 
         sg.InputText(size=(40,1), 
                      key="classification_model", 
                      disabled=True, 
                      default_text=pengaturan.get_config('classification_model')),
         sg.FileBrowse(button_text='Browse', 
                       target='classification_model', 
                       file_types=(('Model', '*.pkl'),))],
        [sg.Text("Eigenface Model \t\t:"), 
         sg.InputText(size=(40,1), 
                      key="eigenface_model", 
                      disabled=True, 
                      default_text=pengaturan.get_config('eigenface_model')),
         sg.FileBrowse(button_text='Browse', 
                       target='eigenface_model', 
                       file_types=(('Model', '*.pkl'),))],
        [sg.Text("Threshold Accuracy \t:"), 
         sg.InputText(size=(40,1), 
                      key='threshold_accuracy', 
                      default_text=pengaturan.get_config('threshold_accuracy'))],
        [sg.Text("Jeda Deteksi (s) \t\t:"), 
         sg.InputText(size=(40,1), 
                      key='delay_capture', 
                      default_text=pengaturan.get_config('delay_capture'))],
        [sg.Text("Using Notification \t\t:"), 
         sg.Checkbox('', key='using_notification', 
                      size=(5,5),
                      default=bool(int(pengaturan.get_config('using_notification'))))],
        [get_button_style("save_pengaturan", filename="save.png")]
    ]
    return layout


# In[201]:


def get_layout(home_records, history_records, history_headings, listKelas, user_records, user_headings, user_record):
    
    layout_home = get_layout_home(home_records)
    layout_history = get_layout_history(history_records, history_headings, listKelas)
    layout_user = [
            [sg.Column(get_layout_User(user_records, user_headings), key='user_list'), 
             sg.Column(get_layout_UserDetail(user_record), visible=False, key='user_detail')]
    ]
    layout_pengaturan = layout_get_pengaturan()

    layout = [[sg.TabGroup([
                            [sg.Tab('Home', layout_home, border_width=20), 
                             sg.Tab('History', layout_history, border_width=20),
                             sg.Tab('User', layout_user, border_width=20),
                             sg.Tab('Pengaturan', layout_pengaturan, border_width=20)]
                            ])
              ]]
    return layout


# - facerecognition

# In[202]:


def screen_capture(window_name= 'Zoom Meeting'):
    if window_name not in pyautogui.getAllTitles():
        return None
    try :
        fw = pyautogui.getWindowsWithTitle(window_name)[0]
        fw.maximize()
        pyautogui.click(fw.center)
        fw.activate()
    except :
        pass
    
    time.sleep(1)

    sct = ImageGrab.grab() 
    img = np.array(sct)
    
    return img


# In[203]:


def preprocess(img):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    img_list = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img_face = img_gray[y:y+h, x:x+w]  # crop face image 
        img_resize = cv2.resize(img_face, (100, 100)) # resize to 100 x 100 pixel
        img_list.append(img_resize)
    return img_list, faces


# In[204]:


def read_model(filename, path=""):
    with open(os.path.join(path, filename), 'rb') as in_name:
        model = pickle.load(in_name)
        return model


# In[205]:


def recognizer(labels, window_name= 'Zoom Meeting', 
               show_toast=bool(pengaturan.get_config('using_notification')),
               threshold=float(pengaturan.get_config('threshold_accuracy'))):
    # get data     
    img = screen_capture(window_name=window_name)
    if img is None :
        print("[INFO] could not find window !")
        return None

    # preprocessing    
    img_list, face_coords = preprocess(img)
    if len(img_list) < 1 :
        print("[INFO] could not find face!")
        return None
    else : 
        # convert each detected face to 1D array feature vector
        img_list_flatten = [img.flatten() for img in img_list]

        # apply PCA to each 1D array feature vector
        img_list_pca = pca.transform(img_list_flatten)

        # predict data using SVM    
        ids = np.array(model_svm.predict(img_list_pca))

        proba = model_svm.predict_proba(img_list_pca)
        confidence = np.array([np.max(p) for p in proba])

        reff = [i for i, conf in enumerate(confidence) if conf >= threshold]

        filtered_ids = ids[reff]
        filtered_confidence = confidence[reff]
        
        label_output = [labels[i] for i in filtered_ids]
        
        for i in range(len(filtered_ids)):
            title = "Name : %s (%.2f%%)" % (label_output[i], (filtered_confidence[i]*100))
            if show_toast:
                toaster.show_toast("Attendance Systems", title)
        
        
    return filtered_ids


# - utils

# In[206]:


def create_window(layout):
    return sg.Window(title="Aplikasi Pencatatan Kehadiran", layout=layout, margins=(10, 10), finalize=True)


# In[207]:


def format_history(records):
    history_records = [list(item.values()) for item in records]
    for i, item in enumerate(history_records) :
        item.insert(0, i)
        del item[4:6]

    if len(history_records) < 1 :
        history_records = [[""]*len(history_headings)]
    return history_records


# In[208]:


def format_user(records):
    user_records = [list(item.values()) for item in records]
    for i, item in enumerate(user_records) :
        item[0] = i
        del item[5:7]
        
    if len(user_records) < 1 :
        user_records = [[""]*len(user_records)]
    return user_records


# In[209]:


def export_csv(records, headings, export_name='data.csv'):
    if len(records) > 0 : 
        df = pd.DataFrame(records, columns=headings)

        try :
            df.to_csv(export_name, index=None)
            sg.Popup("Data berhasil disimpan di %s!" % export_name)
        except :
            print("Gagal menyimpan file!")
    else :
        sg.Popup("Tidak dapat menyimpan file, data kosong!")


# In[210]:


def update_form_user(user_record):
    window['formUser_Nama'].update(value=user_record['Nama'])
    window['formUser_NIM'].update(value=user_record['NIM'])
    window['formUser_Kelamin'].update(value=user_record['JenisKelamin'])
    window['formUser_Umur'].update(value=user_record['Umur'])
    window['formUser_Alamat'].update(value=user_record['Alamat'])
    window['formUser_PredictionId'].update(value=user_record['PredictionId'])
    window['formUser_NamaFoto'].update(value=user_record['NamaFoto'])
    window['formUser_Foto'].update(filename='photo/' + user_record['NamaFoto'])


# - init data

# In[211]:


toaster = ToastNotifier()

labels = ["-"]*10 
for record in user.select_user():
    labels.insert(record['PredictionId'], record['Nama'])
    
curr_date = datetime.datetime.now().strftime("%Y-%m-%d")

#  init table history
curr_user_records = []
history_headings = [" No ", 
                    " Nama                        ", 
                    " NIM             ", 
                    " Kelamin ", 
                    " Jam Masuk       "]
history_records = [[""]*len(history_headings)]
listKelas = kehadiran.select_nama_kelas()

# init table users
user_headings = [" No ",
                 " Nama                        ",
                 " NIM              ",
                 " Kelamin ",
                 " Umur   ",
                 " Prediction Id"]
user_records = format_user(user.select_user())
user_records = [[""]*len(user_headings)] if len(user_records) < 2 else user_records

# init form user 
null_user_record = dict(
    Id = '',
    Nama = '',
    NIM = '',
    JenisKelamin = '',
    Umur = '',
    Alamat = '',
    PredictionId = '',
    NamaFoto = 'default_user_icon.png'
)
user_record = null_user_record

# init config
pca = read_model(pengaturan.get_config('eigenface_model'))
model_svm = read_model(pengaturan.get_config('classification_model'))
delay_capture = int(pengaturan.get_config('delay_capture')) #second 


# - main program

# In[213]:


layout = get_layout(curr_user_records, 
                    history_records, 
                    history_headings, 
                    listKelas, 
                    user_records, 
                    user_headings, 
                    user_record)
window = create_window(layout)

window['akhiri'].update(disabled=True)
window['reset'].update(disabled=True)
window['tanggal_history'].update(value=curr_date)

status = ''
start_time = ''
update = False
reset = False

while True :
    event, values = window.read(timeout=25)
    
    # ----------- home event handler -----------
    if event == 'mulai' or update == True:
        if event == 'mulai' and (values['Kelas'] == '' or values['JumlahPeserta'] == '' ):
            sg.Popup("Isi data kelas terlebih dahulu!")
        else :
            window.close()

            if event == 'mulai' :
                start_time = datetime.datetime.now()

            curr_user_records = user.get_user_in_class(curr_date, values['Kelas'])
            listKelas = kehadiran.select_nama_kelas()
            history_records = format_history(curr_user_records)

            layout = get_layout(curr_user_records, 
                        history_records, 
                        history_headings, 
                        listKelas, 
                        user_records, 
                        user_headings, 
                        user_record)
            window = create_window(layout)

            status = 'berlangsung' if not reset else '-'
            window['Kelas'].update(value=values['Kelas'])
            window['JumlahPeserta'].update(value=values['JumlahPeserta'])
            window['JumlahHadir'].update(value=len(curr_user_records))
            window['Status'].update(value=status)
            window['JamMulai'].update(value=start_time.strftime("%H:%M:%S") if not reset else '-')

            window['mulai'].update(disabled=True if not reset else False)
            window['akhiri'].update(disabled=False if not reset else True)
            window['reset'].update(disabled=True)

            window['search_history'].update(value=values['Kelas'])
            window['tanggal_history'].update(value=curr_date)

            if update :
                update = False
                reset = False
        
    if event == 'akhiri':
        status = 'selesai'
        window['Status'].update(value=status)
        window['mulai'].update(disabled=True)
        window['akhiri'].update(disabled=True)
        window['reset'].update(disabled=False)
        
        kehadiran.update_kehadiran_selesai(values['Kelas'], 
                                           curr_date, 
                                           datetime.datetime.now().strftime("%Y-%m-%d"))

    if event == 'reset' :
        window['Kelas'].update(value='')
        window['JumlahPeserta'].update(value='')
        update = True
        reset = True
        
    # --------------- event handler screen history -------------------------     
    if event == 'button_search_history' :
        user_records = user.get_user_in_class(values['tanggal_history'], values['search_history'])
        history_records = format_history(user_records)
        window['table_history'].update(history_records)
        
    if event == 'button_export_history' :        
        export_name = "%s/absensi_%s_%s.csv" % (
                                               values['button_export_history'], 
                                               values['search_history'], 
                                               values['tanggal_history'])
        user_records = user.get_user_in_class(values['tanggal_history'], values['search_history'])
        history_records = format_history(user_records)
        
        export_csv(history_records, history_headings, export_name = export_name)
          
    # ----------- event handler screen user & user detail -------------------- 
    if event == 'button_search_user' :
        user_records = format_user(user.select_user(values['search_user']))
        window['table_user'].update(user_records)
        
    if event == 'button_export_user' :        
        export_name = "%s/user_export.csv" % (values['button_export_user'])
        user_records = format_user(user.select_user(values['search_user']))

        export_csv(user_records, user_headings, export_name = export_name)
    
    if event == 'create_new_user' :
        update_form_user(null_user_record)   
        window['user_list'].update(visible=False)
        window['user_detail'].update(visible=True)
        
    if event == 'table_user':
        idx = int(values['table_user'][0])
        PredictionId = format_user(user.select_user(values['search_user']))[idx][-1]
        user_record = user.select_user_by_prediction_id(PredictionId)

        update_form_user(user_record)        
        window['user_list'].update(visible=False)
        window['user_detail'].update(visible=True)
        
    if event =='formUser_delete' :
        PredictionId = int(values['formUser_PredictionId'])
        user_record = user.select_user_by_prediction_id(PredictionId)
        user.delete_user(user_record['Id'])
        sg.Popup("User %s berhasil didelete!" % user_record['Nama'])
        
    if event =='formUser_save' :
        PredictionId = int(values['formUser_PredictionId'])
        user_record = user.select_user_by_prediction_id(PredictionId)
        user_record = user_record if user_record is not None else {}
        
        user_record['Nama'] = values['formUser_Nama']
        user_record['NIM'] = values['formUser_NIM']
        user_record['JenisKelamin'] = values['formUser_Kelamin']
        user_record['Umur'] = values['formUser_Umur']
        user_record['Alamat'] = values['formUser_Alamat']
        user_record['PredictionId'] = values['formUser_PredictionId']
        user_record['NamaFoto'] = values['formUser_NamaFoto'].split("/")[-1]
        
        if len(values['formUser_NamaFoto'].split("/")) > 1 :
            shutil.move(values['formUser_NamaFoto'], 'photo/' + values['formUser_NamaFoto'].split("/")[-1])  
        
        if 'Id' in user_record:
            ret = user.update_user(user_record)
            sg.Popup("User %s berhasil diupdate!" % user_record['Nama'])
        else :
            ret = user.create_user(user_record)
            sg.Popup("User %s berhasil disimpan!" % user_record['Nama'])
        
    if event == 'formUser_save' or event =='formUser_delete' or event == 'formUser_cancel' :
        user_records = format_user(user.select_user(values['search_user']))
        window['table_user'].update(user_records)
        
        window['user_list'].update(visible=True)
        window['user_detail'].update(visible=False)
    
    # ---------------- event handler pengaturan -------------
    if event == 'save_pengaturan':
        pengaturan.update_config('classification_model', values['classification_model'])
        pengaturan.update_config('eigenface_model', values['eigenface_model'])
        pengaturan.update_config('threshold_accuracy', float(values['threshold_accuracy']))
        pengaturan.update_config('using_notification', bool(int(values['using_notification'])))
        pengaturan.update_config('delay_capture', int(values['delay_capture']))
        
        sg.Popup("Aplikasi perlu di-restart setelah update pengaturan!")
        break
    
    # ------------- face recognizer handler -----------------
    if status == 'berlangsung':
        curr_time = datetime.datetime.now()
        diff_time = curr_time - start_time
        window['Durasi'].update(value=str(diff_time).split(".")[0])
        
        if curr_time.second % delay_capture == 0 :
            predicted_ids = recognizer(labels, show_toast=False)
            if predicted_ids is not None :
                for idx in predicted_ids :
                    if idx not in [record['PredictionId'] for record in curr_user_records] :
                        record_user = user.select_user_by_prediction_id(int(idx))
                        if record_user is not None :
                            
                            record_kehadiran = {}
                            record_kehadiran['UserId'] = record_user['Id']
                            record_kehadiran['JamMasuk'] = datetime.datetime.now().strftime("%H:%M:%S")
                            record_kehadiran['NamaKelas'] = values['Kelas']
                            record_kehadiran['JamKelasMulai'] = start_time.strftime("%H:%M:%S")
                            record_kehadiran['JamKelasBerakhir'] = ''
                            record_kehadiran['Status'] = status
                            record_kehadiran['Date'] = curr_date

                            kehadiran.create_kehadiran(record_kehadiran)
                            
                            update = True
                            
            
    if event == sg.WIN_CLOSED:
        break

window.close()


# In[58]:


window.close()


# In[ ]:





# In[ ]:




