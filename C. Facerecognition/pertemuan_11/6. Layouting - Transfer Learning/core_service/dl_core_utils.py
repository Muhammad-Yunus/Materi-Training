import os
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

class Preprocessing():
    def detect_face(self, img, size=(50,50), is_gray=True):
        img = img[70:195,78:172]
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size)
        return img

    def print_progress(self, val, val_len, folder, bar_size=20):
        progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
        if val == 0:
            print("", end = "\n")
        else:
            print("[%s] (%d samples)\t label : %s \t\t" % (progr, val+1, folder), end="\r")

    def load_dataset(self, dataset_folder = "dataset/",  size=(50,50), is_gray=True, max_sample=50):
        names = []
        images = []
        for folder in os.listdir(dataset_folder):
            files = os.listdir(os.path.join(dataset_folder, folder))[:max_sample]
            for i, name in enumerate(files): 
                if name.find(".jpg") > -1 :
                    img = cv2.imread(os.path.join(*[dataset_folder, folder, name]))
                    img = self.detect_face(img, size=size, is_gray=is_gray) # detect face using mtcnn and crop to 100x100
                    if img is not None :
                        images.append(img)
                        names.append(folder)

                        self.print_progress(i, len(files), folder)
        return names, images

    def image_augmentator(self, images, names):
        augmented_images = []
        augmented_names = []
        for i, img in enumerate(images):
            try :
                augmented_images.extend(self.img_augmentation(img))
                augmented_names.extend([names[i]] * 20)
            except Exception as e:
                print(e)
        images.extend(augmented_images)
        names.extend(augmented_names)
        return names, images
        
    def img_augmentation(self, img):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
        M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
        M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
        M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
        M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
        M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
        M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
        M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
        M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
        M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
        M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
        M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])

        imgs = []
        imgs.append(cv2.warpAffine(img, M_rot_5, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_rot_neg_5, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_rot_10, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_rot_neg_10, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_3, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_neg_3, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_6, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_neg_6, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_y3, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_neg_y3, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_y6, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.warpAffine(img, M_trans_neg_y6, (w, h), borderValue=(255,255,255)))
        imgs.append(cv2.add(img, 10))
        imgs.append(cv2.add(img, 30))
        imgs.append(cv2.add(img, -10))
        imgs.append(cv2.add(img, -30)) 
        imgs.append(cv2.add(img, 15))
        imgs.append(cv2.add(img, 45))
        imgs.append(cv2.add(img, -15))
        imgs.append(cv2.add(img, -45))

        return imgs
    
    def convert_categorical(self, names):
        le = LabelEncoder()
        le.fit(names)
        labels = le.classes_
        print("labels : ", labels)
        name_vec = le.transform(names)

        categorical_name_vec = to_categorical(name_vec)

        return categorical_name_vec
    
    def split_dataset(self, images, categorical_name_vec, test_size=0.15):
        x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.float32),   
                                                    np.array(categorical_name_vec),        
                                                    test_size=test_size, 
                                                    random_state=42)
        return x_train, x_test, y_train, y_test
    
class Evaluation():
    def plot_history(self, history):
        names = [['accuracy', 'val_accuracy'], 
                 ['loss', 'val_loss']]
        for name in names :
            fig1, ax_acc = plt.subplots()
            plt.plot(history.history[name[0]])
            plt.plot(history.history[name[1]])
            plt.xlabel('Epoch')
            plt.ylabel(name[0])
            plt.title('Model - ' + name[0])
            plt.legend(['Training', 'Validation'], loc='lower right')
            plt.grid()
            plt.show()
            
            
    def plot_confusion_matrix(self, y_test, y_pred, labels):
        cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=np.arange(len(labels)))
        np.set_printoptions(precision=2)
        self.plot_confusion_matrix_base(cnf_matrix, classes=labels,normalize=False,
                      title='Confusion matrix')
        
    def plot_confusion_matrix_base(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 8))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    def report(self, y_test, y_pred, labels):
        print(classification_report(y_test.argmax(axis=1), 
                            y_pred.argmax(axis=1), 
                            target_names=labels))
    
class Ped():
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