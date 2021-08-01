import os
import cv2
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools

#############################################################################
#
#
# Preprocessing
#
#
#############################################################################

class Preprocessing : 
    def __init__(self, DATASET_FOLDER = "Dataset_Tomat/"):
        self.labels = []
        self.image_list = []
        self.image_range = []
        self.image_edged = []
        self.contours_list = []
        self.filtered_contours_list = []
        self.image_croped = []
        self.image_resized = []
        self.DATASET_FOLDER = DATASET_FOLDER
        
        # define range of red color in HSV
        self.lower_red = np.array([-10, 75, 50])
        self.upper_red = np.array([10, 255, 255])

        # define range of green color in HSV
        self.lower_green = np.array([35, 100, 50])
        self.upper_green = np.array([70, 255, 255])
        
        # define range of yellow color in HSV
        self.lower_yellow = np.array([10, 125, 50])
        self.upper_yellow = np.array([35, 255, 255])
        
        # define range of black color in HSV
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([255, 255, 50])
        
    def ImageRead(self):
        for folder in os.listdir(self.DATASET_FOLDER):
            for sample in os.listdir(os.path.join(self.DATASET_FOLDER, folder)):
                sample_imgs = []
                for file in os.listdir(os.path.join(self.DATASET_FOLDER, folder, sample)) :
                    img = cv2.imread(os.path.join(self.DATASET_FOLDER, folder, sample, file).replace("\\","/"))
                    sample_imgs.append(img)
                self.image_list.append(sample_imgs)
                self.labels.append(folder) # append label (name) of image
                            
    def RangeTresholding(self):
        for sample_imgs in self.image_list : 
            sample_range_imgs = []
            for img in sample_imgs :
                # convert to hsv
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # apply range thresholding
                mask_green = cv2.inRange(hsv.copy(), self.lower_green, self.upper_green)
                mask_red = cv2.inRange(hsv.copy(), self.lower_red, self.upper_red)
                mask_yellow = cv2.inRange(hsv.copy(), self.lower_yellow, self.upper_yellow)
                mask_black = cv2.inRange(hsv.copy(), self.lower_black, self.upper_black)

                mask = mask_green + mask_red + mask_yellow + mask_black 
                res = cv2.bitwise_and(img, img, mask= mask)
                sample_range_imgs.append(res)
            self.image_range.append(sample_range_imgs)
            
    def EdgeDetection(self):
        for sample_imgs in self.image_range :
            sample_eged_imgs = []
            for img in sample_imgs :
                edged = cv2.Canny(img, 200, 210)
                sample_eged_imgs.append(edged)
            self.image_edged.append(sample_eged_imgs)
            
    def FindContour(self):
        for sample_imgs in self.image_edged:
            sample_contours = []
            for img in sample_imgs:
                # find contour
                contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                sample_contours.append(contours)
            self.contours_list.append(sample_contours)
        self.image_edged = []
        
    def FilterContour(self, min_area=50, min_w=10, min_h=10):
        for sample_contours in self.contours_list:
            sample_filtered_contours = []
            for contours in sample_contours :
                filtered_contours = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w*h
                    if not (area < min_area or w < min_w or h < min_h) :
                        filtered_contours.append(cnt)
                sample_filtered_contours.append(filtered_contours)
            self.filtered_contours_list.append(sample_filtered_contours)
        
    def CropByContour(self):
        for i in range(len(self.image_range)): # crop all removed background image by contour 
            sample_roi = []
            for j in range(len(self.image_range[i])):
                img = self.image_range[i][j]
                cnt = np.concatenate(self.filtered_contours_list[i][j], axis=0) # concate all remaining contour each image
                x, y, w, h = cv2.boundingRect(cnt)
                roi = img[y:y+h, x:x+w]
                sample_roi.append(roi)
            self.image_croped.append(sample_roi)
        self.image_range = []        
        
    def Resize(self, size=(172,172)):
        for sample_imgs in self.image_croped:
            sample_resize = []
            for img in sample_imgs :
                resized = cv2.resize(img, (size[0], size[1]))
                sample_resize.append(resized)
            self.image_resized.append(sample_resize)
        self.image_croped = []
        
    def SaveAllImage(self, RESIZED_FOLDER = "resized_tomato/"):
        if not os.path.exists(RESIZED_FOLDER) :
            os.mkdir(RESIZED_FOLDER)    
            
        try :
            shutil.rmtree(RESIZED_FOLDER)
            os.mkdir(RESIZED_FOLDER)
        except Exception as e:
            print("[ERROR] ", e)
            
        for i in range(len(self.image_resized)):
            # check if folder exist. if not, create that folder    
            folder_path = RESIZED_FOLDER + self.labels[i] + "/"
            if not os.path.exists(folder_path) :
                os.mkdir(folder_path)
                
            for j in range(len(self.image_resized[i])):
                # get image
                img = self.image_resized[i][j]

                if img is None :
                    continue

                # check if folder per sample is exist. if not, create that folder    
                sample_path = RESIZED_FOLDER + self.labels[i] + "/" + "sample_%03d" % i + "/"
                if not os.path.exists(sample_path) :
                    os.mkdir(sample_path)

                # save image
                file_name = "img_%03d.jpg" % j
                file_path = sample_path + file_name

                cv2.imwrite(file_path, img)
        self.image_resized = []
            
            
            
#############################################################################
#
#
# Feature Extraction
#
#
#############################################################################
class FeatureExtraction : 
    def __init__(self, PREPROCESSED_DATASET_FOLDER = "resized_tomato/"):
        self.labels = []
        self.image_list = []
        self.statistical_features = []
        self.glcm_matrix_list = []
        self.color_ch = ['b', 'g', 'r']
        self.glcm_feature_list = []
        self.texture_feature_labels = ['correlation', 'homogeneity', 'contrast', 'energy']
        self.PREPROCESSED_DATASET_FOLDER = PREPROCESSED_DATASET_FOLDER
        
    def ImageRead(self):
        for folder in os.listdir(self.PREPROCESSED_DATASET_FOLDER):
            for sample in os.listdir(os.path.join(self.PREPROCESSED_DATASET_FOLDER, folder)):
                sample_imgs = []
                for file in os.listdir(os.path.join(self.PREPROCESSED_DATASET_FOLDER, folder, sample)):
                    img = cv2.imread(os.path.join(self.PREPROCESSED_DATASET_FOLDER, folder, sample, file).replace("\\","/"))
                    sample_imgs.append(img)
                self.image_list.append(sample_imgs)
                self.labels.append(folder) # append label (name) of image
                
    def CalcStatisticalFeature(self):
        for sample  in self.image_list:
            sample_feature = []
            for img in sample : 
                feature_ch = {}
                for i in range(len(self.color_ch)):
                    feature_ch[self.color_ch[i]] = {
                        'mean' : img[:,:,i].mean(),
                        'std' : img[:,:,i].std(),
                        'skewness' : skew(img[:,:,i].reshape(-1))
                    }
                sample_feature.append(feature_ch)
            self.statistical_features.append(sample_feature)
            
    def CalcGLCMMatrix(self):
        for sample in self.image_list:
            sample_matrix = []
            for img in sample :
                matrix_ch = {}
                for i in range(len(self.color_ch)):
                    # grab r, g, b channel
                    img_ch = img[:,:,i]

                    # calculate GLCM 
                    glcm_img = greycomatrix(img_ch, 
                                        distances=[1],  # distance 1 pixel
                                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # angel 0, 45, 90, 135 degre
                                        levels=256, # number of grey-levels counted in 8 bit grayscale image
                                        symmetric=True, 
                                        normed=True)

                    matrix_ch[self.color_ch[i]] = glcm_img
                sample_matrix.append(matrix_ch)
            self.glcm_matrix_list.append(sample_matrix)
            
    def CalcGLCMTextureFeature(self):
        for sample_matrix in self.glcm_matrix_list:
            sample_feature = []
            for glcm_matrix in sample_matrix :
                feature_ch = {}
                for ch in self.color_ch:
                    feature_item = {}
                    for feature in self.texture_feature_labels:
                        out = greycoprops(glcm_matrix[ch], feature)[0]
                        feature_item[feature] = out.tolist()
                    feature_ch[ch] = feature_item
                sample_feature.append(feature_ch)
            self.glcm_feature_list.append(sample_feature)
            
#############################################################################
#
#
# Postprocessing
#
#
#############################################################################
class Postprocessing :
    def __init__ (self, statistical_features, glcm_feature_list, labels):
        self.X = []
        self.y = []
        self.X_train = []
        self.X_test = [] 
        self.y_train = []
        self.y_test = []
        self.statistical_features = statistical_features
        self.glcm_feature_list = glcm_feature_list
        self.labels = labels
        self.labels_name = []
        self.labels_vec = []
        self.test_size = 0.33
        
        
    def transformX(self):
        # transform statistical feature to 2D matrix
        x1 = []
        for sample in self.statistical_features : 
            sample_x = []
            for channel in sample :
                x = []
                for feature in list(channel.values()) :
                    data = list(feature.values())
                    x.extend(data)
                sample_x.extend(x)
            x1.append(sample_x)

        # transform GLCM feature to 2D matrix
        x2 = []
        for sample in self.glcm_feature_list: 
            sample_x = []
            for channel in sample :
                x = []
                for features in list(channel.values()):
                    item_feature = []
                    for item in list(features.values()) :
                        item_feature.extend(item)
                    x.extend(item_feature)
                sample_x.extend(x)
            x2.append(sample_x)
            
        # concate 2d Matrix
        x1 = np.array(x1).astype(np.float32)
        x2 = np.array(x2).astype(np.float32)
        self.X = np.concatenate((x1, x2), axis=1)
        print("X size :\n", self.X.shape)
        
    def transformY(self):
        
        enc = OneHotEncoder(handle_unknown='ignore')
        y = np.array(self.labels).reshape(-1, 1)
        enc.fit(y)
        self.labels_name = enc.categories_[0]
        print("labels_name :\n", self.labels_name)
        self.labels_vec = enc.transform(y).toarray()
        print("labels_vec unique :\n", np.unique(self.labels_vec))
        self.y = np.array(self.labels_vec).astype(np.float32)

        print("y size :\n", self.y.shape)
        
    def splitData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                    self.X, self.y, test_size=self.test_size, random_state=42)
        print("Split size :\n", self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        
#############################################################################
#
#
# Train Model
#
#
#############################################################################
class TrainingMLP:
    def __init__(self, X_train, y_train, X_test, y_test, labels_vec, labels_name, max_iteartion=100, min_accuracy=0.9, input_dim=2):
        self.mlp = cv2.ml.ANN_MLP_create()
        
        input_dim = X_train.shape[1] if X_train is not None else input_dim
        output_dim = len(labels_name)
        print("dim", input_dim, output_dim)
        network_layer = np.array([input_dim, 32, output_dim]).astype(np.uint0)
        self.mlp.setLayerSizes(network_layer)
        self.mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        self.mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        
        # set term criteria : maximum 100 iteration or stop when achieve 90% accuracy
        self.mlp.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iteartion, min_accuracy))
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = []
        self.labels_vec = labels_vec
        self.labels_name = labels_name
        
        self.base_path = os.path.expanduser('~/Tomato Grading Systems')
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.model_name = os.path.join(self.base_path, 'klasifikasi_tomat_mlp_model.xml')
        
    def train(self):
        self.mlp.train(self.X_train, cv2.ml.ROW_SAMPLE, self.y_train)
        self.mlp.save(self.model_name)
        
    def validate(self, title='Confusion matrix - Klasifikasi Tomat'):
        self.mlp.load(self.model_name)
        self.y_pred = self.mlp.predict(self.X_test)[1]
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1), labels=np.unique(self.labels_vec))
        np.set_printoptions(precision=2)


        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix(cnf_matrix, classes=self.labels_name, normalize=False,
                              title=title)
        
        # Print Classification Report
        print(classification_report(self.y_test.argmax(axis=1), 
                                    self.y_pred.argmax(axis=1), 
                                    target_names=self.labels_name))
        
        report = classification_report(self.y_test.argmax(axis=1), 
                                    self.y_pred.argmax(axis=1), 
                                    target_names=self.labels_name)

        report_path = os.path.join(self.base_path, 'Report %s.txt' % title)
        with open(report_path, "w") as text_file:
            text_file.write(report)
        
        
    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(5, 4))

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
        
        cm_path = os.path.join(self.base_path, '%s.png' % title)
        plt.savefig(cm_path, bbox_inches='tight')
        
        plt.close()
    
    
#############################################################################
#
#
# Prediction
#
#
#############################################################################
class Prediction:
    def __init__(self, labels_name, model_name = os.path.expanduser('~/Tomato Grading Systems/klasifikasi_tomat_mlp_model.xml')):
        self.mlp = cv2.ml.ANN_MLP_load(model_name)
        self.labels_name = labels_name
        self.output = ""
        
    def predict(self, X):
        y = self.mlp.predict(X)[1]

        y_proba = max(min(y.max()*100, 100), 0)
        y_label = self.labels_name[y.argmax(axis=1)[0]]
        self.output  = "%s (%d%%)" % (y_label, y_proba)
        print("Predicted Label : %s" % self.output)