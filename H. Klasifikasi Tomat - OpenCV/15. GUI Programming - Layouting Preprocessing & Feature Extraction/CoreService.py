import os
import cv2
import shutil
import numpy as np

from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops

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
            for file in os.listdir(os.path.join(self.DATASET_FOLDER, folder)):
                img = cv2.imread(os.path.join(self.DATASET_FOLDER, folder, file))
                self.image_list.append(img)
                self.labels.append(folder) # append label (name) of image
                            
    def RangeTresholding(self):
        for img in self.image_list :          
            # convert to hsv
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # apply range thresholding
            mask_green = cv2.inRange(hsv.copy(), self.lower_green, self.upper_green)
            mask_red = cv2.inRange(hsv.copy(), self.lower_red, self.upper_red)
            mask_yellow = cv2.inRange(hsv.copy(), self.lower_yellow, self.upper_yellow)
            mask_black = cv2.inRange(hsv.copy(), self.lower_black, self.upper_black)

            mask = mask_green + mask_red + mask_yellow + mask_black 
            res = cv2.bitwise_and(img, img, mask= mask)
            self.image_range.append(res)
            
    def EdgeDetection(self):
        for img in self.image_range :
            edged = cv2.Canny(img, 200, 210)
            self.image_edged.append(edged)
            
    def FindContour(self):
        for img in self.image_edged:
            # find contour
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.contours_list.append(contours)
        
    def FilterContour(self, min_area=50, min_w=10, min_h=10):
        for contours in self.contours_list:
            filtered_contours = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w*h
                if not (area < min_area or w < min_w or h < min_h) :
                    filtered_contours.append(cnt)
            self.filtered_contours_list.append(filtered_contours)

    def CropByContour(self):
        for i in range(len(self.image_range)): # crop all removed background image by contour 
            img = self.image_range[i]
            cnt = np.concatenate(self.filtered_contours_list[i], axis=0) # concate all remaining contour each image
            x, y, w, h = cv2.boundingRect(cnt)
            roi = img[y:y+h, x:x+w]
            self.image_croped.append(roi)
                
    def Resize(self, size=(172,172)):
        for img in self.image_croped:
            resized = cv2.resize(img, (size[0], size[1]))
            self.image_resized.append(resized)
            
    def SaveAllImage(self, RESIZED_FOLDER = "resized_tomato/"):        
        if not os.path.exists(RESIZED_FOLDER) :
            os.mkdir(RESIZED_FOLDER)
            
        try :
            shutil.rmtree(RESIZED_FOLDER)
            os.mkdir(RESIZED_FOLDER)
        except Exception as e:
            print("[ERROR] ", e)

        for i in range(len(self.image_resized)):

            # get image
            img = self.image_resized[i]

            if img is None :
                continue
                
            # check if folder exist. if not, create that folder    
            folder_path = RESIZED_FOLDER + self.labels[i] + "/"
            if not os.path.exists(folder_path) :
                os.mkdir(folder_path)

            # save image
            file_name = self.labels[i] + "_%03d.jpg" % i
            file_path = os.path.join(RESIZED_FOLDER,self.labels[i],file_name)

            cv2.imwrite(file_path, img)
            
            
            
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
            for file in os.listdir(os.path.join(self.PREPROCESSED_DATASET_FOLDER, folder)):
                img = cv2.imread(os.path.join(self.PREPROCESSED_DATASET_FOLDER, folder, file))
                self.image_list.append(img)
                self.labels.append(folder) # append label (name) of image
                
    def CalcStatisticalFeature(self):
        for img  in self.image_list:
            feature_ch = {}
            for i in range(len(self.color_ch)):
                feature_ch[self.color_ch[i]] = {
                    'mean' : img[:,:,i].mean(),
                    'std' : img[:,:,i].std(),
                    'skewness' : skew(img[:,:,i].reshape(-1))
                }
            
            self.statistical_features.append(feature_ch)
            
    def CalcGLCMMatrix(self):
        for img in self.image_list:
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
            self.glcm_matrix_list.append(matrix_ch)
            
    def CalcGLCMTextureFeature(self):
        for glcm_matrix in self.glcm_matrix_list:
            feature_ch = {}
            for ch in self.color_ch:
                feature_item = {}
                for feature in self.texture_feature_labels:
                    out = greycoprops(glcm_matrix[ch], feature)[0]
                    feature_item[feature] = out.tolist()
                feature_ch[ch] = feature_item
            self.glcm_feature_list.append(feature_ch)