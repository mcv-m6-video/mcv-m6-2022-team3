import cv2
import numpy as np
import os
from boxes_detection import obtain_bboxes
from morphology_utils import morphological_filtering

class BackgroundModel():
    def __init__(self, video_path, roi_path, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        self.video_path = video_path
        self.frame_count = num_frames_training
        self.training_data = None
        self.color_format = color_format
        self.height = height
        self.width = width
        self.roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY) > 0
        
    def fit(self):
        pass
    
    def convert_color_space(self, img):
        if self.color_format in ("RGB", "BGR"):
            return img
        elif self.color_format == "grayscale":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.color_format in ("HSV", "HS"):
            img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if self.color_format == "HS":
                return img_conv[:,:,(0,1)]
            return img_conv
        elif self.color_format in ("LAB", "Lab", "AB", "ab"):
            img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            if self.color_format in ("AB", "ab"):
                return img_conv[:,:,(1,2)]
            return img_conv
        elif self.color_format in ("YUV", "YCrCb", "UV", "CrCb"):
            img_conv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            if self.color_format in ("UV", "CbCr"):
                return img_conv[:,:,(1,2)]
            return img_conv

    
class GaussianStaticModel(BackgroundModel):
    def __init__(self, video_path, roi_path, alpha=4, debug_ops=True, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        super().__init__(video_path, roi_path, color_format=color_format, height=height, width=width, num_frames_training=num_frames_training)
        self.alpha = alpha
        self.bg_model_name = "bg_gaussian_data"
        self.debug_ops = False
        self.min_h, self.min_w, self.max_h, self.max_w = 50, 50, 600, 800
        
        if not os.path.exists(self.bg_model_name):
            os.mkdir(self.bg_model_name)
        
    def fit(self):
        print("Fitting model...")
        self.cap = cv2.VideoCapture(self.video_path)
        # Compute mean first
        mean_img_path = os.path.join(self.bg_model_name, "mean.npy")
        if os.path.exists(mean_img_path):
            mean_img = np.load(mean_img_path)
        else:
            if self.color_format == "grayscale":
                mean_img = np.zeros([self.height, self.width], dtype=np.float32)

            elif self.color_format in ("HS", "AB", "ab", "UV", "CbCr"):
                mean_img = np.zeros([self.height, self.width, 2], dtype=np.float32)
            
            else:
                mean_img = np.zeros([self.height, self.width, 3], dtype=np.float32)

            for i in range(self.frame_count):
                ret, img = self.cap.read()
                img = self.convert_color_space(img).astype(np.float32)
                mean_img = mean_img + img/self.frame_count
            
            self.cap.release()
            np.save(mean_img_path, mean_img)
        
        # Then compute std
        std_img_path = os.path.join(self.bg_model_name, "std.npy")
        if os.path.exists(std_img_path):
            std_img = np.load(std_img_path)
        else:  
            self.cap = cv2.VideoCapture(self.video_path)
            if self.color_format == "grayscale":
                std_img = np.zeros([self.height, self.width], dtype=np.float32)

            elif self.color_format in ("HS", "AB", "ab", "UV", "CbCr"):
                std_img = np.zeros([self.height, self.width, 2], dtype=np.float32)
            
            else:
                std_img = np.zeros([self.height, self.width, 3], dtype=np.float32)
            
            for i in range(self.frame_count):
                ret, img = self.cap.read()
                img = self.convert_color_space(img).astype(np.float32)
                std_img = std_img + ((mean_img - img)**2) / (self.frame_count-1)
            
            self.cap.release()
            std_img = np.sqrt(std_img)
            np.save(std_img_path, std_img)
        
        self.mean = mean_img
        self.std = std_img
        
    def infer(self, img):
        if self.debug_ops:
            cv2.imshow("orig_img", img); cv2.waitKey(0)
        img = self.convert_color_space(img).astype(np.float32)
        fg_gauss_model = (np.abs(img-self.mean) >= self.alpha * (self.std + 2)) #.astype(np.uint8) * 255
        if (len(fg_gauss_model.shape)==3) and (fg_gauss_model.shape[2]==2):
            fg_gauss_model = np.logical_and(fg_gauss_model[:,:,0], fg_gauss_model[:,:,1])
        elif (len(fg_gauss_model.shape)==3) and (fg_gauss_model.shape[2]==3):
            fg_gauss_model = np.logical_and(np.logical_and(fg_gauss_model[:,:,0], fg_gauss_model[:,:,1]), fg_gauss_model[:,:,2])
        fg_gauss_model = fg_gauss_model * self.roi
        if self.debug_ops:
            cv2.imshow("first_fg_img", fg_gauss_model.astype(np.uint8)*255); cv2.waitKey(0)
        filtered_fg = morphological_filtering(fg_gauss_model)
        detections = obtain_bboxes(filtered_fg, min_h=self.min_h, max_h=self.max_h, min_w=self.min_w, max_w=self.max_w)
        return detections, filtered_fg.astype(np.uint8)*255
    
class GaussianDynamicModel(GaussianStaticModel):
    def __init__(self, video_path, roi_path, rho=0.02, alpha=4, debug_ops=True, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        super().__init__(video_path, roi_path, alpha=alpha, debug_ops=debug_ops, color_format=color_format, height=height, width=width, num_frames_training=num_frames_training)
        self.rho = rho

    def infer(self, img):
        # Update background model
        img = self.convert_color_space(img).astype(np.float32)
        fg_gauss_model = (np.abs(img-self.mean) >= self.alpha * (self.std + 2)) #.astype(np.uint8) * 255
        if (len(fg_gauss_model.shape)==3) and (fg_gauss_model.shape[2]==2):
            fg_gauss_model = np.logical_and(fg_gauss_model[:,:,0], fg_gauss_model[:,:,1])
        elif (len(fg_gauss_model.shape)==3) and (fg_gauss_model.shape[2]==3):
            fg_gauss_model = np.logical_and(np.logical_and(fg_gauss_model[:,:,0], fg_gauss_model[:,:,1]), fg_gauss_model[:,:,2])
        fg_gauss_model = fg_gauss_model * self.roi
        bg = ~fg_gauss_model
        
        self.mean[bg] = self.rho * img[bg] + (1-self.rho) * self.mean[bg]
        self.std[bg] = np.sqrt(self.rho * np.power(img[bg] - self.mean[bg], 2) + (1-self.rho) * np.power(self.std[bg], 2))
        
        filtered_fg = morphological_filtering(fg_gauss_model)
        detections = obtain_bboxes(filtered_fg)
        return detections, filtered_fg.astype(np.uint8)*255


class BackgroundModelCV2(BackgroundModel):
    def __init__(self, video_path, roi_path, background_substractor, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        super().__init__(video_path, roi_path, color_format=None, height=height, width=width, num_frames_training=num_frames_training)
        self.bgsegm = background_substractor
        self.min_h, self.min_w, self.max_h, self.max_w = 50, 50, 600, 800

    def fit(self):

        print("Fitting model...")
        self.cap = cv2.VideoCapture(self.video_path)
        for i in range(self.frame_count):
            ret, img = self.cap.read()
            self.bgsegm.apply(img)
        self.cap.release()

    def infer(self, img):

        mask = self.bgsegm.apply(img)
        mask = mask * self.roi

        filtered_fg = morphological_filtering(mask)
        detections = obtain_bboxes(filtered_fg, min_h=self.min_h, max_h=self.max_h, min_w=self.min_w, max_w=self.max_w)

        return detections, filtered_fg.astype(np.uint8)*255