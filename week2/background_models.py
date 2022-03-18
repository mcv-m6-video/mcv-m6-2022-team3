import cv2
import numpy as np
import os
from boxes_detection import obtain_bboxes

class BackgroundModel():
    def __init__(self, video_path, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        self.video_path = video_path
        self.frame_count = num_frames_training
        self.training_data = None
        self.color_format = color_format
        self.height = height
        self.width = width
        
    def fit(self):
        pass
    
    def convert_color_space(self, img):
        if self.color_format == "RGB":
            return img
        if self.color_format == "grayscale":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.color_format == "RGB":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #...

    
class GaussianStaticModel(BackgroundModel):
    def __init__(self, video_path, alpha=4, debug_ops=True, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        super().__init__(video_path, color_format=color_format, height=height, width=width, num_frames_training=num_frames_training)
        self.alpha = alpha
        self.bg_model_name = "bg_gaussian_data"
        self.debug_ops = False
        
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
                for i in range(self.frame_count):
                    ret, img = self.cap.read()
                    img = self.convert_color_space(img).astype(np.float32)
                    mean_img = mean_img + img/self.frame_count
            
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
            
                for i in range(self.frame_count):
                    ret, img = self.cap.read()
                    img = self.convert_color_space(img).astype(np.float32)
                    std_img = std_img + img/self.frame_count
            
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
        if self.debug_ops:
            cv2.imshow("first_fg_img", fg_gauss_model.astype(np.uint8)*255); cv2.waitKey(0)
        from morphology_utils import morphological_filtering
        filtered_fg = morphological_filtering(fg_gauss_model, self.debug_ops)
        detections = obtain_bboxes(filtered_fg)
        return detections, filtered_fg.astype(np.uint8)*255
    