import cv2
import numpy as np

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
    def __init__(self, video_path, alpha=4, color_format="grayscale", height=1080, width=1920, num_frames_training=510):
        super().__init__(video_path, color_format=color_format, height=height, width=width, num_frames_training=num_frames_training)
        self.alpha = alpha
        
    def fit(self):
        print("Fitting model...")
        self.cap = cv2.VideoCapture(self.video_path)
        # Compute mean first
        
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
        
        # Then compute std
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
        self.mean = mean_img
        self.std = np.sqrt(std_img)
        
    def infer(self, img):
        img = self.convert_color_space(img).astype(np.float32)
        return (np.abs(img-self.mean) >= self.alpha * (self.std + 2)).astype(np.uint8) * 255
    
    