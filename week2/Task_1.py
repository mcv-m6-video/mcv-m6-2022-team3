import cv2
from cv2 import cvtColor
import numpy as np

video_path = "/home/aszummer/Documents/MCV/M6/mcv-m6-2022-team3/lab1-data/AICity_data/AICity_data/train/S03/c010/vdo.avi"

cap = cv2.VideoCapture(video_path)
frame_num = 0
ret, img = cap.read()
frame_25per = 510
training_data = np.zeros([1080,1920,510],dtype=np.uint8)
mean_data = np.zeros([1080,1920], dtype=np.uint8)
std_data = np.zeros([1080,1920], dtype=np.uint8)

mean_data_exist = cv2.imread('mean.png',0)
std_data_exist = cv2.imread('std.png',0)

display = True
wait_time = 1

while(True):
    ret, img = cap.read()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if frame_num < frame_25per:
        if mean_data_exist is None:
            print(f'Calculating frame {frame_num}')
            training_data[:,:,frame_num] = frame
    
    if frame_num == frame_25per:
        if mean_data_exist is not None:
            print('loaded existing')
            mean_data = mean_data_exist
            std_data = std_data_exist
        else:
            mean_data = np.mean(training_data[:,:,:],axis=2)
            std_data = np.std(training_data[:,:,:],axis=2)
            cv2.imwrite('mean.png', mean_data)
            cv2.imwrite('std.png', std_data)

    if frame_num > frame_25per:
        alpha = 4
        mask = abs(frame - mean_data) > alpha*(std_data+2)
        frame = np.ones_like(frame)*mask
        frame = frame*255


        # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel=(500,500))
        # frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel=(500,500))

    if display:
        display_frame = cv2.resize(frame, tuple(np.int0(0.5*np.array(img.shape[:2][::-1]))))
        # display_frame = frame
        cv2.imshow('frame',display_frame)
        k = cv2.waitKey(wait_time)
        if k == ord('q'):
            break
        elif k == ord('p'):
            wait_time = int(not(bool(wait_time)))

    frame_num += 1;
    
    if not ret:
        break

cap.release()
cv2.destroyAllWindows()