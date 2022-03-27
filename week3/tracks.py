import random
import numpy as np
from copy import deepcopy
from utils import get_rect_iou

class Track(object):
    def __init__(self, id, first_detection, first_frame_number, camera=0):
        self.id = id
        self.detections = [first_detection]
        self.frame_numbers = [first_frame_number]
        self.visualization_color = (int(random.random() * 256), int(random.random() * 256), int(random.random() * 256))
        self.terminated = False
        self.camera = camera

    def get_track_boxes(self):
        return self.detections

    def add_detection(self, detection, frame_number):
        self.detections.append(detection)
        self.frame_numbers.append(frame_number)

    def last_detection(self):
        return self.detections[-1], self.frame_numbers[-1]

class TrackHandler:
    def __init__(self):
        self.track_id_counter = 0
        
    def update_tracks(self, frame_detections, frame_number):
        pass
    
    def create_new_track(self):
        pass
        
        
class TrackHandlerOverlap(TrackHandler):
    def __init__(self, max_frame_skip=0, min_iou=0.4):
        super().__init__()
        self.max_frame_skip = max_frame_skip
        self.min_iou = min_iou
        self.live_tracks = []
        self.terminated_tracks = []
    
    def create_new_track(self, first_detection, first_frame_number):
        new_track = Track(self.track_id_counter, first_detection, first_frame_number)
        self.track_id_counter += 1
        return new_track
    
    def update_tracks(self, frame_detections, frame_number):
        new_live_tracks = []
        
        # Update live / terminated tracks 
        for track in self.live_tracks:
            _, last_frame_number = track.last_detection()
            if abs(frame_number - last_frame_number) <= self.max_frame_skip+1:
                new_live_tracks.append(track)
            else:
                self.terminated_tracks.append(track)
        
        self.live_tracks = new_live_tracks
        new_tracks = []
        # Update tracks
        for detection in frame_detections:
            max_iou, best_idx = 0, -1
            for idx in range(len(self.live_tracks)):
                last_track_detection, last_frame_number = self.live_tracks[idx].last_detection()
                #if frame_number - last_frame_number <= self.max_frame_skip+1:
                iou = get_rect_iou(detection, last_track_detection)
                if iou > self.min_iou and iou > max_iou:
                    max_iou = iou
                    best_idx = idx
                
            # If match : update track detection
            if best_idx != -1:
                self.live_tracks[best_idx].add_detection(detection, frame_number)
            else: # Otherwise create new track
                new_track = self.create_new_track(detection, frame_number)
                new_tracks.append(new_track)
                
        # Add new tracks to live tracks
        for new_track in new_tracks:
            self.live_tracks.append(new_track)
        

    