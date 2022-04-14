"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import random
import argparse
from filterpy.kalman import KalmanFilter
from visual_features import ReIDNetwork

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.visualization_color = (int(random.random() * 256), int(random.random() * 256), int(random.random() * 256))
    self.history = []
    self.frames = []
    self.threshold = 10
    
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self, frame_number=0):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    self.frames.append(frame_number)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)
  
  def is_static(self):
    """
    Analyze if this track is static based on initial detections and final detections
    """
    if len(self.history) > 10:
      history = self.history[:10]+self.history[-10:]
      track_history = np.concatenate(history)
      centers_std = np.std((track_history[:, [0, 1]] + track_history[:, [2, 3]]) / 2, 0)
      _is_static = centers_std[0] < self.threshold and centers_std[1] < self.threshold
      return _is_static
    else:
      return False


class IoUTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.time_since_update = 0
    self.id = IoUTracker.count
    IoUTracker.count += 1
    self.visualization_color = (int(random.random() * 256), int(random.random() * 256), int(random.random() * 256))
    self.history = [bbox]
    self.frames = []
    self.threshold = 10
    
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.history.append(bbox)
    
  def predict(self, frame_number=0):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    return [self.history[-1][:-1]]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return [self.history[-1][:-1]]
  
  def is_static(self):
    """
    Analyze if this track is static based on initial detections and final detections
    """
    if len(self.history) > 10:
      history = self.history[:10]+self.history[-10:]
      track_history = np.concatenate(history)
      if track_history.ndim == 1:
        track_history = track_history.reshape(len(track_history)//4, 4)
      centers_std = np.std((track_history[:, [0, 1]] + track_history[:, [2, 3]]) / 2, 0)
      _is_static = centers_std[0] < self.threshold and centers_std[1] < self.threshold
      return _is_static
    else:
      return False


def associate_detections_to_trackers(detections,trackers,feature_vectors=None, new_feature_vectors=None, iou_threshold = 0.3, beta=0.8):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)
  sim_matrix = iou_matrix
  if not feature_vectors is None:
    if len(feature_vectors) > 0:
      feature_vectors, new_feature_vectors = np.concatenate(feature_vectors, 0), np.concatenate(new_feature_vectors, 0)
      reid_matrix = new_feature_vectors@feature_vectors.transpose(1, 0)
      sim_matrix = sim_matrix * (1-beta) + reid_matrix * beta

  if min(sim_matrix.shape) > 0:
    a = (sim_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-sim_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(sim_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, online_filtering=False, max_age=1, min_hits=3, iou_threshold=0.3, tracker_type="IoU"):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.dead_trackers = []
    self.frame_count = 0
    self.online_filtering = online_filtering
    self.tracker_instantiation = IoUTracker if tracker_type == "IoU" else KalmanBoxTracker 

  def update(self, image=None, dets=np.empty((0, 5)), frame_number=0):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict(frame_number)[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      dead_track = self.trackers.pop(t)
      self.dead_trackers.append(dead_track)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, iou_threshold=self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = self.tracker_instantiation(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits): 
          if self.online_filtering:
            if not trk.is_static():
              ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          else:
            ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            
        i -= 1
        # remove dead tracklet if has exceeded max age
        if(trk.time_since_update > self.max_age):
          dead_track = self.trackers.pop(i)
          self.dead_trackers.append(dead_track)
          
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
  
  def postprocess_trackers(self, threshold=100):
    # Remove static tracks + ReID? -> Fix fragmented tracks with visual model maybe
    all_tracks = self.dead_trackers + self.trackers
    non_static_tracks = []
    for track in all_tracks:
      track_history = np.concatenate(track.history)
      # track_history = track_history.reshape((len(track_history)//4, 4))
      centers_std = np.std((track_history[:, [0, 1]] + track_history[:, [2, 3]]) / 2, 0)
      if centers_std[0] > threshold or centers_std[1] > threshold:
        non_static_tracks.append(track)
    
    self.non_static_tracks = non_static_tracks
  
  def get_frame_at_x(self, frame_number):
    # Get frames at the specified frame number saved on the frames
    detections = []
    if self.non_static_tracks is None:
      self.non_static_tracks = self.dead_trackers + self.trackers
    
    for track in self.non_static_tracks:
      track_frames = np.array(track.frames)
      track_history = np.concatenate(track.history)
      # track_history = track_history.reshape((len(track_history)//4, 4))
      bbox = track_history[track_frames == frame_number]
      if len(bbox) > 0:
        bbox = bbox[0]
        track_id = track.id+1
        detections.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])

    return np.array(detections)
  
  def get_frames_before_x(self, frame_number):
    # Get frames at the specified frame number saved on the frames
    all_detections = []
    if self.non_static_tracks is None:
      self.non_static_tracks = self.dead_trackers + self.trackers
    
    for track in self.non_static_tracks:
      detections = []
      track_frames = np.array(track.frames)
      track_history = np.concatenate(track.history)
      # track_history = track_history.reshape((len(track_history)//4, 4))
      bboxes = track_history[track_frames <= frame_number]
      if len(bboxes) > 0:
        for bbox in bboxes:
          track_id = track.id+1
          detections.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
      
      if len(detections) > 0:
        all_detections.append(np.array(detections))
        
    return all_detections
  
class DeepSORT(Sort):
  def __init__(self, online_filtering=False, max_age=1, min_hits=3, iou_threshold=0.3):
    super().__init__(online_filtering=online_filtering, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    self.reid_model = ReIDNetwork()
    
  def update(self, image=None, dets=np.empty((0, 5)), frame_number=0):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    feature_vectors = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict(frame_number)[0]
      feature_vectors.append(self.trackers[t].feature_vector)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      dead_track = self.trackers.pop(t)
      self.dead_trackers.append(dead_track)
    new_feature_vectors = []
    for det in dets:
      det = det.astype(np.int)
      x1,y1,x2,y2,_ = det
      car_patch = image[y1:y2, x1:x2]
      feature_vector = self.reid_model.extract_features(car_patch)
      new_feature_vectors.append(feature_vector)
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, feature_vectors=feature_vectors, new_feature_vectors=new_feature_vectors, iou_threshold=self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      x1,y1,x2,y2,_ = dets[m[0], :].astype(np.int)
      car_patch = image[y1:y2, x1:x2]
      self.trackers[m[1]].update(dets[m[0], :], new_feature_vectors[m[0]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        x1,y1,x2,y2,_ = dets[i, :].astype(np.int)
        car_patch = image[y1:y2, x1:x2]
        feature_vector = self.reid_model.extract_features(car_patch)
        trk = KalmanWithFeatures(dets[i,:], new_feature_vectors[i])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits): 
          if self.online_filtering:
            if not trk.is_static():
              ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          else:
            ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            
        i -= 1
        # remove dead tracklet if has exceeded max age
        if(trk.time_since_update > self.max_age):
          dead_track = self.trackers.pop(i)
          self.dead_trackers.append(dead_track)
          
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
  
class KalmanWithFeatures(KalmanBoxTracker):
  def __init__(self, bbox, feature_vector, alpha=0.7):
    super().__init__(bbox)
    self.feature_vector = feature_vector
    self.alpha = alpha
    
  def update(self, bbox, new_feature_vector):
    # Here update bbox and feature_vector
    super().update(bbox)
    # update feature_vector
    self.feature_vector = self.alpha * new_feature_vector + (1-self.alpha) * self.feature_vector
    self.feature_vector = self.feature_vector / np.linalg.norm(self.feature_vector, ord=2)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")