import os
import cv2
import sys
import numpy as np
import scipy
from tqdm import tqdm
import motmetrics as mm
from collections import defaultdict

import utils
from sort import DeepSORT, linear_assignment
from datasets import AICityDatasetDetector


STORED_DETECTIONS_NAME = "dets_all.txt"
STORED_FEATURES_NAME = "features_all.txt"
REMOVE_SMALL_BOXES = True
THRESHOLD_SMALL = 20*20


CAM_ORDERS = {  # Videos early in time first
    'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
    'S03': ['c013', 'c014', 'c012', 'c011', 'c015', 'c010'],
    'S04': ['c016', 'c017', 'c020', 'c019', 'c018', 'c021',
            'c023', 'c022', 'c025', 'c024', 'c026', 'c028',
            'c030', 'c031', 'c032', 'c029', 'c034', 'c033',
            'c035', 'c037', 'c036', 'c038', 'c039', 'c040']
}


def task2(sequence_path, dets_sequence_path, max_frames_skip, min_iou, sim_thr):
    """
    Given precomputed detections and feature vectors for the detections, this performs MTMC tracking,
    and evaluates this.
    """

    # Read all the necessary data, per camera
    videos = []
    vid_lengths = []
    gt_datasets = []
    detections = []
    for camera in CAM_ORDERS[os.path.basename(sequence_path)]:
        # Save videos capture
        video_path = os.path.join(sequence_path, camera, 'vdo.avi')
        cap = cv2.VideoCapture(video_path)
        vid_lengths.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        videos.append(cap)

        # Read detections file (to save computation, tracks are not postprocessed: online)
        det_path = os.path.join(dets_sequence_path, camera + '.npy')
        model_detections = np.load(det_path)
        detections.append(model_detections)

        # Read gt data
        dataset_path = os.path.dirname(sequence_path)
        sequences = {os.path.basename(sequence_path): [camera]}
        dataset = AICityDatasetDetector(dataset_path, sequences)
        gt_datasets.append(dataset)


    # Iterate over videos, frame by frame
    video_finished = [False for _ in videos]
    frame_number = 0
    id_map = {}  # for re-id on multicameras
    vid_ids = [[] for _ in videos]
    vid_fvs = [[] for _ in videos]
    tracks_centers = defaultdict(list)
    acc = mm.MOTAccumulator()
    with tqdm(total=max(vid_lengths), file=sys.stdout) as pbar:
        while not all(video_finished):
            for idx, cap in enumerate(videos):
                if not video_finished[idx]:
                    ret, img = cap.read()
                    video_finished[idx] = not ret
                    if not ret:
                        continue

                    # Read precomputed detections
                    frame_mask = detections[idx][:,0] == frame_number
                    ids = detections[idx][frame_mask, 1].astype(np.int)
                    dets = detections[idx][frame_mask, 2:6]
                    feature_vectors = detections[idx][frame_mask, 6:][:,np.newaxis,:]

                    # Separate new detections from existing ones
                    new_id, new_fvs, det_idxs = [], [], []
                    for det_idx, (id, fv) in enumerate(zip(ids, feature_vectors)):
                        if id not in id_map:
                            new_id.append(id)
                            new_fvs.append(fv)
                            det_idxs.append(det_idx)
                        else:
                            tracks_centers[id].append([(dets[det_idx,0]+dets[det_idx,2])/2, (dets[det_idx,1]+dets[det_idx,3])/2])
                            ids[det_idx] = id_map[id]
                            if len(tracks_centers[id]) > 10:
                                std = np.std(tracks_centers[id], 0)
                                static_thr = 100
                                if std[0] < static_thr and std[1] < static_thr:
                                    ids[det_idx] = -1  # mark static objects
                            #if int(id) in vid_ids[idx]:
                            #    fv_idx = vid_ids[idx].index(int(id))
                            #    vid_fvs[idx][fv_idx] = 0.6*np.array(fv) + 0.4*vid_fvs[idx][fv_idx]

                    if sum([len(ids) for ids in vid_ids]) == 0:
                        vid_ids[idx].extend(new_id)
                        vid_fvs[idx].extend(new_fvs)
                        continue

                    # Hungarian algorithm between new detections and existing dets. on other videos
                    if len(new_id) > 0:
                        v1 = np.concatenate(new_fvs, 0)
                        v2 = np.concatenate([x for i,x in enumerate(vid_fvs) if i != idx and x], 0)
                        v2 = v2.squeeze(1)
                        v2_ids = np.concatenate([x for i,x in enumerate(vid_ids) if i != idx and x], 0)
                        similarity_mat = v1 @ v2.T
                        matched = linear_assignment(-similarity_mat)

                        unmatched_ids = []
                        unmatched_fvs = []
                        for m in range(len(v1)):
                            if m not in matched[:,0]:
                                unmatched_ids.append(new_id[m])
                                unmatched_fvs.append(new_fvs[m])

                        # Filter out matches with low similarity
                        for match in matched:
                            if similarity_mat[match[0], match[1]] < sim_thr:  # TODO tune this
                                unmatched_ids.append(new_id[match[0]])
                                unmatched_fvs.append(new_fvs[match[0]])
                            else:
                                id_map[new_id[match[0]]] = v2_ids[match[1]]
                                ids[det_idxs[match[0]]] = v2_ids[match[1]]


                        # Add new ids which are not matched
                        vid_ids[idx].extend(unmatched_ids)
                        vid_fvs[idx].extend(unmatched_fvs)
                        for id in unmatched_ids:
                            id_map[id] = id

                    _, gt = gt_datasets[idx][frame_number]
                    if gt:
                        if "boxes" in list(gt.keys()):
                            gt_boxes = gt['boxes']
                            gt_this_frame = [int(x) for x in gt['track_id']]
                            dets = dets[ids != -1]
                            if len(dets) > 0:
                                dets_centers = np.vstack([(dets[:,0]+dets[:,2])/2, (dets[:,1]+dets[:,3])/2]).T
                                gt_centers = np.vstack([(gt_boxes[:,0]+gt_boxes[:,2])/2, (gt_boxes[:,1]+gt_boxes[:,3])/2]).T
                                dists = scipy.spatial.distance_matrix(dets_centers, gt_centers).T.tolist()
                                acc.update(
                                    gt_this_frame,
                                    [id for id in ids if id != -1],
                                    dists,
                                    frame_number*100 + idx,  # unique frameid between cameras
                                )                            # https://github.com/cheind/py-motmetrics/issues/34
                            else:
                                acc.update(
                                    gt_this_frame,
                                    [], [],
                                    frame_number*100 + idx,
                                )

            frame_number += 1
            pbar.update(1)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary[['idf1','idp','idr','precision','recall']])


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-s",
                        dest="sequence_path",
                        required=True,
                        type=str,
                        help="Path to the sequence folder (containing the videos and cameras)")
    parser.add_argument("-t",
                        dest="path_tracklets",
                        required=True,
                        type=str,
                        help="Path to the folder where the generated tracklets are.")
    parser.add_argument("-m",
                        dest="max_frames_skip",
                        required=False,
                        default=20,
                        type=int)
    parser.add_argument("-u",
                        dest="min_iou",
                        required=False,
                        default=20,
                        type=int)
    parser.add_argument("-i",
                        dest="sim_thr",
                        required=False,
                        default=0.4,
                        type=float,
                        help="Similarity threshold.")
    args = parser.parse_args()
    return args.sequence_path, args.path_tracklets, args.max_frames_skip, args.min_iou, args.sim_thr
    


if __name__ == "__main__":
    sequence_path, path_tracklets, max_frames_skip, min_iou, sim_thr = parse_arguments()

    task2(
        sequence_path=sequence_path,
        dets_sequence_path=path_tracklets,
        max_frames_skip=max_frames_skip,
        min_iou=min_iou,
        sim_thr=sim_thr)

