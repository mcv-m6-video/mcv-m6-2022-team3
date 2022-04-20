import os
import cv2
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

import utils
from sort import DeepSORT, linear_assignment
from datasets import AICityDatasetDetector


STORED_DETECTIONS_NAME = "dets_all.txt"
STORED_FEATURES_NAME = "features_all.txt"


def task2(sequence_path, dets_sequence_path, max_frames_skip, min_iou, det_thr, sim_thr):
    """
    Given precomputed detections and feature vectors for the detections, this performs MTMC tracking,
    and evaluates this.
    """

    # Read all the necessary data, per camera
    videos = []
    vid_lengths = []
    trackers = []
    gt_datasets = []
    detections = []
    features = []
    for camera in sorted(os.listdir(sequence_path)):
        # Save videos capture
        video_path = os.path.join(sequence_path, camera, 'vdo.avi')
        cap = cv2.VideoCapture(video_path)
        vid_lengths.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        videos.append(cap)

        # Create tracker objects
        track_handler = DeepSORT(max_age=max_frames_skip, iou_threshold=min_iou, tracker_type="kalman")
        trackers.append(track_handler)

        # Read detections file
        det_path = os.path.join(dets_sequence_path, camera, STORED_DETECTIONS_NAME)
        model_detections = utils.parse_predictions_rects(det_path)
        detections.append(model_detections)

        # Read features file
        features_path = os.path.join(dets_sequence_path, camera, STORED_FEATURES_NAME)
        model_feature_vectors = utils.parse_feature_vectors(features_path)
        features.append(model_feature_vectors)

        # Read gt data
        dataset_path = os.path.dirname(sequence_path)
        sequences = {os.path.basename(sequence_path): [camera]}
        dataset = AICityDatasetDetector(dataset_path, sequences)
        gt_datasets.append(dataset)


    # Iterate over videos, frame by frame
    video_finished = [False for _ in videos]
    frame_number = 0
    id_map = {}  # for re-id on multicameras
    #knns = [None for _ in videos]
    vid_ids = [[] for _ in videos]
    vid_fvs = [[] for _ in videos]
    with tqdm(total=max(vid_lengths), file=sys.stdout) as pbar:
        while not all(video_finished):
            for idx, cap in enumerate(videos):
                if not video_finished[idx]:
                    ret, img = cap.read()
                    video_finished[idx] = not ret
                    if not ret:
                        continue

                    # Read precomputed detections
                    model_detections = detections[idx]
                    frame_ids = model_detections[1][0]
                    final_dets, final_scores = model_detections[1][1][frame_ids == frame_number], model_detections[1][2][frame_ids == frame_number]

                    # Filter detections by score -> hyperparam
                    dets_keep = final_dets[final_scores > det_thr]
                    dets_keep = np.hstack([dets_keep, final_scores[final_scores > det_thr][:,np.newaxis]])

                    dets_final_keep = np.hstack([final_dets, final_scores[:,np.newaxis]])

                    # Update tracker
                    model_feature_vectors = features[idx]
                    frame_feature_vectors = model_feature_vectors[1][frame_ids == frame_number]
                    dets, feature_vectors = trackers[idx].update(image=img, dets=dets_final_keep, frame_feature_vectors = frame_feature_vectors)
                    # dets is [bbox, id] -> we will ReID with feature vectors

                    # Separate new detections from existing ones
                    new_id, new_fvs, det_idxs = [], [], []
                    for det_idx, (id, fv) in enumerate(zip(dets[:,-1], feature_vectors)):
                        if int(id) not in id_map:
                            new_id.append(int(id))
                            new_fvs.append(np.array(fv))
                            det_idxs.append(det_idx)
                        else:
                            dets[det_idx,-1] = id_map[int(id)]

                    if idx == 0 and len(vid_fvs[idx]) == 0:
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
                                dets[det_idxs[match[0]],-1] = v2_ids[match[1]]


                        # Add new ids which are not matched
                        vid_ids[idx].extend(unmatched_ids)
                        vid_fvs[idx].extend(unmatched_fvs)
                        for id in unmatched_ids:
                            id_map[id] = id

                    #import pdb; pdb.set_trace()


                    """
                    for det_idx, (id, fv) in enumerate(zip(dets[:,-1], feature_vectors)):
                        if int(id) not in id_map:
                            most_sim_value = None
                            most_sim_id = None
                            from_idx = 0

                            for fvs_idx, fvs in enumerate(vid_fvs):
                                if fvs_idx != idx and len(fvs) > 0:
                                    sim = (torch.stack(fvs) @ fv.T).squeeze()
                                    if sim.max() > 0.6:
                                        most_sim_value = max(sim.max(), most_sim_value or 0)
                                        most_sim_id = vid_ids[fvs_idx][sim.argmax()]
                                        from_idx = fvs_idx

                            #for knn_idx, knn in enumerate(knns):
                            #    if knn and knn_idx != idx:
                            #        dist, nn_id = knn.kneighbors(fv, return_distance=True)
                            #        if dist < 0.4:
                            #            most_sim_dist = min(dist, most_sim_dist or np.inf)
                            #            most_sim_id = int(nn_id) if most_sim_dist == dist else most_sim_id
                            if most_sim_id:
                                print(f"camera: {idx}-{from_idx} | {int(id)} : {most_sim_id}")
                                id_map[int(id)] = most_sim_id
                                dets[det_idx,-1] = most_sim_id
                            else:
                                # New id! different from anything seen yet
                                vid_ids[idx].append(int(id))
                                vid_fvs[idx].append(fv)
                                id_map[int(id)] = int(id)
                        else:
                            dets[det_idx,-1] = id_map[int(id)]

                    # Update knn for this camera
                    #knns[idx] = KNeighborsClassifier(n_neighbors=1)
                    #fvs = torch.stack(vid_fvs[idx])
                    #if len(fvs.shape) > 1:
                    #    fvs = fvs.squeeze(1)
                    #knns[idx].fit(fvs, np.array(vid_ids[idx]))

                    #import pdb; pdb.set_trace()
                    """

            frame_number += 1
            pbar.update(1)


task2("/home/group03/m6/train/S01", "/home/group03/m6/precomputed_w5/FasterRCNN_finetune_S03_S04_e2/S01", 5, 0.4, 0.8, 0.7)
