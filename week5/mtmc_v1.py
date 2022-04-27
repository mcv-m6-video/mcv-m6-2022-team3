import os
import scipy
import itertools
import numpy as np
from argparse import ArgumentParser

import motmetrics as mm

from sort import linear_assignment
from datasets import AICityDatasetDetector

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as con_comp
from scipy.sparse import coo_matrix

def mtmc_concomp(sim_thr, sequence, path_tracklets, path_gt, path_save):
    tracklets_by_video = {}
    ids_by_video = {}
    num_tracks_by_video = []
    idx2cam = {}
    gt_datasets = []
    for idx, cam in enumerate(os.listdir(os.path.join(path_tracklets, sequence))):
        idx2cam[idx] = cam
        cam_info = np.load(os.path.join(path_tracklets, sequence, cam))
        num_tracks_by_video.append(len(set(cam_info[:,1])))
        
        tracklets_mean = {}
        detections_ids = list(set(cam_info[:,1]))
        for tracklet_id in detections_ids:
            tracklets_mean[tracklet_id] = cam_info[cam_info[:,1] == tracklet_id,-512:].mean(axis=0)
            
        tracklets_by_video[idx] = tracklets_mean
        ids_by_video[idx] = detections_ids
        
        # read gt data for evaluation
        dataset = AICityDatasetDetector(
            path_gt,
            {sequence: [cam.split('.')[0]]})
        gt_datasets.append(dataset)
        
    start_videos = np.cumsum([0] + num_tracks_by_video)

    # compute similarities between every pair of cameras
    cam_ids = tracklets_by_video.keys()
    combinations = list(itertools.combinations(cam_ids, 2))

    rows = []
    columns = []
    data = np.ones(len(rows))
    for idx1, idx2 in combinations:
        f1 = np.array(list(tracklets_by_video[idx1].values()))
        f2 = np.array(list(tracklets_by_video[idx2].values()))
        similarity_matrix = (f1 @ f2.T)
        matches = linear_assignment(-similarity_matrix)
        
        for match in matches:
            if similarity_matrix[match[0], match[1]] > sim_thr:
                rows.append(start_videos[idx1] + match[0])
                columns.append(start_videos[idx2] + match[1])
                
    # create the graph and find the connected components
    graph = coo_matrix((data, (rows, columns)), shape=(start_videos[-1],start_videos[-1])).toarray()
    graph = csr_matrix(graph)
    _, labels = con_comp(csgraph=graph, directed=False, return_labels=True)

    # evaluate
    ids_map_by_video = {}
    for cam_idx, detection_ids in ids_by_video.items():
        ids_map_by_video[cam_idx] = {}
        for det_idx, det_id in enumerate(detection_ids):
            ids_map_by_video[cam_idx][det_id] = labels[start_videos[cam_idx] + det_idx]

    acc = mm.MOTAccumulator(auto_id=True)

    for cam_idx, id_map in ids_map_by_video.items():
        cam = idx2cam[cam_idx]
        cam_info = np.load(os.path.join(path_tracklets, sequence, cam))
        new_ids = np.vectorize(ids_map_by_video[cam_idx].get)(cam_info[:,1])
        dataset = gt_datasets[cam_idx]
        for frame_id in set(cam_info[:,0]):
            frame_mask = cam_info[:,0] == frame_id
            ids = new_ids[frame_mask]
            dets = cam_info[frame_mask, 2:6]

            _, gt = dataset[frame_id]
            if gt and "boxes" in list(gt.keys()):
                if len(dets) > 0:
                    gt_boxes = gt['boxes']
                    gt_this_frame = [int(x) for x in gt['track_id']]

                    dets_centers = np.vstack([(dets[:,0]+dets[:,2])/2, (dets[:,1]+dets[:,3])/2]).T
                    gt_centers = np.vstack([(gt_boxes[:,0]+gt_boxes[:,2])/2, (gt_boxes[:,1]+gt_boxes[:,3])/2]).T  # gt_boxes
                    dists = scipy.spatial.distance_matrix(dets_centers, gt_centers).T.tolist()
                    acc.update(
                        gt_this_frame,
                        ids,
                        dists
                    )
                else:
                    acc.update(
                        gt_this_frame,
                        [], []
                    )

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary[['idf1','idp','idr','precision','recall']])

    # Save detections with global ids (so we can later plot them...)
    if path_save:
        for cam_idx, id_map in ids_map_by_video.items():
            cam = idx2cam[cam_idx]
            cam_info = np.load(os.path.join(path_tracklets, sequence, cam))
            new_ids = np.vectorize(ids_map_by_video[cam_idx].get)(cam_info[:,1])
            reID_cam_info = cam_info.copy()
            reID_cam_info[:,1] = new_ids
            
            path_seq = os.path.join(path_save, sequence)
            os.makedirs(path_seq, exist_ok=True)
            np.save(os.path.join(path_seq, cam), reID_cam_info)



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-s",
                        dest="sequence",
                        required=True,
                        type=str,
                        help="Sequence of videos that we want to use. Options: S01, S03, S04.")
    parser.add_argument("-t",
                        dest="path_tracklets",
                        required=True,
                        type=str,
                        help="Path to the folder where the thacklets are.")
    parser.add_argument("-g",
                        dest="path_gt",
                        required=True,
                        type=str,
                        help="Path to the ground truth files.")
    parser.add_argument("-i",
                        dest="sim_thr",
                        required=False,
                        default=0.4,
                        type=float,
                        help="Similarity threshold.")
    parser.add_argument("-a",
                        dest="path_save",
                        required=False,
                        default=None,
                        type=str,
                        help="Path to the folder where to save the detections with the new IDs.")
    args = parser.parse_args()
    return args.sequence, args.path_tracklets, args.path_gt, args.path_save, args.sim_thr
    


if __name__ == "__main__":
    sequence, path_tracklets, path_gt, path_save, sim_thr = parse_arguments()
    mtmc_concomp(sim_thr, sequence, path_tracklets, path_gt, path_save)