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

def mtmc_concompish(sim_thr, sequence, path_tracklets, path_gt, path_save):
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

    # create graph
    nodes = []
    edges = []
    for idx1, idx2 in combinations:
        f1 = np.array(list(tracklets_by_video[idx1].values()))
        f2 = np.array(list(tracklets_by_video[idx2].values()))
        similarity_matrix = (f1 @ f2.T)
        matches = linear_assignment(-similarity_matrix)
        
        det_ids1 = list(tracklets_by_video[idx1].keys())
        det_ids2 = list(tracklets_by_video[idx2].keys())
        
        for match in matches:
            if similarity_matrix[match[0], match[1]] > sim_thr:
                #print(match)
                node1 = {'cam_id': idx1, 'id': start_videos[idx1] + match[0]}
                node2 = {'cam_id': idx2, 'id': start_videos[idx2] + match[1]}
                
                edge = {'node_id_1': start_videos[idx1] + match[0], 'node_id_2': start_videos[idx2] + match[1], 'score': similarity_matrix[match[0], match[1]],
                        'node_idx_1': len(nodes), 'node_idx_2': len(nodes)+1}
                
                nodes.append(node1)
                nodes.append(node2)
                edges.append(edge)

    # sort edges by score
    edges.sort(key=lambda x:x['score'], reverse=True)
                
    # find the connected components
    connected_components = []
    for idx, edge in enumerate(edges):
        node1 = nodes[edge['node_idx_1']]
        node2 = nodes[edge['node_idx_2']]
        
        # check if one of the nodes is connected to an existing connected component
        cc_node1 = [i for i, cc in enumerate(connected_components) if node1['id'] in cc['nodes']]
        cc_node1 = cc_node1[0] if len(cc_node1) > 0 else None
        cc_node2 = [i for i, cc in enumerate(connected_components) if node2['id'] in cc['nodes']]
        cc_node2 = cc_node2[0] if len(cc_node2) > 0 else None
        
        # no node is connected to an existing connected component
        if cc_node1 is None and cc_node2 is None:
            connected_components.append({'nodes':[node1['id'], node2['id']], 'cams': [node1['cam_id'], node2['cam_id']]})
        
        # only node2 is connected to a connected component and node1 can be joined to it (its from a different camera)
        elif cc_node1 is None and node1['cam_id'] not in connected_components[cc_node2]['cams']:
            connected_components[cc_node2]['nodes'].append(node1['id'])
            connected_components[cc_node2]['cams'].append(node1['cam_id'])
            
        # only node1 is connected to a connected component and node2 can be joined to it (its from a different camera)
        elif cc_node2 is None and node2['cam_id'] not in connected_components[cc_node1]['cams']:
            connected_components[cc_node1]['nodes'].append(node2['id'])
            connected_components[cc_node1]['cams'].append(node2['cam_id'])
        
        # both nodes are connected to a connected component and are from different cameras
        elif cc_node1 is not None and cc_node2 is not None and node1['cam_id'] not in connected_components[cc_node2]['cams'] and node2['cam_id'] not in connected_components[cc_node1]['cams'] and any([c1 in connected_components[cc_node2]['cams'] for c1 in connected_components[cc_node1]['cams']]):
            cc1 = connected_components.pop(max(cc_node1, cc_node2))
            cc2 = connected_components.pop(min(cc_node1, cc_node2))
            cc_nodes = cc1['nodes'] + cc2['nodes']
            cc_cams = cc1['cams'] + cc2['cams']
            connected_components.append({'nodes': cc_nodes, 'cams': cc_cams})

    # assign label to each connected component
    labels = np.zeros(start_videos[-1])
    for i, cc in enumerate(connected_components):
        nodes = cc['nodes']
        cams = cc['cams']
        for node in nodes:
            labels[node] = i+1
            
    # assign new label to the nodes that don't belong to any connected component
    l = labels.max() + 1
    for i,label in enumerate(labels):
        if label == 0:
            labels[i] = l
            l += 1

    labels = labels.astype(int)

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
    mtmc_concompish(sim_thr, sequence, path_tracklets, path_gt, path_save)