import numpy as np

def voc_ap(rec, prec):
    """Compute VOC AP given precision and recall. Using the VOC 07 11-point method for AP computation."""
    # 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap


def voc_eval(predictions, annotations, ovthresh=0.5):
    """
    VOC eval of AP for a specific class.
    [predictions]: Predictions with the format:
        image_ids, BB, confidence = predictions
        Where each element is a list of all detected bounding boxes with their confidence and their frame id.
    [annotations]: Annotations with the format:
        {
            'imagename': {
                'bbox': [['x1', 'y1', 'x2', 'y2'], ['x1', 'y1', 'x2', 'y2']],
                'difficult':[True, False],
                'det': [False, False]
            }, ...
        }
    [ovthresh]: Overlap threshold (default = 0.5)
    (Code from detectron2: https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/evaluation/pascal_voc_evaluation.py#L187)
    """

    # parse gt_rects a esto
    # class_recs = {
    #     'imagename': {
    #         'bbox': [['x1', 'y1', 'x2', 'y2'], ['x1', 'y1', 'x2', 'y2']],
    #         'difficult':[True, False],
    #         'det': [False, False]
    #     }
    # }
    class_recs = {}
    npos = 0
    for frame_id in sorted(list(annotations.keys())):
        bboxes = annotations[frame_id]
        class_recs[frame_id] = {
            'bbox': bboxes,
            'difficult': list(np.zeros(len(bboxes)).astype(np.bool)), # treat all GT as "non-difficult" -> All bboxes are GT
            'det': list(np.zeros(len(bboxes)).astype(np.bool))
        }
        npos += len(bboxes)
    
    # image_ids = [0] # list of frame ids
    # confidence = np.array([1]) # list of confidences ( one for each detection )
    # BB = np.array([['x1', 'y1', 'x2', 'y2']]) # bounding boxes of all detections
    image_ids, BB, confidence = predictions
    
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = np.array(R['bbox']).astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return ap
    