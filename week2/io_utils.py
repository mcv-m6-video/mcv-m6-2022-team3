import utils

def read_annotations(annotations):
    # gt_rect format: 
    # {frame_number: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], [], ...], ...}
    print("Reading annotations from:", annotations)
    gt_rect = utils.parse_xml_reacts(annotations)
    return gt_rect
