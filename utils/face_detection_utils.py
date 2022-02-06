import cv2
import numpy as np

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def infer(img, model, input_key, thresh = 0.5, input_size=(640, 640)):
    center_cache = {}
    scores_list = []
    bboxes_list = []
    reorganized_key = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32']
    
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio>model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
    img[:new_height, :new_width, :] = resized_img

    blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)

    result = model.infer(inputs={input_key: blob})
    
    net_outs = [result[out] for out in reorganized_key]

    input_height = blob.shape[2]
    input_width = blob.shape[3]

    fmc = 3
    _feat_stride_fpn = [8, 16, 32]
    _num_anchors = 2

    for idx, stride in enumerate(_feat_stride_fpn):
        scores = net_outs[idx][0]
        bbox_preds = net_outs[idx + fmc][0]
        bbox_preds = bbox_preds * stride
        
        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)

        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )

        anchor_centers = np.stack([anchor_centers]*_num_anchors, axis=1).reshape( (-1,2) )

        center_cache[key] = anchor_centers

        pos_inds = np.where(scores>=thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale

    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]
    boxes = det[:,:4]
    scores = det[:,4:]
    det = np.concatenate((scores, boxes), axis=1)
    det = np.concatenate((np.zeros((det.shape[0],1), dtype=int), det), axis=1)
    return det

def nms(dets, thresh = 0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
