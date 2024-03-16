import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy
from lap import lapjv 
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
# from yolox.tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    # ious = bbox_ious(
    #     np.ascontiguousarray(atlbrs, dtype=np.float32),
    #     np.ascontiguousarray(btlbrs, dtype=np.float32)
    # )
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

# def point_within_box_distance(atracks, btracks):
#     # scores = []
#     scores = np.zeros([len(atracks), len(btracks)])
#     for a_idx, atrack in enumerate(atracks):
#         points = atrack.traj.curr_points
#         for b_idx, btrack in enumerate(btracks):
#             bbox = btrack.tlbr
#             count = 0
#             for point in points:
#                 if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
#                     count += 1
#             score = count / len(points)
#             scores[a_idx, b_idx] = score
#     cost_matrix = 1 - scores
#     return cost_matrix

# def point_within_box_distance(atracks, btracks):
#     # scores = []
#     scores = np.zeros([len(atracks), len(btracks)])
#     apoints = np.array([atrack.traj.curr_points.detach().cpu().numpy() for atrack in atracks])
#     try:
#         scores = point_within_box(apoints, [btrack.tlbr for btrack in btracks])
#     except:
#         pass
#     cost_matrix = 1 - scores
#     return cost_matrix

def point_within_box_distance(atracks, btracks, weight_pt=0.25):
    scores = np.zeros([len(atracks), len(btracks)])
    try:
        for a_idx, atrack in enumerate(atracks):
            points = atrack.traj.curr_points
            bboxes = np.array([btrack.tlbr for btrack in btracks])
            within_bbox = np.logical_and(np.logical_and(bboxes[:, 0] <= points[:, 0], points[:, 0] <= bboxes[:, 2]),
                                        np.logical_and(bboxes[:, 1] <= points[:, 1], points[:, 1] <= bboxes[:, 3]))
            count = np.sum(within_bbox, axis=1)
            score = count / len(points)
            scores[a_idx, :] = score
    except:
        pass
    cost_matrix = 1 - scores
    return cost_matrix


def point_within_box_distance_v3(atracks, btracks):
    # scores = []
    scores = np.zeros([len(atracks), len(btracks)])
    for a_idx, atrack in enumerate(atracks):
        # points = atrack.next_points
        points = atrack.curr_points
        for b_idx, btrack in enumerate(btracks):
            bbox = btrack.tlbr
            visibility_score = 0
            for idx, point in enumerate(points):
                if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
                    # visibility_score += atrack.next_visibility[idx] 
                    visibility_score += atrack.curr_visibility[idx] 
            if len(points) > 0:
                score = visibility_score / len(points)
            else:
                score = 0
            scores[a_idx, b_idx] = score
    cost_matrix = 1 - scores
    return cost_matrix

def point_within_box_distance_v3(atracks, btracks):
    # scores = []
    scores = np.zeros([len(atracks), len(btracks)])
    for a_idx, atrack in enumerate(atracks):
        # points = atrack.next_points
        points = atrack.curr_points
        for b_idx, btrack in enumerate(btracks):
            bbox = btrack.tlbr
            # visibility_score = 0
            # for idx, point in enumerate(points):
            #     if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
            #         # visibility_score += atrack.next_visibility[idx] 
            #         visibility_score += atrack.curr_visibility[idx] 
            mask = (points[..., 0] >= bbox[0]) & (points[..., 0] <= bbox[2]) & \
                (points[..., 1] >= bbox[1]) & (points[..., 1] <= bbox[3])
            visibility_score = torch.sum(mask).item()
            if len(points) > 0:
                score = visibility_score / len(points)
            else:
                score = 0
            scores[a_idx, b_idx] = score
    cost_matrix = 1 - scores
    return cost_matrix

def point_within_box_distance_v4(atracks, btracks):
    ''' points '''
    points = [track.curr_points for track in atracks]
    if len(points) == 0:
        return torch.tensor([])
    max_length = max(len(p) for p in points)
    padded_points = [F.pad(torch.tensor(p), (0, 0, 0, max_length - len(p))) for p in points]
    points = torch.stack(padded_points)
    num_points = points.shape(1)
    ''' boxes '''
    boxes = torch.stack([track.tlbr for track in btracks])
    num_boxes = boxes.shape(0)
    ''' expand '''
    expanded_points = points.unsqueeze(1).expand(-1, num_boxes, -1, -1)
    expanded_boxes = boxes.unsqueeze(0).expand(num_points, -1, -1, -1)
    ''' score '''
    num_points_inside_boxes = ((expanded_points[..., 0] >= expanded_boxes[..., 0]) &
                            (expanded_points[..., 1] >= expanded_boxes[..., 1]) &
                            (expanded_points[..., 0] <= expanded_boxes[..., 2]) &
                            (expanded_points[..., 1] <= expanded_boxes[..., 3])).sum(-1)
    scores = num_points_inside_boxes.float() / num_points.float()
    cost_matrix = 1 - scores
    return cost_matrix

def point_index_transform(tracks):
    pt_idx = [track.point_idx for track in tracks]
    max_length = max(len(pt) for pt in pt_idx)
    pt_idx_padded = [F.pad(torch.tensor(pt).squeeze(-1), (0, max_length - len(pt))) for pt in pt_idx]
    pt_idx = torch.stack(pt_idx_padded)
    return pt_idx

def point_within_box_distance_v5(atracks, btracks, weight_pt):
    if len(atracks) > 0 or len(btracks) > 0:
        ''' box '''
        _ious = np.zeros((len(atracks), len(btracks)))
        if _ious.size == 0:
            return 1 - _ious
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        scores_iou = ious(atlbrs, btlbrs)
        ''' points '''
        a_pt_idx = point_index_transform(atracks).cpu()
        a_num_track, a_num_pt = a_pt_idx.shape
        b_pt_idx = point_index_transform(btracks).cpu()
        b_num_track, b_num_pt = b_pt_idx.shape
        expanded_a_pt_idx = a_pt_idx.unsqueeze(1).unsqueeze(3).expand(a_num_track, b_num_track, a_num_pt, b_num_pt)
        expanded_b_pt_idx = b_pt_idx.unsqueeze(0).unsqueeze(2).expand(a_num_track, b_num_track, a_num_pt, b_num_pt)
        num_points_inside_boxes = (expanded_a_pt_idx > 0) & (expanded_b_pt_idx > 0) & \
        (expanded_a_pt_idx == expanded_b_pt_idx)
        scores_pt = torch.sum(num_points_inside_boxes, dim=(-2,-1)).float() / torch.count_nonzero(a_pt_idx, dim=1)
        scores = weight_pt * scores_pt + (1-weight_pt) * scores_iou
        cost_matrix = 1 - scores
        cost_matrix = cost_matrix.cpu().numpy()
    else:
        cost_matrix = np.zeros((0, 0))
    return cost_matrix

# def point_within_box(apoint_sets, btlbrs):
#     '''
#     apoint_sets: list of point sets, e.g., [[(x1,y1), (x2,y2), ...], [(x1,y1), (x2,y2), ...], ...]
#     btlbrs: list of tlbrs
#     '''
#     num_sets = apoint_sets.shape[0]
#     num_points = apoint_sets.shape[1]
#     apoints = apoint_sets.reshape(-1,2)
#     btlbrs = np.array(btlbrs)
#     num_boxes = btlbrs.shape[0]
#     expanded_points = np.ascontiguousarray(np.repeat(apoints[:,np.newaxis,:], num_boxes, axis=1))
#     expanded_boxes = np.ascontiguousarray(np.repeat(btlbrs[np.newaxis,:,:], num_points*num_sets, axis=0))
#     points_within_boxes = (expanded_points[..., 0] > expanded_boxes[..., 0]) & \
#                             (expanded_points[..., 1] > expanded_boxes[..., 1]) & \
#                             (expanded_points[..., 0] < expanded_boxes[..., 2]) & \
#                             (expanded_points[..., 1] < expanded_boxes[..., 3])
#     points_within_boxes = points_within_boxes.reshape(num_sets, num_points, num_boxes)
#     score = np.sum(points_within_boxes, axis=1)
#     return score

# def point_within_box(apoint_sets: List[List[Tuple[float, float]]], btlbrs: List[Tuple[float, float, float, float]]) -> np.ndarray:
#     '''
#     apoint_sets: list of point sets, e.g., [[(x1,y1), (x2,y2), ...], [(x1,y1), (x2,y2), ...], ...]
#     btlbrs: list of tlbrs
#     '''
#     num_sets = len(apoint_sets)
#     num_points = len(apoint_sets[0])
#     apoints = np.array(apoint_sets).reshape(-1,2)
#     btlbrs = np.array(btlbrs)
#     num_boxes = btlbrs.shape[0]
#     expanded_points = np.repeat(apoints[:,np.newaxis,:], num_boxes, axis=1)
#     expanded_boxes = np.repeat(btlbrs[np.newaxis,:,:], num_points*num_sets, axis=0)
#     points_within_boxes = (expanded_points[..., 0] > expanded_boxes[..., 0]) & \
#                             (expanded_points[..., 1] > expanded_boxes[..., 1]) & \
#                             (expanded_points[..., 0] < expanded_boxes[..., 2]) & \
#                             (expanded_points[..., 1] < expanded_boxes[..., 3])
#     points_within_boxes = points_within_boxes.reshape(num_sets, num_points, num_boxes)
#     score = np.sum(points_within_boxes, axis=1)
#     return score

# def point_within_box_distance_v6(atracks, btracks, weight_pt):
#     if len(atracks) > 0 or len(btracks) > 0:
#         ''' box '''
#         _ious = np.zeros((len(atracks), len(btracks)))
#         if _ious.size == 0:
#             return 1 - _ious
#         atlbrs = [track.tlbr for track in atracks]
#         btlbrs = [track.tlbr for track in btracks]
#         scores_iou = ious(atlbrs, btlbrs)
#         ''' points '''
#         apoint_sets = np.array([track.next_points for track in atracks])
#         scores_pt = point_within_box(apoint_sets, btlbrs)
#         scores = weight_pt * scores_pt + (1-weight_pt) * scores_iou
#         cost_matrix = 1 - scores
#     else:
#         cost_matrix = np.zeros((0, 0))
#     return cost_matrix


# def point_within_box_distance(atracks, btracks):
#     scores = np.zeros([len(atracks), len(btracks)])
#     try:
#         for a_idx, atrack in enumerate(atracks):
#             points = atrack.traj.curr_points
#             bboxes = np.array([btrack.tlbr for btrack in btracks])
#             within_bbox = np.logical_and(np.logical_and(bboxes[:, 0] <= points[:, 0], points[:, 0] <= bboxes[:, 2]),
#                                         np.logical_and(bboxes[:, 1] <= points[:, 1], points[:, 1] <= bboxes[:, 3]))
#             count = np.sum(within_bbox, axis=1)
#             score = count / len(points)
#             scores[a_idx, :] = score
#     except:
#         pass
#     cost_matrix = 1 - scores
#     return cost_matrix

def point_within_box_distance(atracks, btracks, mode='square', weight_pt=0.25):
    scores_pt = np.zeros([len(atracks), len(btracks)])
    atlbrs = np.array([atrack.tlbr for atrack in atracks])
    btlbrs = np.array([btrack.tlbr for btrack in btracks])
    try:
        for a_idx, atrack in enumerate(atracks):
            points = atrack.curr_points
            within_bbox = np.logical_and(np.logical_and(btlbrs[:, 0] < points[:, 0, None], points[:, 0, None] < btlbrs[:, 2]),
                                        np.logical_and(btlbrs[:, 1] < points[:, 1, None], points[:, 1, None] < btlbrs[:, 3]))
            # within_bbox = (points[:, 0, None] > btlbrs[:, 0]) & (points[:, 0, None] < btlbrs[:, 2]) & \
            #                 (points[:, 1, None] > btlbrs[:, 1]) & (points[:, 1, None] < btlbrs[:, 3])
            count = np.sum(within_bbox, axis=0)
            # add a weight of box area change to the score, where large candidate boxes are punished
            area_candidate = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
            area_track = (atrack.tlbr[2] - atrack.tlbr[0]) * (atrack.tlbr[3] - atrack.tlbr[1])
            area_weight =  np.minimum(np.round(area_track / area_candidate, 1), 1)
            score = area_weight * count / len(points) 
            scores_pt[a_idx, :] = score
    except:
        pass
    scores_iou = ious(atlbrs, btlbrs)
    # scores = weight_pt * scores_pt + (1-weight_pt) * scores_iou
    # scores = math.sqrt(scores_pt**2+scores_iou**2)
    if mode == 'arithmetic':
        scores = np.mean([scores_pt, scores_iou], axis=0)
    elif mode == 'geometric':
        scores = np.sqrt(scores_pt * scores_iou)
    elif mode == 'harmonic':
        scores = 2 * scores_pt * scores_iou / (scores_pt + scores_iou)
    elif mode == 'max':
        scores = np.maximum(scores_pt, scores_iou)
    elif mode == 'quadratic':
        scores = np.sqrt((weight_pt*scores_pt**2 + (1-weight_pt)*scores_iou**2))
    elif mode == 'solely_iou':
        scores = scores_iou
    else:
        raise Exception('Unkown mode of score fusion in matching! Supported modes: arithmetic, geometric, harmonic, max, quadratic.')
    # if len(scores_iou) > 0 and len(scores_pt) > 0:
    #     print(f'iou: {scores_iou}')
    #     print(f'pt: {scores_pt}')
    cost_matrix = 1 - scores
    return cost_matrix




# def point_within_box_distance_v6(atracks, btracks, weight_pt):
#     if len(atracks) > 0 or len(btracks) > 0:
#         ''' box '''
#         _ious = np.zeros((len(atracks), len(btracks)))
#         if _ious.size == 0:
#             return 1 - _ious
#         atlbrs = [track.tlbr for track in atracks]
#         btlbrs = [track.tlbr for track in btracks]
#         scores_iou = ious(atlbrs, btlbrs)
#         ''' points '''
#         apoint_sets = np.array([track.next_points for track in atracks])
#         scores_pt = point_within_box(apoint_sets, btlbrs)
#         scores = weight_pt * scores_pt + (1-weight_pt) * scores_iou
#         cost_matrix = 1 - scores
#     else:
#         cost_matrix = np.zeros((0, 0))
#     return cost_matrix



def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


# def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
#     if cost_matrix.size == 0:
#         return cost_matrix
#     gating_dim = 2 if only_position else 4
#     gating_threshold = kalman_filter.chi2inv95[gating_dim]
#     measurements = np.asarray([det.to_xyah() for det in detections])
#     for row, track in enumerate(tracks):
#         gating_distance = kf.gating_distance(
#             track.mean, track.covariance, measurements, only_position)
#         cost_matrix[row, gating_distance > gating_threshold] = np.inf
#     return cost_matrix


# def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
#     if cost_matrix.size == 0:
#         return cost_matrix
#     gating_dim = 2 if only_position else 4
#     gating_threshold = kalman_filter.chi2inv95[gating_dim]
#     measurements = np.asarray([det.to_xyah() for det in detections])
#     for row, track in enumerate(tracks):
#         gating_distance = kf.gating_distance(
#             track.mean, track.covariance, measurements, only_position, metric='maha')
#         cost_matrix[row, gating_distance > gating_threshold] = np.inf
#         cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
#     return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost