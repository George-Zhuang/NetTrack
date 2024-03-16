#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp
from shutil import copy

__all__ = ["vis"]
from colorsys import hsv_to_rgb


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, line_thickness=10):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = line_thickness

    radius = max(5, int(im_w/140.))
    # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
        #             thickness=text_thickness)
    return im

def plot_crossmark(image, coordinates, color, sz=8, thickness=3):
    x, y = coordinates
    cv2.line(image,
             (int(x - sz//2), int(y - sz//2)),
             (int(x + sz//2), int(y + sz//2)),
             color, thickness)
    cv2.line(image,
             (int(x + sz//2), int(y - sz//2)),
             (int(x - sz//2), int(y + sz//2)),
             color, thickness)
    return image

def plot_points(image, point_stacks, tlwhs=None, point_thickness=10, crossmark_thickness=7): # 15, 7
    im = np.ascontiguousarray(np.copy(image))
    xmax = im.shape[1]
    for i, point_stack in enumerate(point_stacks):
        for point in point_stack:
            x, y = point
            hue = x / xmax
            r, g, b = hsv_to_rgb(hue, 1, 1)
            r, g, b = int(r*255), int(g*255), int(b*255)
            color = (r,g,b)
            if tlwhs is not None:
                for tlwh in tlwhs:
                    x1, y1, w, h = tlwh
                    if x1 <= x <= x1+w and y1 <= y <= y1+h:
                        intbox = tuple(map(int, (x, y)))
                        cv2.circle(im, intbox, point_thickness, color, -1)
                    else:
                        im = plot_crossmark(im, (x, y), color, sz=point_thickness, thickness=crossmark_thickness)
                        # cv2.circle(im, intbox, point_thickness, color, -1)
            else:
                intbox = tuple(map(int, (x, y)))
                cv2.circle(im, intbox, point_thickness, color, -1)
    return im

def plot_heatmap(heatmap, output_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap.cpu().numpy().transpose(), cmap='YlOrRd')
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_result(img_path, stracks, frame_id, seq_name, output_path=None, vis_box=False, vis_points=False, extra_points=None, min_area=10, max_area=1000000000000):
    if stracks is not None:
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_points = []
        result = ''
        for t in stracks:
            tlwh = t.tlwh
            tid = t.track_id
            is_nan = (True in np.isinf(tlwh))
            if tlwh[2] * tlwh[3] > min_area and tlwh[2] * tlwh[3] < max_area and not is_nan:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_points.append(t.curr_points)
                # save results
                line = f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                if line not in result:
                    result += line
        if vis_box:
            online_im = cv2.imread(img_path)
            online_im = plot_tracking(online_im, online_tlwhs, online_ids, frame_id=frame_id, fps=0)
        if vis_points:
            if extra_points is not None:
                online_points = extra_points
            online_im = plot_points(online_im, online_points, online_tlwhs)
        if vis_box or vis_points:
            save_folder = osp.join(output_path, seq_name)
            dst_visualize_path = osp.join(save_folder, f'{frame_id:06d}.jpg')
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(dst_visualize_path, online_im)
        return result
    else:
        if vis_box or vis_points:
            copy(img_path, osp.join(output_path, seq_name, f'{frame_id:06d}.jpg'))
    
def save_result_mc(img_path, stracks, frame_id, seq_name, output_path=None, vis_box=False, vis_points=False, extra_points=None, min_area=10):
    if stracks is not None:
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_points = []
        result = ''
        for t in stracks:
            tlwh = t.tlwh
            tid = t.track_id
            is_nan = (True in np.isinf(tlwh))
            if tlwh[2] * tlwh[3] > min_area and not is_nan:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_points.append(t.curr_points)
                # save results
                line = f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1,{t.cat}\n"
                if line not in result:
                    result += line
        if vis_box:
            online_im = cv2.imread(img_path)
            online_im = plot_tracking(online_im, online_tlwhs, online_ids, frame_id=frame_id, fps=0)
        if vis_points:
            if extra_points is not None:
                online_points = extra_points
            online_im = plot_points(online_im, online_points)
        if vis_box or vis_points:
            save_folder = osp.join(output_path, seq_name)
            dst_visualize_path = osp.join(save_folder, f'{frame_id:06d}.jpg')
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(dst_visualize_path, online_im)
        return result
    else:
        if vis_box or vis_points:
            copy(img_path, osp.join(output_path, seq_name, f'{frame_id:06d}.jpg'))

_COLORS = np.array(
    [
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.301, 0.745, 0.933,
        0.850, 0.325, 0.098,
        0.000, 0.447, 0.741,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

