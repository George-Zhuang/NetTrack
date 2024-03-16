
import pyrealsense2 as rs
import numpy as np
from numpy import typing as npt
import cv2
from loguru import logger

import torch
import random
import torch.multiprocessing as mp

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../'))
from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert
from torchvision.ops import nms
from tracker.nettrack import NetTracker
from cotracker.predictor import CoTrackerPredictor


class TrackerArgs(object):
    def __init__(self):
        self.checkpoint = './weights/cotracker/cotracker_stride_4_wind_8.pth'
        self.aspect_ratio = [1/6, 6]

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def detection(image: npt.NDArray, text_prompt:str, model):
    # args
    box_threshold = 0.1
    text_threshold = 0.1
    nms_threshold = 0.7

    prompt_cat = [cat.strip() for cat in text_prompt.split('.')]
    prompt_cat_id = [i for i in range(1, len(prompt_cat)+1)]
    h, w = image.shape[:2]
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
    boxes_image_cxcxwh = boxes * np.array([w, h, w, h])
    boxes_image_xyxy = box_convert(boxes=boxes_image_cxcxwh, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    category = []
    for text in phrases:
        cat_id_valid = 0
        for cat, cat_id in zip(prompt_cat, prompt_cat_id):
            text_compare = text.replace(' ', '')
            cat_compare = cat.replace(' ', '')
            if text_compare == cat_compare:
                category.append(cat_id)
                cat_id_valid = 1
                break
        if cat_id_valid == 0:
            category.append(0)
    category = np.array(category)
    result = np.concatenate([boxes_image_xyxy, logits.reshape(-1,1), category.reshape(-1,1)], axis=1)
    # nms
    keep = nms(torch.tensor(result[:, :4]), torch.tensor(result[:, 4]), iou_threshold=nms_threshold)
    result = result[keep.numpy()]

    return result

def tracking(image: npt.NDArray, det_result: npt.NDArray, args: TrackerArgs):
    # pred = det_result[:5]
    # # validate aspect ratio
    # w = pred[2] - pred[0]
    # h = pred[3] - pred[1]
    # aspect_ratio = w / h
    # if aspect_ratio < args.aspect_ratio[0] or aspect_ratio > args.aspect_ratio[1]:
    #     continue
    # preds.append(pred)
    # seq_preds.append(preds)
    # img_paths.append(img_path)
    pass

class Camera(object):
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_color_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return color_frame

    def stop(self):
        self.pipeline.stop()


def main():
    camera = Camera()

    device = 0 # cuda device number
    torch.cuda.set_device(device)
    setup_seed(0)

    # tracker_args = TrackerArgs()
    # # load detation model
    # det_model = load_model('./weights/groundingdino/GroundingDINO_SwinB_cfg.py', './weights/groundingdino/groundingdino_swinb_cogcoor.pth')
    # logger.info('Detection model loaded.')
    # # load tracker model
    # point_tracker = CoTrackerPredictor(checkpoint=tracker_args.checkpoint).to(device)
    # tracker = NetTracker(args=tracker_args, point_tracker=point_tracker)
    # logger.info('Tracker model loaded.')

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            color_frame = camera.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', image)
            if cv2.waitKey(1) == 27:
                break

    finally:
        # Stop streaming
        camera.stop()

if __name__ == '__main__':
    main()