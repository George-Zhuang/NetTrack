import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import torch.multiprocessing as mp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../'))
from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert
from torchvision.ops import nms

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def det_demo(args, text_prompt, model, video_dir, video_name, camera):
    prompt_cat = [cat.strip() for cat in text_prompt.split('.')]
    prompt_cat_id = [i for i in range(1, len(prompt_cat)+1)]
    img_list = glob(os.path.join(video_dir, '*.jpg'))
    # img_list = glob(os.path.join(video_dir, '*.png'))
    img_list.sort()
    if args.prompt_mode == 'multi':
        if isinstance(args.second_prompt, list):
            second_prompt = ' . '.join(args.second_prompt) + ' .'
        else:
            second_prompt = args.second_prompt
    if args.max_frame is not None and len(img_list) > args.max_frame:
        img_list = img_list[:args.max_frame]
    save_path = os.path.join(args.res_dir, video_name, camera, args.suffix)
    for img_path in img_list:
        os.makedirs(os.path.join(save_path, 'txt'), exist_ok=True)
        txt_path = os.path.join(save_path, 'txt', os.path.basename(img_path)[:-4]+'.txt')
        # if os.path.exists(txt_path):
        #     continue
        # box [cx, cy, w, h], logit [score]
        image_source, image = load_image(img_path)
        h, w, _ = image_source.shape
        if args.prompt_mode == 'multi' and int(os.path.basename(img_path)[:-4]) > args.second_prompt_frame:
            text_prompt = second_prompt
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold
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
        keep = nms(torch.tensor(result[:, :4]), torch.tensor(result[:, 4]), iou_threshold=args.nms_threshold)
        result = result[keep.numpy()]
        # save result
        if args.save_txt:
            np.savetxt(txt_path, result, fmt='%.2f', delimiter=',', newline='\n')
        if args.save_vis:
            vis_path = os.path.join(save_path, os.path.basename(img_path))
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(vis_path, annotated_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Zero-shot detection demo")
    # model
    parser.add_argument('--config_file', type=str, default="./weights/groundingdino/GroundingDINO_SwinB_cfg.py")
    parser.add_argument('--weights', type=str, default="./weights/groundingdino/groundingdino_swinb_cogcoor.pth")
    parser.add_argument('--box_threshold', type=float, default=0.1)
    parser.add_argument('--text_threshold', type=float, default=0.1)
    parser.add_argument('--nms_threshold', type=float, default=0.7)
    # input data
    parser.add_argument('--seq', type=str, nargs='+', default='4', help='all or None ==> all the seqs')
    parser.add_argument('--camera', type=str, nargs='+', default='', help='left, middle, right')
    parser.add_argument('--data_dir', type=str, default='./data/cloth/demo/images')
    parser.add_argument('--text_prompt', type=str, nargs='+', default='clothes in hand')
    parser.add_argument('--max_frame', type=int, default=1000)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--prompt_mode', type=str, choices=['single', 'multi'], default='single')
    parser.add_argument('--second_prompt', type=str, nargs='+', default='white clothes')
    parser.add_argument('--second_prompt_frame', type=int, default=400)
    # output data
    parser.add_argument('--res_dir', type=str, default="./output/det_res")
    parser.add_argument('--save_txt', type=bool, default=True)
    parser.add_argument('--save_vis', type=bool, default=True)
    args = parser.parse_args()
    setup_seed(0)

    # transform text prompt
    if isinstance(args.text_prompt, list):
        text_prompt = ' . '.join(args.text_prompt) + ' .'
    else:
        text_prompt = args.text_prompt
    model = load_model(args.config_file, args.weights)
    if args.seq in [None, 'all', 'ALL', 'All']:
        seqs = os.listdir(args.data_dir)
        seqs.sort()
    else:
        seqs = args.seq
    if not isinstance(args.camera, list):
        cameras = [args.camera]
    else:
        cameras = args.camera
    for video_name in tqdm(seqs):
        for camera in cameras:
            video_dir = os.path.join(args.data_dir, video_name, camera, args.suffix)
            print(f'Processing {video_dir} ...')
            det_demo(args, text_prompt, model, video_dir, video_name, camera)





