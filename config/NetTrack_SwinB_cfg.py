####################### Detection ######################
# model
config_file = "./weights/groundingdino/GroundingDINO_SwinB_cfg.py"
weights = "./weights/groundingdino/groundingdino_swinb_cogcoor.pth"
box_threshold = 0.1
text_threshold = 0.1
nms_threshold = 0.7
# input data
seq = '4'
camera = ''
data_dir = './data/cloth/demo/images'
text_prompt = 'clothes in hand'
max_frame = 1000
suffix = ''
prompt_mode = 'single'
second_prompt = 'white clothes'
second_prompt_frame = 400
# output data
res_dir = "./output/det_res"
save_txt = False
save_vis = False

####################### Tracking #######################
data_dir = './data/cloth/demo/images'
seq = '4'
camera = ''
det_preds = './output/det_res/'
suffix = ''
max_frame = 1000000000
# point tracker config
point_tracker = 'cotracker'
checkpoint = './weights/cotracker/cotracker_stride_4_wind_8.pth'
# tracker config
device = 0
track_thresh = 0.5
score_mode = 'quadratic'
aspect_ratio = [1/6, 6]
# output data
output_path = './output/track_res'
save_result = True
min_area = 10
max_area = 100000000
vis_box = True
vis_points = False