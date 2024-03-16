################### run the demo video in the offline mode ###################
video_name="test_V1"

################### (Optional) Video to image ###################
python tools/cloth_demo/video2image.py \
    --videos ${video_name}.mp4 \
    --video_folder ./data/cloth/demo/videos \
    --image_folder ./data/cloth/demo/images

################### Detection ###################
python tools/cloth_demo/det_demo.py \
    --data_dir ./data/cloth/demo/images \
    --seq ${video_name} \
    --camera '' \
    --text_prompt "clothes in hand" \
    # --prompt_mode multi \
    # --second_prompt "black clothes" \
    # --second_prompt_frame 200 # prompt change frame

################### Tracking ###################
python tools/cloth_demo/track_demo.py \
    --data_dir ./data/cloth/demo/images \
    --seq ${video_name} \
    --track_thres 0.5 \
    --camera '' \
    --max_area 100000

# ################### Image to video ###################
ffmpeg -f image2 -i ./output/track_res/${video_name}_/%06d.jpg ./output/track_res/${video_name}.mp4 