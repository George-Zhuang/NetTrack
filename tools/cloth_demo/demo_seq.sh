# run the demo video
video_name="clothes_in_hand"
# python tools/cloth_demo/video2image.py \
#     --videos ${video_name}.mp4 \
#     --video_folder ./data/cloth/demo/videos \
#     --image_folder ./data/cloth/demo/images

# python tools/cloth_demo/det_demo.py \
#     --data_dir ./data/cloth/demo/images \
#     --seq ${video_name} \
#     --camera '' \
#     --text_prompt "clothes in hand"

python tools/cloth_demo/track_demo.py \
    --data_dir ./data/cloth/demo/images \
    --seq ${video_name} \
    --camera '' \
    --max_area 100000

ffmpeg -f image2 -i ./output/track_res/${video_name}_/%06d.jpg ./output/track_res/${video_name}.mp4 