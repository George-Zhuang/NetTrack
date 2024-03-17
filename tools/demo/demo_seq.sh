# run the demo video
video_dir=${1:-./data/demo}
video=${2:-demo}
text_prompt=${3:-bird}
python tools/demo/video2image.py
python tools/demo/det_demo.py --seq_dir $video_dir/images --seq $video --text_prompt $text_prompt
python tools/demo/track_demo.py --seq_dir $video_dir/images --seq $video 
ffmpeg -f image2 -i ./data/demo/track_res/$video/%06d.jpg ./data/demo/track_res/$video_res.mp4