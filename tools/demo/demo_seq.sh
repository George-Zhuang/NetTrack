# run the demo video
python tools/demo/video2image.py
python tools/demo/det_demo.py --seq_dir ./data/cloth/demo --seq demo --text_prompt bird
python tools/demo/track_demo.py --seq demo
ffmpeg -f image2 -i ./data/demo/track_res/demo/%06d.jpg ./data/demo/track_res/demo_res.mp4