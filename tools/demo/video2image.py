# video to image sequences
import os

video_folder = './data/demo/videos'
image_folder = './data/demo/images'
os.makedirs(image_folder, exist_ok=True)
videos = os.listdir(video_folder)
videos.sort()
for video in videos:
    video_path = os.path.join(video_folder, video)
    image_path = os.path.join(image_folder, video.split('.')[0])
    os.makedirs(image_path, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -q:v 1 -start_number 1 {image_path}/%06d.jpg")
    print(f"Video {video} converted to image sequence {image_path}.")