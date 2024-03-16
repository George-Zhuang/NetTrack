# video to image sequences
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert video to image sequences")
    parser.add_argument('--video_folder', type=str, default='./data/cloth/demo/videos')
    parser.add_argument('--image_folder', type=str, default='./data/cloth/demo/images')
    parser.add_argument('--videos', type=str, nargs='+', default='4', help='all or None ==> all the videos')
    args = parser.parse_args()

    image_folder = args.image_folder
    video_folder = args.video_folder
    if args.videos in [None, 'all', 'ALL', 'All']:
        videos = os.listdir(video_folder)
    else:
        videos = args.videos
    videos.sort()
    os.makedirs(image_folder, exist_ok=True)
    for video in videos:
        video_path = os.path.join(video_folder, video)
        image_path = os.path.join(image_folder, video.split('.')[0])
        os.makedirs(image_path, exist_ok=True)
        os.system(f"ffmpeg -i {video_path} -q:v 1 -start_number 1 {image_path}/%06d.jpg")
        print(f"Video {video} converted to image sequence {image_path}.")