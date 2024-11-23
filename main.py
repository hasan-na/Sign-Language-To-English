import torch # type: ignore
import json
import os
from yt_dlp import YoutubeDL # type: ignore

#load JSON file
def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def download_video(video_url, save_path):
    os.makedirs(save_path, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),  # Save format
        'format': 'bestvideo',  
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
        print(f"Downloaded: {video_url}")

def main():
    train_json = load_json('MS-ASL/MSASL_train.json')
    test_json = load_json('MS-ASL/MSASL_test.json')
    val_json = load_json('MS-ASL/MSASL_val.json')

    # Download videos
    for entry in train_json:
        if 'url' in entry:
            download_video(entry['url'], save_path='data/train')
    for entry in test_json:
        if 'url' in entry:
            download_video(entry['url'], save_path='data/test')
    for entry in val_json:
        if 'url' in entry:
            download_video(entry['url'], save_path='data/val')
    
    print('Downloaded all videos')

if __name__ == '__main__':
    main()
