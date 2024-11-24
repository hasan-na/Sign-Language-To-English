import re
import ffmpeg # type: ignore
import json
import os

#load JSON file
def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
    
# Normalize a string (lowercase, no spaces, etc.)
def normalize_string(s):
    s = re.sub(r'[^\x00-\x7F]+', '', s)  
    s = re.sub(r'[^\w\s]', '', s)  
    s = re.sub(r'\s+', '', s) 
    return s.lower().strip()   

# Create a mapping between JSON `file` and actual filenames
def create_file_mapping(json_data, video_folder):
    video_files = {normalize_string(os.path.splitext(f)[0]): f for f in os.listdir(video_folder)}
    mapping = {}
    for item in json_data:
        normalized_file = normalize_string(item["file"])
        if normalized_file in video_files:
            mapping[item["file"]] = video_files[normalized_file]
            print(f"Matched: {item['file']} -> {video_files[normalized_file]}")
        else:
            print(f"Warning: No match found for {normalized_file}")
    return mapping

def clip_video(video_path, start_time, end_time, save_path, label):
    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .output(save_path + '.mp4')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        print(f"Clipped: {video_path} to {save_path}")
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode('utf8')}")

def process_video(video_path, output_path, mapping, json_data):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for item in json_data:
        video = mapping.get(item['file'])
        if not video:
            print(f"Warning: No video found for {item['file']}")
            continue
            
        combined_video_path = os.path.join(video_path, video)
        start_time = item['start_time']
        end_time = item['end_time']
        label = item.get('clean_text') 
        combined_output_path = os.path.join(output_path, label) 
        clip_video(combined_video_path, start_time, end_time, combined_output_path, label)

def main():        
    train_json = load_json('MS-ASL/MSASL_train.json')
    test_json = load_json('MS-ASL/MSASL_test.json')
    val_json = load_json('MS-ASL/MSASL_val.json')
    
    train_mapping = create_file_mapping(train_json, 'data/train')
    test_mapping = create_file_mapping(test_json, 'data/test')
    val_mapping = create_file_mapping(val_json, 'data/val')

    process_video('data/train', 'data/train_clipped', train_mapping, train_json)
    process_video('data/test', 'data/test_clipped', test_mapping, test_json)
    process_video('data/val', 'data/val_clipped', val_mapping, val_json)

if __name__ == '__main__':
    main()