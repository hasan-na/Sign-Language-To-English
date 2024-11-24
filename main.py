import torch # type: ignore
import json
import os

# Add ffmpeg to the PATH
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-master-latest-win64-gpl\bin"


#load JSON file
def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
   
def main():
    
    train_json = load_json('MS-ASL/MSASL_train.json')
    test_json = load_json('MS-ASL/MSASL_test.json')
    val_json = load_json('MS-ASL/MSASL_val.json')
    
if __name__ == '__main__':
    main()
