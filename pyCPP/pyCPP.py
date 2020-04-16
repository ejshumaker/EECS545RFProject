import os
import sys
import subprocess
import shutil

def main(video_path):
    subprocess.run(['../fastMCD/build/fastMCD', video_path, '1'], shell=False)

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print('path of video files not found')
        sys.exit(-1)
    video_path = sys.argv[1]
    results_path = sys.argv[2]
    main(video_path)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    cond = True
    batch_size = 50
    prev_new_file_size = -1
    source_dir = './results/'
    while cond:
        new_files = os.listdir(source_dir)
        if(len(new_files) == batch_size or len(new_files) == prev_new_file_size):
            for file_name in new_files:
                shutil.move(source_dir + file_name, results_path)

        prev_new_file_size = len(new_files)
        if(prev_new_file_size == 0):
            cond = False
    os.rmdir(source_dir)

       
