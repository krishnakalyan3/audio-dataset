from pathlib import Path
import re
from p_tqdm import p_map
from functools import partial
import uuid
import webdataset as wds
import logging
import shutil
import os
from datasets import load_dataset, Audio
from pathlib import Path
import re
from p_tqdm import p_map
from datasets import Dataset
from functools import partial
import uuid
import webdataset as wds
from file_utils import json_load, json_dump
import logging
import shutil
import os
import time
from datasets import concatenate_datasets, load_dataset
import soundfile as sf
import librosa
import numpy as np
import pandas as pd

ROOT = "/home/evobits/krishna/dl_ears_copy/"
AUDIO_SAVE_SAMPLE_RATE = 48000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                     filename='out_subset.log', filemode='w')

def audio_to_flac(audio_in_path, audio_out_path, sample_rate=48000, no_log=True, segment_start:float=0, segment_end:float=None):
    log_cmd = ' -v quiet' if no_log else ''
    segment_cmd = f'-ss {segment_start} -to {segment_end}' if segment_end else ''
    os.system(
        f'ffmpeg -y -i "{audio_in_path}" -vn {log_cmd} -flags +bitexact '
        f'-ar {sample_rate} -ac 1 {segment_cmd} "{audio_out_path}"')

def process_stem(row, working_dir):
    current_uuid = str(uuid.uuid4())
    txt_encode_raw = row['attributes_string']
    speaker_id = row["file_path"].split("/")[1]

    # Write Json
    json_dic = {"text": txt_encode_raw, "speaker_id": speaker_id}
    json_save_path = working_dir  / (current_uuid + ".json")
    json_dump(json_dic, json_save_path)

    # Write Flac
    flac_save_path = working_dir  / (current_uuid + ".flac")
    flac_tmp_save_path = Path(f"{ROOT}" + row["file_path"][1:])
    audio_to_flac(flac_tmp_save_path, flac_save_path, AUDIO_SAVE_SAMPLE_RATE)

    os.remove(flac_tmp_save_path)

if __name__ == '__main__':
    start_time = time.time()  # Start timer

    # Load the dataset
    df = pd.read_csv(f"{ROOT}speaker_data.csv")
    df["folder"] = df.file_path.apply(lambda x:x.split("/")[1])
    distinct_folders = df['folder'].unique()
    
    # 90, 10 split
    train_folders = distinct_folders[0:int(len(distinct_folders) * .90)]
    test_folders = np.setdiff1d(distinct_folders, train_folders)    

    train_df = df[df['folder'].isin(train_folders)]
    test_df = df[df['folder'].isin(test_folders)]

    os.system(f"mkdir -p {ROOT}train/tmp")
    os.system(f"mkdir -p {ROOT}test/tmp")

    # Train Split
    rows = [row for _, row in train_df.iterrows()]
    output_dir = Path(f"{ROOT}train/")
    p_map(partial(process_stem, working_dir=output_dir), rows, num_cpus=90)

    # Test Split
    rows = [row for _, row in test_df.iterrows()]
    output_dir = Path(f"{ROOT}test/")
    p_map(partial(process_stem, working_dir=output_dir), rows, num_cpus=90)

    os.system(f"rm -r {ROOT}train/tmp")
    os.system(f"rm -r {ROOT}test/tmp")


    #for index, row in df.iterrows():
    #    process_stem(row, Path("/home/evobits/krishna/ears_train"))

# python merge_dir.py --root_dir /home/evobits/krishna/dl_ears_copy/train --to_rename  True
# python merge_dir.py --root_dir /home/evobits/krishna/dl_ears_copy/test --to_rename True
# python3 make_tar.py --input /home/evobits/krishna/dl_ears_copy --output output --dataclass all --num_element 512

# Download Dataset
#for X in $(seq -w 001 107); do
#  curl -L https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip -o p${X}.zip
#  unzip p${X}.zip
#  rm p${X}.zip
#done
