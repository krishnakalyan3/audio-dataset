from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets, concatenate_datasets
import polars as pl
import time
from pathlib import Path
import re
from p_tqdm import p_map
from functools import partial
import uuid
import webdataset as wds
from datasets import load_dataset
import librosa
from file_utils import json_load, json_dump
import logging
import shutil
import os
import time  # Import time module
import pandas as pd
import numpy as np
import soundfile as sf


AUDIO_SAVE_SAMPLE_RATE = 48000
split = "train"
ds_desc =  pd.DataFrame(load_dataset("ylacombe/mls-eng-10k-descriptions-10k-v5",  split=split).select_columns(['book_id', 'speaker_id', 'begin_time', 'text_description']))
ds_audio = load_dataset("parler-tts/mls_eng_10k", split=split).select_columns(['book_id', 'speaker_id', 'begin_time', 'audio'])

def process_stem(stem, working_dir):
    # https://github.com/huggingface/parler-tts/blob/8b8c576e2dbdc29172e30be7d68fac9357cd92c5/training/data.py#L236
    txt_encode = ds_desc[(ds_desc.book_id == stem["book_id"]) & (ds_desc.begin_time == stem["begin_time"]) & (ds_desc.speaker_id == str(stem["speaker_id"])) ].text_description.iloc[0]

    # Prepare JSON
    current_uuid = str(uuid.uuid4())
    json_dic = {"text": txt_encode.strip(), "speaker_id": str(stem["speaker_id"])}
    json_save_path = working_dir  / (current_uuid + ".json")
    json_dump(json_dic, json_save_path)

    # Process Flac
    current_mix = stem["audio"]["array"]
    flac_save_path = working_dir  / (current_uuid + ".flac")
    resampled_array = librosa.resample(np.array(current_mix), orig_sr=16000, target_sr=AUDIO_SAVE_SAMPLE_RATE)
    sf.write(flac_save_path, resampled_array, AUDIO_SAVE_SAMPLE_RATE, format='FLAC')

if __name__ == '__main__':
    start_time = time.time()  # Start timer

    working_dir = f"/home/evobits/krishna/parler/{split}"
    os.system(f"rm -rf {working_dir}")
    os.system(f"mkdir -p {working_dir}")

    p_map(partial(process_stem, working_dir=Path(working_dir)), ds_audio, num_cpus=90)

    #for i in ds_audio:
    #    process_stem(i, Path(working_dir))

# python merge_dir.py --root_dir /home/evobits/krishna/parler2/dev --to_rename  True
# python merge_dir.py --root_dir /home/evobits/krishna/parler2/test --to_rename  True
# python3 make_tar.py --input /home/evobits/krishna/parler2/ --output output --dataclass dev --num_element 512
