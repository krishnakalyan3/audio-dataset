from pathlib import Path
import re
from p_tqdm import p_map
from functools import partial
import uuid
import webdataset as wds
from datasets import load_dataset
import json
from datasets import load_dataset
import pandas as pd
from file_utils import json_load, json_dump
import logging
import shutil
import os
import time
from utils import calculate_audio_duration_from_bytes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='out.log', filemode='w')

def process_stem(stem_dir, output_dir):

    try:
        # Create tmp directory
        dataset = wds.WebDataset(str(stem_dir))

        for item in dataset:
            # Check if 2 mins or greater
            current_mix = item["flac"]
            secs = calculate_audio_duration_from_bytes(current_mix)

            if secs >= 2.0:
                # Save JSON
                file_name = str(Path(item["__key__"]).name)
                filtered_df = df[df['__key__'] == item["__key__"]]
                txt_encode = json.loads(item['json'])['text']

                json_dic = {"text": txt_encode, "transcript": filtered_df.transcription.values[0].strip(), "length": secs}
                json_save_path = output_dir  / (file_name + ".json")
                json_dump(json_dic, json_save_path)

                # Save FLAC File
                flac_save_path = output_dir  / (file_name + ".flac")
                with open(flac_save_path, 'wb') as f:
                    f.write(current_mix)

    except Exception as e:
        logging.error(f"Error processing {stem_dir} {e}")


if __name__ == '__main__':
    start_time = time.time()  # Start timer
    transcript = Path('/home/evobits/.cache/huggingface/hub/datasets--DavidFM43--emo_webds_transcriptions/snapshots/81bd2f489fbffcde8a785acf11db559bad7c6e73/data')
    emo_speech =  Path('/home/evobits/.cache/huggingface/hub/datasets--krishnakalyan3--emo_webds/snapshots/b853b1d6f383dc19b26f884453111de8b2fca9f9/dataset/')

    for i in ["train"]:
        output_dir = f"/home/evobits/krishna/2sec/output/{i}"
        os.system(f"rm -rf {output_dir}")
        os.system(f"mkdir -p {output_dir}")
        output_dir_path = Path(output_dir)
        t_path = list((transcript).glob(f'{i}*.parquet'))[0]
        e_path = list((emo_speech / i).glob('*.tar'))
        df = pd.read_parquet(t_path)
        p_map(partial(process_stem, output_dir=output_dir_path), e_path, num_cpus=90)
        #process_stem(e_path[0], t_df, output_dir_path)

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    logging.info(f"Total time taken for processing: {total_time:.2f} seconds")