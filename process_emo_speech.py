from pathlib import Path
import re
from p_tqdm import p_map
from functools import partial
import uuid
import webdataset as wds
from datasets import load_dataset

from file_utils import json_load, json_dump
import logging
import shutil
import os
import time  # Import time module
from utils import calculate_audio_duration_from_bytes
#from transcription import get_transcript

AUDIO_SAVE_SAMPLE_RATE = 48000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='out.log', filemode='w')

def audio_to_flac(audio_in_path, audio_out_path, sample_rate=48000, no_log=True, segment_start:float=0, segment_end:float=None):
    log_cmd = ' -v quiet' if no_log else ''
    segment_cmd = f'-ss {segment_start} -to {segment_end}' if segment_end else ''
    os.system(
        f'ffmpeg -y -i "{audio_in_path}" -vn {log_cmd} -flags +bitexact '
        f'-ar {sample_rate} -ac 1 {segment_cmd} "{audio_out_path}"')


def process_stem(stem_dir, working_dir):
    try:
        print(stem_dir)
        # Create tmp directory
        dataset = wds.WebDataset(str(stem_dir))

        for item in dataset:
            # Save FLAC File
            current_mix = item["flac"]
            secs = calculate_audio_duration_from_bytes(current_mix)

            if secs >= 2.0:
                current_uuid = str(uuid.uuid4())
                txt_encode_raw = item['txt'].decode('utf-8', 'ignore')
                txt_encode = txt_encode_raw.replace('woman', 'person').replace('female', '').replace('man', 'person').replace('male', '')

                # Save JSON
                current_mix = item["flac"]
                #whisper = get_transcript(current_mix)
                json_dic = {"text": txt_encode}
                json_save_path = working_dir  / (current_uuid + ".json")
                json_dump(json_dic, json_save_path)

                # Save FLAC File
                flac_tmp_save_path = working_dir  / "tmp" /(current_uuid + ".flac")
                flac_save_path = working_dir  / (current_uuid + ".flac")

                with open(flac_tmp_save_path, 'wb') as f:
                    f.write(current_mix)

                audio_to_flac(flac_tmp_save_path, flac_save_path, AUDIO_SAVE_SAMPLE_RATE)

                # delete flac_save_path
                os.remove(flac_tmp_save_path)

    except Exception as e:
        logging.error(f"Error processing {stem_dir} {e}")


if __name__ == '__main__':
    start_time = time.time()  # Start timer
    input_dir = Path('/home/evobits/.cache/huggingface/hub/datasets--EQ4You--Emotional_Speech/snapshots/45489c0efecd92c626d9fb2c32c6fbca61bce84a/')
    stem_list = sorted(list(input_dir.glob('*.tar')))
    logging.info(f"Found {len(stem_list)} tar files to process.")

    os.system(f"mkdir -p /home/evobits/krishna/2sec/emo_processed_data_train/tmp")
    output_dir = Path("/home/evobits/krishna/2sec/emo_processed_data_train/")
    p_map(partial(process_stem, working_dir=output_dir), stem_list, num_cpus=90)

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    logging.info(f"Total time taken for processing: {total_time:.2f} seconds")