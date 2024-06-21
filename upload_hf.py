from huggingface_hub import HfApi, logging
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi()
logging.set_verbosity_debug()

api.upload_folder(
    folder_path="<enter_path>",
    repo_id="krishnakalyan3/emo_webds_2",
    path_in_repo="data/test",
    repo_type="dataset",
    token="<your_token>",
    multi_commits=True,
    multi_commits_verbose=True,
)