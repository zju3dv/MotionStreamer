import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="lxxiao/MotionStreamer",
  repo_type="model",
  local_dir="./",
  allow_patterns=["Evaluator_272/*"],
  local_dir_use_symlinks=False,
  resume_download=True,
  max_workers=8
)
