import os
import sys
from huggingface_hub import snapshot_download

assess_token = sys.argv[1]

snapshot_download(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    local_dir='llms/Llama-3.2-1B-Instruct',
    local_dir_use_symlinks=False,
    token=assess_token
)

snapshot_download(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    local_dir='llms/Llama-3.2-3B-Instruct',
    local_dir_use_symlinks=False,
    token=assess_token
)

snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    local_dir='llms/Llama-3.1-8B-Instruct',
    local_dir_use_symlinks=False,
    token=assess_token
)

snapshot_download(
    repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    local_dir='llms/DeepSeek-R1-Distill-Qwen-1.5B',
    local_dir_use_symlinks=False,
    token=assess_token
)

snapshot_download(
    repo_id='deepseek-ai/DeepSeek-OCR',
    local_dir='DeepSeek-OCR',
    local_dir_use_symlinks=False,
    token=assess_token
)

snapshot_download(
    repo_id='AlbertTan/CoLaR',
    local_dir='CoLaR',
    local_dir_use_symlinks=False,
    token=assess_token
)

os.system('cp -f modeling_deepseekocr.py ./DeepSeek-OCR/modeling_deepseekocr.py')