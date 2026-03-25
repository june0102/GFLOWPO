# GflowPO

Prompt optimization via GflowNet

## Installation
한줄씩 치셔야 됩니다


```bash
git clone https://github.com/june0102/GFLOWPO.git
```

```bash
# rd_test2 / B200 (cu128) environment notes
#
# 1) Activate the env and keep user-site packages out of the way:
#    conda activate rd_test2
#    conda env config vars set PYTHONNOUSERSITE=1
#    conda deactivate && conda activate rd_test2
#
# 2) Install PyTorch first (already done in your case):
#    python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
#      --index-url https://download.pytorch.org/whl/cu128
#
# 3) Install this file:
#    python -m pip install -r requirements_rd_test2.txt
#
# 4) Install CUDA-coupled packages last:
#    python -m pip install xformers==0.0.30
#    MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
#    python -m pip install vllm==0.9.2

```

## Text Classificaion
```bash
./TC.sh
```
## Induction Task(BigBench Instruction Induction)
```bash
./BBII_TC.sh
./BBII_TG.sh

```
## Instruction Induction dataset
```bash
./II.sh
```

## Question Answering
```bash
MMLU.sh
```
