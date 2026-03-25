# GflowPO

Prompt optimization via GflowNet

## Installation
한줄씩 치셔야 됩니다


```
git clone https://github.com/june0102/GFLOWPO.git
```

```bash
conda create -n rd_test python=3.10 -y
conda activate rd_test

pip install -U pip setuptools wheel
pip install -U torch==2.6.0 torchvision torchaudio accelerate transformers
pip install vllm==0.8.4
pip install flash-attn==2.7.1.post4 --no-build-isolation
pip install typed-argument-parser csv_logger peft sentence_transformers editdistance pandas fastchat matplotlib datasets trl wandb
pip install -r requirements.txt

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
