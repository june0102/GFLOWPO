conda create -n rd_test python=3.10 -y
conda activate rd_test

pip install -U pip setuptools wheel

pip install -U torch==2.6.0 torchvision torchaudio accelerate transformers
pip install -U accelerate transformers
pip install vllm==0.8.4
pip install flash-attn==2.7.1.post4
pip install typed-argument-parser csv_logger peft sentence_transformers editdistance pandas fastchat matplotlib datasets trl wandb
pip install -r requirements.txt

Text Classificaion

./TC.sh

Induction Task

./BBII_TC.sh
./BBII_TG.sh
./II.sh

Question Answering

MMLU.sh

