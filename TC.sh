proc_name=gpu_0
exp_name=$proc_name
cnt=0
gpus=(3)
n_gpus=${#gpus[@]}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#tasks=('rte' 'sst2' 'qnli' 'mnli' 'mrpc' 'snli')
tasks=('sst2')
seed=0

# EMA decay 설정 (0.99 = z_ema = 0.99 * z_ema_prev + 0.01 * z_batch)
ema_decay=0.99

# Train buffer 설정
train_buffer_max_size=1000
train_buffer_update_mode=heapq  # fifo, uniform, heapq

# Online/Offline sampling 설정
use_offline_sampling=true
online_ratio=0.5  # 0.0 = all offline, 1.0 = all online
offline_start_step=100

# Queue update mode 설정
queue_update_mode=acc # acc, log_reward

# Condition buffer 설정 (true: offline_start_step에서 train_buffer 복사하여 condition_buffer 생성)
use_condition_buffer=true

# Best candidate selection mode 설정 (log_reward or acc)
best_candidate_mode=acc

# 기존 세션이 있으면 종료
tmux kill-session -t "$proc_name$cnt" 2>/dev/null || true

tmux new -d -s "$proc_name$cnt"
tmux send-keys -t "$proc_name$cnt" "if [ -f \"\$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/anaconda3/etc/profile.d/conda.sh\"; elif [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; elif command -v conda >/dev/null 2>&1; then source \"\$(conda info --base)/etc/profile.d/conda.sh\"; fi" C-m
tmux send-keys -t "$proc_name$cnt" "conda activate rd_test2" C-m
tmux send-keys -t "$proc_name$cnt" "cd \"$SCRIPT_DIR\"" C-m
tmux send-keys -t "$proc_name$cnt" "export PYTHONPATH=\"$SCRIPT_DIR:\$PYTHONPATH\"" C-m

for i in "${!tasks[@]}"; do
    task="${tasks[$i]}"

    offline_flag=""
    if [ "$use_offline_sampling" = true ]; then
        offline_flag="--use_offline_sampling"
    fi

    condition_buffer_flag=""
    if [ "$use_condition_buffer" = true ]; then
        condition_buffer_flag="--condition_buffer"
    fi
    
    tmux send-keys -t "$proc_name$cnt" \
    "CUDA_VISIBLE_DEVICES=${gpus[$cnt % n_gpus]} python -m junmo.train \
    --task classification \
    --dataset ${task} \
    --agent_model google/gemma-1.1-7b-it \
    --eval_model google/gemma-1.1-7b-it \
    --cache_dir ~/.cache/huggingface \
    --train_steps 200 \
    --eval_period 100 \
    --batch_size 4 \
    --grad_acc_steps 4 \
    --max_prompt_length 150 \
    --num_example 5 \
    --gamma 1.0 \
    --reward acc \
    --ema_decay ${ema_decay} \
    --exp_name ${task}_acc_beta_0.01_ema_${ema_decay}_buffer_${train_buffer_update_mode}_online_${online_ratio}_m_start_${offline_start_step}+10_queue_${queue_update_mode}_best_${best_candidate_mode}_seed_${seed}_revision \
    --lr 1e-4 \
    --m_step_freq 1 \
    --m_step \
    --wandb_mode online \
    --train_buffer_max_size ${train_buffer_max_size} \
    --train_buffer_update_mode ${train_buffer_update_mode} \
    ${offline_flag} \
    --online_ratio ${online_ratio} \
    --offline_start_step ${offline_start_step} \
    --seed ${seed} \
    ${condition_buffer_flag}
    " C-m
done
