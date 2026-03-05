#!/bin/bash
# GPU 0: All MMLU tasks (57 subjects) with SEED 0 - tmux session

proc_name=mmlu_gpu0
SEED=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS=(
    'high_school_world_history'
)

# 기존 세션이 있으면 종료
tmux kill-session -t "$proc_name" 2>/dev/null || true

tmux new -d -s "$proc_name"
tmux send-keys -t "$proc_name" "if [ -f \"\$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/anaconda3/etc/profile.d/conda.sh\"; elif [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; elif command -v conda >/dev/null 2>&1; then source \"\$(conda info --base)/etc/profile.d/conda.sh\"; fi" C-m
tmux send-keys -t "$proc_name" "conda activate rd_test" C-m
tmux send-keys -t "$proc_name" "cd \"$SCRIPT_DIR\"" C-m
tmux send-keys -t "$proc_name" "export PYTHONPATH=\"$SCRIPT_DIR:\$PYTHONPATH\"" C-m

for TASK in "${TASKS[@]}"; do
    tmux send-keys -t "$proc_name" \
    "CUDA_VISIBLE_DEVICES=0 python -m junmo.train \
    --task qa \
    --dataset ${TASK} \
    --agent_model google/gemma-1.1-7b-it \
    --eval_model google/gemma-1.1-7b-it \
    --seed ${SEED} \
    --train_steps 150 \
    --batch_size 1 \
    --grad_acc_steps 16 \
    --max_prompt_length 150 \
    --num_example 5 \
    --gamma 1.0 \
    --lr 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --ema_decay 0.99 \
    --reward_epsilon 1e-8 \
    --temp_low 0.5 \
    --temp_high 2.0 \
    --lm_sched_start 1.0 \
    --lm_sched_end 1.0 \
    --lm_sched_horizon 10 \
    --use_offline_sampling \
    --offline_start_step 100 \
    --online_ratio 0.5 \
    --condition_buffer \
    --m_step \
    --train_buffer_max_size 1000 \
    --eval_period 5 \
    --wandb_project mmlu_gfn \
    --wandb_mode online \
    --exp_name mmlu_${TASK}_seed${SEED}_use_val_acc
    " C-m
done

echo "Started tmux session: $proc_name"
echo "Attach with: tmux attach -t $proc_name"
