#!/bin/bash
# GPU 0: bbii_tc tasks (3 tasks) - tmux session

proc_name=bbii_tc_gpu0
SEED=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS=('disambiguation_qa'
    'epistemic_reasoning'
    'hyperbaton'
    'implicatures'
    'logical_fallacy_detection'
    'movie_recommendation'
    'navigate'
    'ruin_names'
    'snarks'
    'sports_understanding')
# 기존 세션이 있으면 종료
tmux kill-session -t "$proc_name" 2>/dev/null || true

tmux new -d -s "$proc_name"
tmux send-keys -t "$proc_name" "if [ -f \"\$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/anaconda3/etc/profile.d/conda.sh\"; elif [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; elif command -v conda >/dev/null 2>&1; then source \"\$(conda info --base)/etc/profile.d/conda.sh\"; fi" C-m
tmux send-keys -t "$proc_name" "conda activate rd_test2" C-m
tmux send-keys -t "$proc_name" "cd \"$SCRIPT_DIR\"" C-m
tmux send-keys -t "$proc_name" "export PYTHONPATH=\"$SCRIPT_DIR:\$PYTHONPATH\"" C-m

for TASK in "${TASKS[@]}"; do
    tmux send-keys -t "$proc_name" \
    "CUDA_VISIBLE_DEVICES=0 python -m junmo.train \
    --task bbii_tc \
    --dataset ${TASK} \
    --agent_model mistralai/Mistral-7B-Instruct-v0.2 \
    --eval_model google/gemma-1.1-7b-it \
    --seed ${SEED} \
    --train_steps 200 \
    --batch_size 4 \
    --grad_acc_steps 4 \
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
    --eval_period 100 \
    --wandb_project bbii_tc_gfn \
    --wandb_mode online \
    --exp_name bbii_tc_${TASK}_gpu0
    " C-m
done

echo "Started tmux session: $proc_name"
echo "Attach with: tmux attach -t $proc_name"
