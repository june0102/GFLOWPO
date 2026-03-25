proc_name=gpu_3
exp_name=$proc_name
cnt=0
gpus=(3)
n_gpus=${#gpus[@]}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# tasks=(
#     'antonyms'
#     'word_in_context'
#     'rhymes'
#     'num_to_verbal'
#     'cause_and_effect'
#     'larger_animal'
#     'second_word_letter'
#     'taxonomy_animal'

# )
datasets=(
    'rhymes'
    'num_to_verbal'
    'cause_and_effect'
    'larger_animal'
    'second_word_letter'
    'taxonomy_animal'
    'negation'
    'common_concept'
    'diff'
    'translation_en-es'
    'orthography_starts_with'
    'sentiment'
    'informal_to_formal'
    'sum'
    'singular_to_plural'
    'active_to_passive'
    'translation_en-de'
    'sentence_similarity'
    'translation_en-fr'
    'letters_list'
    'first_word_letter'
    'synonyms'
)
# bf=('/home/ubuntu/suhan/GFN_PO_V1/Buffer/ref_bf_rte_default.json' '/home/ubuntu/suhan/GFN_PO_V1/Buffer/default_sst.json' '/home/ubuntu/suhan/GFN_PO_V1/Buffer/default_qnli.json' '/home/ubuntu/suhan/GFN_PO_V1/Buffer/default_mnli.json' '/home/ubuntu/suhan/GFN_PO_V1/Buffer/ref_bf_mrpc_default.json' '/home/ubuntu/suhan/GFN_PO_V1/Buffer/ref_bf_snli.json')

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

# Seed 설정
seed=0
# Initial queue seed prompt (for early condition prompts)
# init_queue_prompt="Given the premise and hypothesis, respond yes if the hypothesis is supported, no if contradicted, or maybe if neutral."

# 기존 세션이 있으면 종료
tmux kill-session -t "$proc_name$cnt" 2>/dev/null || true

tmux new -d -s "$proc_name$cnt"
tmux send-keys -t "$proc_name$cnt" "if [ -f \"\$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/anaconda3/etc/profile.d/conda.sh\"; elif [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; elif command -v conda >/dev/null 2>&1; then source \"\$(conda info --base)/etc/profile.d/conda.sh\"; fi" C-m
tmux send-keys -t "$proc_name$cnt" "conda activate rd_test2" C-m
tmux send-keys -t "$proc_name$cnt" "cd \"$SCRIPT_DIR\"" C-m
tmux send-keys -t "$proc_name$cnt" "export PYTHONPATH=\"$SCRIPT_DIR:\$PYTHONPATH\"" C-m

for i in "${!datasets[@]}"; do
    task="${datasets[$i]}"
    # condition_path="${bf[$i]}"  # bf 배열이 주석 처리되어 있으므로 사용하지 않음
    

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
    --task ii \
    --dataset ${task} \
    --agent_model mistralai/Mistral-7B-Instruct-v0.2 \
    --eval_model google/gemma-1.1-7b-it \
    --train_steps 200 \
    --eval_period 100 \
    --batch_size 4 \
    --grad_acc_steps 4 \
    --max_prompt_length 150 \
    --num_example 5 \
    --gamma 1.0 \
    --reward acc \
    --ema_decay ${ema_decay} \
    --exp_name ${task}_acc_beta_reivision_ema_${ema_decay}_buffer_${train_buffer_update_mode}_online_${online_ratio}_m_start_${offline_start_step}+10_queue_update_mode_${queue_update_mode}_shuffled \
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
    # --init_queue_prompt \"${init_queue_prompt}\"
#tmux send-keys -t "$proc_name$cnt" "tmux kill-session -t $proc_name$cnt" C-m
    # --condition_prompts_path ${condition_path} \
