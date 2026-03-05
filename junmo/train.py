'''
CUDA_VISIBLE_DEVICES=0 python -m junmo.train --task classification --dataset mnli --agent_model google/gemma-1.1-2b-it --eval_model google/gemma-1.1-2b-it --epochs 200 --eval_interval 5 --batch_size 48 --max_prompt_length 150 --num_example 5 --prompt_per_example 10 --beta 1.0 --gamma 1.0 --prior_scale 1.0 --lr 1e-5 --m_step_freq 1 --m_step --init_condition "Given the premise and hypothesis, respond 'yes' if the hypothesis is supported, 'no' if contradicted, or 'maybe' if neutral." 
CUDA_VISIBLE_DEVICES=0 python -m junmo.train --task classification --dataset mnli --agent_model google/gemma-1.1-7b-it --eval_model google/gemma-1.1-7b-it --train_steps 1000 --eval_period 50 --batch_size 4 --grad_acc_steps 4 --max_prompt_length 150 --num_example 5 --beta 1.0 --gamma 1.0 --lr 1e-4 --m_step_freq 1 --m_step --wandb_mode online
'''
import os
os.environ["VLLM_USE_V1"]="0"
import argparse
from junmo.utils import seed, load_eval_model_config
from junmo.trainer.gfn_em_ema_revision import GFNEMTrainer as GFNEMEMARevisionTrainer


def parser_args():
    parser = argparse.ArgumentParser(description="StablePrompt with GFlowNet Optimizer")
    parser.add_argument('--eval_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--agent_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('~/.cache/huggingface'))
    
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--wandb_project', type=str, default='gfn_em_tc')
    parser.add_argument('--exp_name', type=str, default="")
    
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'qa', 'ii', 'bbii', 'bbii_tc', 'bbii_tg'])
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--verbalizer', type=str, nargs='+', default=None)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_prompt_length', type=int, default=100)
    parser.add_argument('--train_data_per_labels', type=int, default=16)
    parser.add_argument('--num_example', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--meta_prompt', type=str,
                       default='I gave a friend an instruction and three inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs: \n')

    # GFN hyperparameters
    parser.add_argument('--beta', type=float, default=None,
                       help='Temperature for log accuracy term. If None, auto-computed as 1/train_dataset_size')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Temperature for prior term')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--m_step_freq', type=int, default=1,
                       help='M-step frequency (every N steps)')
    
    # Sampling temperature range
    parser.add_argument('--temp_low', type=float, default=0.5,
                       help='Lower bound of sampling temperature (uniform random)')
    parser.add_argument('--temp_high', type=float, default=2.0,
                       help='Upper bound of sampling temperature (uniform random)')
    
    # Temperature scheduling
    parser.add_argument('--reward_sched_start', type=float, default=1.0)
    parser.add_argument('--reward_sched_end', type=float, default=1.0)
    parser.add_argument('--reward_sched_horizon', type=int, default=1000)
    parser.add_argument('--lm_sched_start', type=float, default=1.0)
    parser.add_argument('--lm_sched_end', type=float, default=1.0)
    parser.add_argument('--lm_sched_horizon', type=int, default=1000)
    
    # Evaluation settings
    parser.add_argument('--num_test_example', type=int, default=20)
    
    # LoRA settings
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    # Initial condition for best instruction
    parser.add_argument('--init_condition', type=str, default=None,
                       help='Initial best instruction to condition the model on. '
                            'If None, no initial conditioning is applied.')
    
    # M-step control
    parser.add_argument('--m_step', action='store_true',
                       help='Enable M-step. If not set, only E-step is performed '
                            'and conditioning text is not added to meta-prompt.')
    
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--eval_model_max_token', type=int, default=256)
    parser.add_argument('--train_steps', type=int, default=200)
    parser.add_argument('--grad_acc_steps', type=int, default=8)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--topk', type=int, default=-1)
    
    parser.add_argument('--reward_epsilon', type=float, default=1e-8,
                       help='Epsilon for log accuracy reward to avoid log(0)')
    parser.add_argument('--reward', type=str, default='acc',
                       choices=['acc', 'posterior'],
                       help='Reward type: "acc" for log accuracy, "posterior" for full posterior reward')
    
    # EMA settings
    parser.add_argument('--ema_decay', type=float, default=0.99,
                       help='EMA decay for log_z (z_ema = decay * z_ema_prev + (1-decay) * z_batch)')
    
    # Initial queue seed prompt
    parser.add_argument('--init_queue_prompt', type=str, default=None,
                       help='Initial prompt to seed the queue for early condition prompts. '
                            'Example: "Given the premise and hypothesis, respond yes if supported, no if contradicted, or maybe if neutral."')
    
    # Train buffer settings
    parser.add_argument('--train_buffer_max_size', type=int, default=10000,
                       help='Maximum size of train buffer for experience replay')
    parser.add_argument('--train_buffer_path', type=str, default=None,
                       help='Path to load existing train buffer from JSON file')
    parser.add_argument('--train_buffer_save', action='store_true',
                       help='Save train buffer to JSON file periodically and at the end')
    parser.add_argument('--train_buffer_update_mode', type=str, default='heapq',
                       help='Train buffer update mode (only heapq supported: keep top accuracy samples)')
    
    # Online/Offline sampling settings
    parser.add_argument('--use_offline_sampling', action='store_true',
                       help='Enable offline sampling from train buffer (mixed with online sampling)')
    parser.add_argument('--online_ratio', type=float, default=0.5,
                       help='Ratio of online sampling vs offline sampling (0.0 = all offline, 1.0 = all online)')
    parser.add_argument('--offline_start_step', type=int, default=100,
                       help='Start using offline sampling after this step (need buffer to be filled first)')
    
    # Initial meta prompt for buffer filling phase (before offline_start_step)
    parser.add_argument('--init_meta_prompt', type=str,
                       default='I gave a friend an instruction and three inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs: \\n',
                       help='Meta prompt used during buffer filling phase (before offline_start_step)')
    # Condition buffer settings
    parser.add_argument('--condition_buffer', action='store_true',
                       help='If set, copy train_buffer to condition_buffer at offline_start_step '
                            'and use condition_buffer for offline sampling instead of train_buffer')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    load_eval_model_config(args)
    seed(args.seed)
    options = {
        'data': args.dataset,
        'seed': args.seed,
    }
    exp_name = f"{args.exp_name}"
    for key, value in options.items():
        exp_name += f"{key}:{value}-"
    args.exp_name = exp_name
    trainer = GFNEMEMARevisionTrainer(args)
    trainer.train()


