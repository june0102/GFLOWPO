from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import json
import html
import wandb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup, 
    GenerationConfig
)
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import random
import re

from junmo.dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset,load_bigbench
# from dataset_utils import load_all_dataset, dataset_dicts, load_qa_dataset, qa_dicts, load_generation_dataset
from junmo.ii_utils import load_ii_data, got_example_ii, TASK_TO_METRIC, got_example_bbh


def clean_special_chars(text: str) -> str:
    """Remove markdown special characters like ** from text"""
    # Remove ** (bold markers)
    text = re.sub(r'\*+', '', text)
    # Remove __ (bold/italic markers)
    text = re.sub(r'_+', ' ', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
import utils
from junmo.utils import evaluate_prompts, evaluate_prompts_chunked, evaluate_prompts_chunked_II, lora_to_base, base_to_lora, JsonlLogger


class CheckoutLogitsProcessor:
    def __init__(self, checkout_first):
        assert checkout_first
        self.checkout_first = checkout_first
        self.logit_logs = list()

    def flush(self) -> List[torch.Tensor]:
        last = self.logit_logs
        self.logit_logs = list()
        return last
    
    def __call__(self, input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self.checkout_first and (len(input_ids) > 0):
            return logits
        
        self.logit_logs.append(logits)
        return logits

class GFNEMTrainer():
    def __init__(self, args):
        self.args = args
        print(args.exp_name)
        wandb.init(
            project=args.wandb_project, 
            config=vars(args),
            name=f"{args.exp_name}",
            mode=args.wandb_mode
        )

        '''
        Agent Model for GFN Training
        '''
        self.device = "cuda:0"  # default 0
        config = AutoConfig.from_pretrained(args.agent_model)
        config.use_cache = True
        config._attn_implementation = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            args.agent_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            config=config,
            device_map=self.device
        )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.agent_model,
            padding_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        '''
        Eval Model for Prompt Evaluation
        '''
        eval_model_config = AutoConfig.from_pretrained(args.eval_model_paths)
        self.gen_config = GenerationConfig.from_pretrained(args.eval_model_paths)
        self.eval_model_tokenizer = AutoTokenizer.from_pretrained(
            args.eval_model_paths,
            padding_side="left"
        )
        self.llm = LLM(
            model=args.eval_model_paths,
            tokenizer=args.eval_model_paths,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=0.3,
            max_seq_len_to_capture=eval_model_config.max_position_embeddings,
            max_num_seqs=256,
            max_logprobs=self.eval_model_tokenizer.vocab_size,
            enable_prefix_caching=True,
        )
        # self.params = SamplingParams(
        #     temperature=0.0, 
        #     top_p=self.gen_config.top_p, 
        #     max_tokens=args.eval_model_max_token
        # )
        self.params4logprobs_logits = CheckoutLogitsProcessor(
            checkout_first=True,
        )
        self.params4logprobs = SamplingParams(
            temperature=0.0, 
            top_p=self.gen_config.top_p, 
            max_tokens=1, 
            logits_processors=[self.params4logprobs_logits],
            detokenize=True,
        )
        
        # II 태스크용 텍스트 생성 SamplingParams
        self.params4gen = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=50,
        )

        self.sentence_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", 
            device=self.device
        )

        '''
        Dataset Loading
        '''
        # Load dataset based on task type
        if args.task == 'classification':
            dataset = load_all_dataset(args.dataset)
            train_dataset = dataset[0]
            test_dataset = dataset[2]
            if args.verbalizer is None:
                verbalizer = dataset_dicts(args.dataset)
            else:
                verbalizer = {str(i): v for i, v in enumerate(args.verbalizer)}
            num_labels = len(verbalizer)
            train_dataset, validation_dataset = utils.create_balanced_subset_and_validation(
                train_dataset,
                args.train_data_per_labels * num_labels,
            )
            metrics = None  # classification 태스크는 metrics 사용 안함
        elif args.task == 'qa':
            dataset = load_qa_dataset(args.dataset)
            train_dataset = dataset[4]      # validation split
            test_dataset = dataset[2]       # test split
            validation_dataset = dataset[4]  # validation split (실제 validation 데이터 사용)
            # test_dataset = utils.create_balanced_subset(test_dataset, 100)
            if args.verbalizer is None:
                verbalizer = qa_dicts()
            else:
                verbalizer = {str(i): v for i, v in enumerate(args.verbalizer)}
            num_labels = len(verbalizer)
            metrics = None  # qa 태스크는 metrics 사용 안함
            print(f'Validation size: {len(validation_dataset)}')
        elif args.task == 'ii':
            # Instruction Induction 태스크
            train_dataset, test_dataset, validation_dataset = load_ii_data(args.dataset)
            verbalizer = None  # II는 verbalizer 사용 안함
            metrics = None
            print(f'II Task Metric: {TASK_TO_METRIC.get(args.dataset, "em")}')
        elif args.task == 'bbii' or args.task == 'bbii_tc' or args.task == 'bbii_tg':
            # BigBench 태스크 (bbii_tc: text classification, bbii_tg: text generation)
            metrics, train_dataset, test_dataset, verbalizer, task_prefix = load_bigbench(args.dataset)
            validation_dataset = train_dataset  
            print(f'BBII Task Type: {args.task}')
            print(f'BBII Task Metric: {metrics}')
            print(f'BBII Task Prefix: {task_prefix}')
            print(f'BBII Verbalizer: {verbalizer}')
        
        print(f'Task: {args.task}')
        print(f'Dataset: {args.dataset}')
        print(f'Verbalizer: {verbalizer}')
        print(f'Train size: {len(train_dataset)}')
        print(f'Test size: {len(test_dataset)}')
 
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.verbalizer = verbalizer
        self.metrics = metrics  # bbii 태스크용 metrics 초기화
        
        # Auto-compute beta if not specified (1/train_dataset_size)
        if args.beta is None:
            args.beta = 1.0 / len(train_dataset)
            print(f'Auto-computed beta: {args.beta:.6f} (1/{len(train_dataset)})')

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        t_total = args.train_steps * args.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, args.num_warmup_steps, t_total)
        self.queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
        self.ema_decay = getattr(self.args, "ema_decay", 0.99)
        self.log_z_ema = None
        self.queue_update_mode = "acc"  # Only acc mode supported
        
        # Initialize queue with seed prompt if provided
        init_queue_prompt = getattr(args, "init_queue_prompt", None)
        if init_queue_prompt is not None:
            # 초기 프롬프트를 낮은 점수로 queue에 추가 (나중에 더 좋은 프롬프트로 대체될 수 있도록)
            self.queue.add(0.0, init_queue_prompt, 0)
            print(f"[Queue] Initialized with seed prompt: {init_queue_prompt[:80]}...")

        # --- local text logs ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.local_log_dir = Path.cwd() / "logs" / f"{args.exp_name}_{ts}"
        self.jsonl_logger = JsonlLogger(self.local_log_dir)
        print(f"[local logs] {self.local_log_dir}")

        self.current_prompt = args.init_condition if args.init_condition is not None else ""
        
        # Prompt buffer for saving generated prompts
        self.prompt_buffer = []
        
        # Train buffer for storing all sampled prompts with rewards (for replay/analysis)
        self.train_buffer = []
        self.train_buffer_max_size = getattr(args, "train_buffer_max_size", 10000)  # 최대 버퍼 크기
        self.train_buffer_save = getattr(args, "train_buffer_save", False)
        self.train_buffer_update_mode = "heapq"  # Only heapq mode supported
        
        # For heapq mode: use min-heap based on log_reward (negate for max-heap behavior)
        import heapq
        self.heapq = heapq
        
        # Online/Offline sampling settings
        self.use_offline_sampling = getattr(args, "use_offline_sampling", False)
        self.online_ratio = getattr(args, "online_ratio", 0.5)
        self.offline_start_step = getattr(args, "offline_start_step", 100)
        
        # Initial meta prompt for buffer filling phase
        self.init_meta_prompt = getattr(args, "init_meta_prompt", args.meta_prompt)
        
        # Condition buffer settings (별도 버퍼로 offline sampling 시 사용)
        self.use_condition_buffer = getattr(args, "condition_buffer", False)
        self.condition_buffer = []
        self.condition_buffer_created = False  # offline_start_step에서 한 번만 복사
        
    def add_to_train_buffer(self, samples: List[Dict]):
        """Add samples to train buffer with max size limit (heapq mode only)."""
        for sample in samples:
            if len(self.train_buffer) < self.train_buffer_max_size:
                # (priority, unique_id, sample) 형태로 저장 (priority = accuracy)
                self.heapq.heappush(self.train_buffer, (sample["accuracy"], random.random(), sample))
            else:
                # 현재 가장 낮은 accuracy를 가진 샘플보다 새 샘플이 더 좋으면 교체
                new_acc = sample["accuracy"]
                min_acc = self.train_buffer[0][0]

                if new_acc > min_acc:
                    self.heapq.heapreplace(self.train_buffer, (sample["accuracy"], random.random(), sample))

    def sample_from_train_buffer(self, batch_size: int) -> List[Dict]:
        """Sample batch from train buffer for replay (heapq mode)"""
        if len(self.train_buffer) == 0:
            return []
        sample_size = min(batch_size, len(self.train_buffer))
        # heapq 모드: (accuracy, unique_id, sample) 튜플에서 sample만 추출
        sampled = random.sample(self.train_buffer, sample_size)
        return [item[2] for item in sampled]
    
    def get_train_buffer_as_list(self) -> List[Dict]:
        """Get train buffer as list of dicts (heapq format)"""
        return [item[2] for item in self.train_buffer]
    
    def copy_train_buffer(self) -> List[Dict]:
        """Create a deep copy of train_buffer for condition_buffer"""
        import copy
        buffer_list = self.get_train_buffer_as_list()
        return copy.deepcopy(buffer_list)
    
    def sample_from_condition_buffer(self, batch_size: int) -> List[Dict]:
        """Sample batch from condition buffer for offline replay"""
        if len(self.condition_buffer) == 0:
            return []
        sample_size = min(batch_size, len(self.condition_buffer))
        return random.sample(self.condition_buffer, sample_size)
    
    
    def train(self):
        #example loading section
        t = tqdm(range(1, self.args.train_steps+1), desc="training", dynamic_ncols=True)

        for global_step in t:
            # II 태스크인 경우 got_example_ii 사용
            if self.args.task == 'ii':
                examples = got_example_ii(self.validation_dataset, shot=self.args.num_example)
            elif self.args.task in ['bbii', 'bbii_tc', 'bbii_tg']:
                examples = utils.got_example_bbh(self.train_dataset, self.verbalizer, shot=self.args.num_example, metrics=self.metrics)
            elif self.args.task == 'qa':
             
                examples = utils.got_example_mmlu(self.validation_dataset,self.verbalizer,shot=self.args.num_example)   
         
            else:
                examples = utils.got_example(self.validation_dataset, self.verbalizer, shot=self.args.num_example)
            batch_metrics = defaultdict(list)
            self.model.train()
            self.optimizer.zero_grad()
            
            # offline_start_step 전에는 init_meta_prompt 사용 (버퍼 채우기 단계)
            if global_step < self.offline_start_step:
                current_meta_prompt = self.init_meta_prompt
                                        
                # top_prompts = [
                #     '',
                #     "Read the instruction and write an output for every one of the inputs. Input-Output Pairs: 1. Input: How did the lunar maria form? Output: Large impacts fractured the Moon's lithosphere allowing lava to fill the impact basins. 2. Input: The reason that small planets tend to lose interior heat faster than larger planets is essentially the same as the reason that Output: a large baked potato takes longer to cool than a small baked potato. 3. Input: If the Moon is setting at noon the phase of the Moon must be Output: third quarter.",
                #     "**Write an explanation or answer for each input.** **1. How did the lunar maria form?** **Output:** Large impacts fractured the Moon's lithosphere allowing lava to fill the impact basins. **2. The reason that small planets tend to lose interior heat faster than larger planets is essentially the same as the reason that ________.** **Output:** a large baked potato takes longer to cool than a small baked potato. **3. If the Moon is setting at noon the phase of the Moon must be:** **Output:** third quarter.",
                #     "**Provide a concise explanation for each output.** **1. The Coriolis effect is observed on planets because:** **Output:** They are rotating and spherical so different latitudes rotate at different speeds (meters/second). **Explanation:** The Coriolis effect is a force that deflects moving objects to the right (left in the southern hemisphere) due to the rotation of the planet. This differential rotation causes different latitudes to rotate at different speeds, leading to the observation of the Coriolis effect. **2. A comet’s tail points in the following direction:** **Output:** Away from the Sun. **Explanation:** The tail of a comet points away from the Sun because the gas and dust in the tail are ionized by the",
                #     "**Explain the differences in the planetary phenomena between Venus, Mars, and Earth.** **Based on the input-output pairs, we can conclude:** **1. Venus' lack of seasons:** - Venus has an incredibly slow rotation (takes 243 Earth days), causing its rotational axis to remain nearly perpendicular to the ecliptic, resulting in little variation in day-night temperatures and no distinct seasons. **2. Mercury's lack of a permanent atmosphere:** - Mercury lacks a strong magnetic field and an atmospheric replenishment mechanism, primarily due to its lack of volcanic outgassing and exposure to solar wind, leading to an inability to retain atmosphere. - **Option: Volcanic Heating** is not a mechanism that",
                #     "**Provide explanations or definitions for the given inputs.** **Input 1:** What dangers are currently faced by each Mars Exploration Rover? **Output:** Opportunity may not be able to move its arm again; Spirit may not get enough solar power during the winter months immediately ahead. **Explanation:** * The Martian environment poses significant challenges for the Mars Exploration Rovers. * Opportunity's arm malfunction may limit its ability to conduct scientific experiments. * Spirit's reduced solar energy during the Martian winter could impact its performance. **Input 2:** In astronomy, the concept of black bodies is very important to better calculate the radiation of stars. Which one is the correct definition of a black body? **Output:** An idealized physical",
                #     "**Explain the differences in seasons between Venus, Mars, and Earth.** **Observations:** - Venus does not have distinct seasons like Mars and Earth. - Mercury lacks a permanent atmosphere. - Comets have tails that point away from the Sun. **Conclusion:** **Venus:** - Venus' rotation axis is nearly perpendicular to the plane of the Solar System, resulting in very little axial tilt and no significant changes in the amount of sunlight received throughout the year. **Mercury:** - Mercury's lack of a permanent atmosphere is likely due to its high surface temperature, which drives away any molecules that might attempt to accumulate in the atmosphere. **Comets:** - The tail of a comet points away from the"
                    
                #     ]

                # acc = evaluate_prompts_chunked(
                # top_prompts,
                # self.test_dataset,
                # self.llm,
                # self.eval_model_tokenizer,
                # self.params4logprobs_logits,
                # self.params4logprobs,
                # self.verbalizer.values() if self.verbalizer else [],
                # )
                # print(acc)
                # exit()
            else:
                current_meta_prompt = self.args.meta_prompt

            # condition_buffer 사용 시: offline_start_step에 도달하면 train_buffer를 복사
            if self.use_condition_buffer and not self.condition_buffer_created and global_step >= self.offline_start_step:
                self.condition_buffer = self.copy_train_buffer()
                self.condition_buffer_created = True
                print(f"[Step {global_step}] Created condition_buffer with {len(self.condition_buffer)} samples (copied from train_buffer)")

            
            for _ in range(self.args.grad_acc_steps):
                loss, metrics, decoded_text, prompts_info = self.get_batch_metrics(current_meta_prompt, examples, self.train_dataset, global_step)

                for k, v in metrics.items():
                    batch_metrics[k].append(v)
                
                # pre-step에서는 파라미터 업데이트 안 함 (버퍼 채우기만)
                if global_step >= self.offline_start_step:
                    loss = loss / self.args.grad_acc_steps
                    loss.backward()

            # pre-step에서는 파라미터 업데이트 안 함
            if global_step >= self.offline_start_step:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.max_norm
                )
                self.optimizer.step()
                self.scheduler.step()

            batch_metrics = {
                k: sum(v) / float(len(v))
                for k, v in batch_metrics.items()
            }
            wandb.log(batch_metrics, step=global_step)
            if global_step % 10 == 0:
                self.jsonl_logger.append(
                    "train.jsonl",
                    {"global_step": int(global_step), "prompts": prompts_info}
                )
                # Save LoRA checkpoint every 10 steps
                # self.save_lora_checkpoint(global_step)
            if global_step % self.args.eval_period == 0:
                top_prompts_data = self.queue.get_top_texts()  # [(accuracy, text, ep), ...]
                top_prompts = [item[1] for item in top_prompts_data]  # 텍스트만 추출
                
                # Queue가 비어있으면 (offline_start_step 전이고 init_queue_prompt 없는 경우) eval 스킵
                if len(top_prompts) == 0:
                    print(f"[Step {global_step}] Queue is empty, skipping evaluation")
                    continue
                    
                # 태스크 타입에 따라 평가 방법 선택
                if self.args.task == 'ii' or self.args.task == 'bbii' or self.args.task == 'bbii_tg':
                    # Text Generation 방식 평가 (II, BBII, BBII_TG)
                    acc = evaluate_prompts_chunked_II(
                        top_prompts,
                        self.test_dataset,
                        self.llm,
                        self.eval_model_tokenizer,
                        self.params4gen,  # 텍스트 생성용 SamplingParams
                        self.args.dataset,  # task 이름 (메트릭 결정용)
                    )
                elif self.args.task == 'bbii_tc':
                    # Text Classification 방식 평가 (BBII_TC)
                    acc = evaluate_prompts_chunked(
                        top_prompts,
                        self.test_dataset,
                        self.llm,
                        self.eval_model_tokenizer,
                        self.params4logprobs_logits,
                        self.params4logprobs,
                        self.verbalizer.values() if self.verbalizer else [],
                    )
                else:
                    # top_prompts = [
                    #     '',
                    #     "Read the instruction and write an output for every one of the inputs. Input-Output Pairs: 1. Input: How did the lunar maria form? Output: Large impacts fractured the Moon's lithosphere allowing lava to fill the impact basins. 2. Input: The reason that small planets tend to lose interior heat faster than larger planets is essentially the same as the reason that Output: a large baked potato takes longer to cool than a small baked potato. 3. Input: If the Moon is setting at noon the phase of the Moon must be Output: third quarter.",
                    #     "**Write an explanation or answer for each input.** **1. How did the lunar maria form?** **Output:** Large impacts fractured the Moon's lithosphere allowing lava to fill the impact basins. **2. The reason that small planets tend to lose interior heat faster than larger planets is essentially the same as the reason that ________.** **Output:** a large baked potato takes longer to cool than a small baked potato. **3. If the Moon is setting at noon the phase of the Moon must be:** **Output:** third quarter.",
                    #     "**Provide a concise explanation for each output.** **1. The Coriolis effect is observed on planets because:** **Output:** They are rotating and spherical so different latitudes rotate at different speeds (meters/second). **Explanation:** The Coriolis effect is a force that deflects moving objects to the right (left in the southern hemisphere) due to the rotation of the planet. This differential rotation causes different latitudes to rotate at different speeds, leading to the observation of the Coriolis effect. **2. A comet’s tail points in the following direction:** **Output:** Away from the Sun. **Explanation:** The tail of a comet points away from the Sun because the gas and dust in the tail are ionized by the",
                    #     "**Explain the differences in the planetary phenomena between Venus, Mars, and Earth.** **Based on the input-output pairs, we can conclude:** **1. Venus' lack of seasons:** - Venus has an incredibly slow rotation (takes 243 Earth days), causing its rotational axis to remain nearly perpendicular to the ecliptic, resulting in little variation in day-night temperatures and no distinct seasons. **2. Mercury's lack of a permanent atmosphere:** - Mercury lacks a strong magnetic field and an atmospheric replenishment mechanism, primarily due to its lack of volcanic outgassing and exposure to solar wind, leading to an inability to retain atmosphere. - **Option: Volcanic Heating** is not a mechanism that",
                    #     "**Provide explanations or definitions for the given inputs.** **Input 1:** What dangers are currently faced by each Mars Exploration Rover? **Output:** Opportunity may not be able to move its arm again; Spirit may not get enough solar power during the winter months immediately ahead. **Explanation:** * The Martian environment poses significant challenges for the Mars Exploration Rovers. * Opportunity's arm malfunction may limit its ability to conduct scientific experiments. * Spirit's reduced solar energy during the Martian winter could impact its performance. **Input 2:** In astronomy, the concept of black bodies is very important to better calculate the radiation of stars. Which one is the correct definition of a black body? **Output:** An idealized physical",
                    #     "**Explain the differences in seasons between Venus, Mars, and Earth.** **Observations:** - Venus does not have distinct seasons like Mars and Earth. - Mercury lacks a permanent atmosphere. - Comets have tails that point away from the Sun. **Conclusion:** **Venus:** - Venus' rotation axis is nearly perpendicular to the plane of the Solar System, resulting in very little axial tilt and no significant changes in the amount of sunlight received throughout the year. **Mercury:** - Mercury's lack of a permanent atmosphere is likely due to its high surface temperature, which drives away any molecules that might attempt to accumulate in the atmosphere. **Comets:** - The tail of a comet points away from the"
                        
                    # ]

                    acc = evaluate_prompts_chunked(
                        top_prompts,
                        self.test_dataset,
                        self.llm,
                        self.eval_model_tokenizer,
                        self.params4logprobs_logits,
                        self.params4logprobs,
                        self.verbalizer.values() if self.verbalizer else [],
                    )
                    # print(acc)
                    # exit()
                eval_metrics = {
                    'eval/mean_acc': acc.mean().item(),
                    'eval/max_acc': acc.max().item(),
            
                }
                wandb.log(eval_metrics, step=global_step)
                
                # Queue 프롬프트를 HTML 형식으로 로깅 (각 프롬프트의 eval accuracy + val accuracy + log reward 포함)
                acc_list = acc.detach().cpu().tolist() if hasattr(acc, "detach") else list(acc)
                
                # 각 프롬프트의 validation accuracy 계산
                val_acc_list = [self.compute_val_accuracy(prompt) for prompt in top_prompts]
                
                # 각 프롬프트의 log_reward를 train_buffer에서 찾기
                log_reward_list = []
                for prompt in top_prompts:
                    log_r = None
                    for item in self.train_buffer:
                        if item[2]["prompt"] == prompt:
                            log_r = item[2]["log_reward"]
                            break
                    log_reward_list.append(log_r)
                
                queue_html = f"<h3>Step {global_step} - Queue Prompts</h3>"
                queue_html += "<table border='1'><tr><th>Rank</th><th>Prompt</th><th>Train Acc</th><th>Val Acc</th><th>LogR</th><th>Eval Acc</th><th>Added Step</th></tr>"
                for i, (item, eval_acc, val_acc, log_r) in enumerate(zip(top_prompts_data, acc_list, val_acc_list, log_reward_list)):
                    log_r_str = f"{log_r:.4f}" if log_r is not None else "N/A"
                    queue_html += f"<tr><td>{i+1}</td><td>{html.escape(item[1])}</td><td>{item[0]:.4f}</td><td>{val_acc:.4f}</td><td>{log_r_str}</td><td>{eval_acc:.4f}</td><td>{item[2]}</td></tr>"
                queue_html += "</table>"
                
                wandb.log({
                    "eval/queue_prompts_html": wandb.Html(queue_html)
                }, step=global_step)
                self.jsonl_logger.append(
                    "eval.jsonl",
                    {
                        "global_step": int(global_step),
                        "info": {
                            "top_prompts": top_prompts,
                            "acc": acc_list,
                        },
                    }
                )
        top_prompts_data = self.queue.get_top_texts()  # [(accuracy, text, ep), ...]
        top_prompts = [item[1] for item in top_prompts_data]  # 텍스트만 추출
        # 태스크 타입에 따라 최종 평가 방법 선택
        if self.args.task == 'ii' or self.args.task == 'bbii' or self.args.task == 'bbii_tg':
            # Text Generation 방식 평가 (II, BBII, BBII_TG)
            acc = evaluate_prompts_chunked_II(
                top_prompts,
                self.test_dataset,
                self.llm,
                self.eval_model_tokenizer,
                self.params4gen,  # 텍스트 생성용 SamplingParams
                self.args.dataset,  # task 이름 (메트릭 결정용)
            )
        elif self.args.task == 'bbii_tc':
            # Text Classification 방식 평가 (BBII_TC)
            acc = evaluate_prompts_chunked(
                top_prompts,
                self.test_dataset,
                self.llm,
                self.eval_model_tokenizer,
                self.params4logprobs_logits,
                self.params4logprobs,
                self.verbalizer.values() if self.verbalizer else [],
            )
        else:
            acc = evaluate_prompts_chunked(
                top_prompts,
                self.test_dataset,
                self.llm,
                self.eval_model_tokenizer,
                self.params4logprobs_logits,
                self.params4logprobs,
                self.verbalizer.values() if self.verbalizer else [],
            )
        eval_metrics = {
            'final_max_acc': acc.max().item(),
            'final_mean_acc': acc.mean().item(),
        }
        acc_list = acc.detach().cpu().tolist() if hasattr(acc, "detach") else list(acc)
        
        # Final Queue 프롬프트를 HTML 형식으로 로깅 (각 프롬프트의 eval accuracy + val accuracy + log reward 포함)
        # 각 프롬프트의 validation accuracy 계산
        final_val_acc_list = [self.compute_val_accuracy(prompt) for prompt in top_prompts]
        
        # 각 프롬프트의 log_reward를 train_buffer에서 찾기
        final_log_reward_list = []
        for prompt in top_prompts:
            log_r = None
            for item in self.train_buffer:
                if item[2]["prompt"] == prompt:
                    log_r = item[2]["log_reward"]
                    break
            final_log_reward_list.append(log_r)
        
        final_queue_html = f"<h3>Final Queue Prompts (Step {global_step})</h3>"
        final_queue_html += "<table border='1'><tr><th>Rank</th><th>Prompt</th><th>Train Acc</th><th>Val Acc</th><th>LogR</th><th>Eval Acc</th><th>Added Step</th></tr>"
        for i, (item, eval_acc, val_acc, log_r) in enumerate(zip(top_prompts_data, acc_list, final_val_acc_list, final_log_reward_list)):
            log_r_str = f"{log_r:.4f}" if log_r is not None else "N/A"
            final_queue_html += f"<tr><td>{i+1}</td><td>{html.escape(item[1])}</td><td>{item[0]:.4f}</td><td>{val_acc:.4f}</td><td>{log_r_str}</td><td>{eval_acc:.4f}</td><td>{item[2]}</td></tr>"
        final_queue_html += "</table>"
        
        eval_metrics["final/queue_prompts_html"] = wandb.Html(final_queue_html)
        wandb.log(eval_metrics, step=global_step)
        self.jsonl_logger.append(
            "final.jsonl",
            {
                "global_step": int(global_step),
                "info": {
                    "top_prompts": top_prompts,
                    "acc": acc_list,
                },
            }
        )
    
    def get_batch_metrics(self, meta_prompt, examples, train_dataset, global_step):
        loss, metrics, decoded_text, prompts_info = self.e_step(
            meta_prompt, examples, train_dataset, global_step
        )        
        # Queue 업데이트 (accuracy 기준)
        # offline_start_step 전에는 base model로 buffer만 채우고, queue에는 추가하지 않음
        # queue는 init_queue_prompt로 시작해서 offline_start_step 이후에만 GFlowNet policy 프롬프트 추가
        if global_step >= self.offline_start_step:
            self.queue.add(self.best_accuracy, self.best_candidate, global_step)

        return loss, metrics, decoded_text, prompts_info


    def e_step(self, meta_prompt, examples, train_dataset, global_step):
        """
        온라인/오프라인 샘플링을 하나의 메서드에서 처리
        """
        
        '''
        1. 온라인/오프라인 결정
        '''
        use_offline = False
        if self.use_offline_sampling and global_step >= self.offline_start_step and len(self.train_buffer) > 0:
            prob = random.uniform(0, 1)
            if prob >= self.online_ratio:
                use_offline = True
        
        '''
        2. Condition text 생성 (온라인/오프라인 모두 적용)
        '''
        full_meta_prompt = meta_prompt + "\n" + examples
        condition_text = ""

        if global_step > self.args.offline_start_step + 10:
            top_prompts_data = self.queue.get_top_texts()
            top_prompts_texts = [item[1] for item in top_prompts_data] if top_prompts_data else []
            num_from_queue = min(1, len(top_prompts_texts))
            queue_samples = random.sample(top_prompts_texts, num_from_queue) if num_from_queue > 0 else []
            
           # Condition buffer에서 최대 2개 선택 (condition_buffer 사용 시 condition_buffer에서, 아니면 train_buffer에서)
            if self.use_condition_buffer and len(self.condition_buffer) > 0:
                buffer_prompts = [item["prompt"] for item in self.condition_buffer]
            else:
                buffer_prompts = [item["prompt"] for item in self.get_train_buffer_as_list()] if self.train_buffer else []
            num_from_buffer = min(2, len(buffer_prompts))
            buffer_samples = random.sample(buffer_prompts, num_from_buffer) if num_from_buffer > 0 else []
            
            # 두 소스에서 뽑은 프롬프트를 합쳐서 랜덤 셔플
            combined_samples = queue_samples + buffer_samples
            random.shuffle(combined_samples)
            
            # 특수기호 제거 (**, __ 등)
            combined_samples = [clean_special_chars(p) for p in combined_samples]
            
            # Condition text 생성 (총 최대 3개)
            if len(combined_samples) > 0:
                condition_text += "\nHere are some reference instructions:\n"
                for i, prompt in enumerate(combined_samples, 1):
                    condition_text += f"{i}. {prompt}\n"

        full_meta_prompt = full_meta_prompt + condition_text
        
        '''
        3. Meta prompt 토큰화
        '''
        chat_input = [
            {"role": "user", "content": full_meta_prompt},
            {"role": "assistant", "content": "The Instruction is: "}
        ]
        encoded = self.tokenizer.apply_chat_template(
            chat_input,
            return_tensors='pt',
            add_generation_prompt=False
        ).to(self.device)
        
        '''
        4. 프롬프트 샘플링 (온라인 vs 오프라인)
        '''
        if use_offline:
            # 오프라인: train_buffer에서 프롬프트 가져오기
            buffer_samples = self.sample_from_train_buffer(self.args.batch_size)
            
            if len(buffer_samples) == 0:
                # Buffer가 비어있으면 온라인으로 fallback
                use_offline = False
            else:
                decoded_responses = [sample["prompt"] for sample in buffer_samples]
                sampling_temp = 0.0
                
                # 프롬프트 토큰화
                input_ids = encoded.repeat(len(decoded_responses), 1)
                attention_mask = torch.ones_like(input_ids)
                prompt_batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                
                # Response 토큰화
                self.tokenizer.padding_side = "right"
                response_inputs = self.tokenizer(
                    decoded_responses,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.device)
                self.tokenizer.padding_side = "left"
                
                # EOS 토큰 추가
                eos_tokens = torch.ones(response_inputs["input_ids"].size(0), 1, dtype=torch.long, device=self.device) * self.tokenizer.eos_token_id
                response_ids = torch.cat([response_inputs["input_ids"], eos_tokens], dim=1)
                response_mask = torch.cat([response_inputs["attention_mask"], torch.ones_like(eos_tokens)], dim=1)
                
                response_batch = {
                    "input_ids": response_ids,
                    "attention_mask": response_mask
                }
                
                # prompt + response 결합
                prompt_responses = torch.cat([input_ids, response_ids], dim=1)
        
        if not use_offline:
            # 온라인: 모델로 새 프롬프트 생성
            input_ids = encoded.repeat(self.args.batch_size, 1)
            attention_mask = torch.ones_like(input_ids)
            prompt_batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            prompt_responses, only_responses, decoded_responses, sampling_temp = self.sample_prompts(prompt_batch)
            
            # Response 후처리
            eos_tokens = torch.ones(input_ids.size(0), 1, dtype=torch.long) * self.tokenizer.eos_token_id
            eos_tokens = eos_tokens.to(input_ids.device)
            only_responses = torch.cat([only_responses, eos_tokens], 1)
            responses = only_responses.cpu()
            pad_mask = (responses == self.tokenizer.eos_token_id).cumsum(1) > 1
            response_lengths = torch.sum((~pad_mask).long(), 1)
            response_ids = []
            for i in range(prompt_responses.size(0)):
                response_len = response_lengths[i].item()
                response_ids.append(responses[i, :response_len])

            response_mask = [torch.ones_like(x) for x in response_ids]
            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
            response_mask = pad_sequence(response_mask, batch_first=True, padding_value=0)
            response_batch = {
                "input_ids": response_ids,
                "attention_mask": response_mask
            }

        '''
        compute q(theta'|full_meta_prompt) -- no condition prompt right now only e-step
        '''
        sum_logpf = self.get_logpf(
            prompt_batch, 
            response_batch
        )  # [batch_size,]

        '''
        compute log_prior 
        (TODO): get_log_prior and get_logpf are basically doing the same thing, needs more refactoring
        '''
        log_prior = self.get_log_prior(
            prompt_responses,
            prompt_len=input_ids.size(1),
            attention_mask=attention_mask,
        )  # [batch_size,]

        '''
        compute log likelihood 
        '''
        # 태스크 타입에 따라 log-likelihood 및 accuracy 계산 방법 선택
        if self.args.task == 'ii' or self.args.task == 'bbii' or self.args.task == 'bbii_tg':
            # Text Generation 방식 (II, BBII, BBII_TG)
            log_ll, acc = self.compute_ll_ii(decoded_responses, train_dataset)  # [batch_size,], [batch_size,]
        elif self.args.task == 'bbii_tc':
            # Text Classification 방식 (BBII_TC) - verbalizer 기반
            log_ll, acc = self.compute_ll_tc(decoded_responses, train_dataset)  # [batch_size,], [batch_size,]
        else:
            log_ll, acc = self.compute_ll_tc(decoded_responses, train_dataset)  # [batch_size,], [batch_size,]
        
        '''
        compute loss
        '''
        gamma = self.get_lm_reward_temp(global_step)
        beta = self.args.beta
        
        '''
        compute reward (acc mode only)
        '''
        # Use log accuracy as reward term
        log_accuracy = self.compute_log_acc_reward(acc)
        log_reward = (log_prior / gamma) + (log_accuracy / beta)
        
        # EMA
        log_z_batch = (log_reward - sum_logpf.detach()).mean()
        
        # EMA 업데이트: z_ema = decay * z_ema_prev + (1 - decay) * z_batch
        if self.log_z_ema is None:
            self.log_z_ema = log_z_batch.item()
        else:
            self.log_z_ema = self.ema_decay * self.log_z_ema + (1 - self.ema_decay) * log_z_batch.item()
        
        # TB Loss는 EMA log_z 사용
        log_z = log_reward.new_full((log_reward.size(0),), self.log_z_ema)

        loss = self.compute_tb_loss(
            log_z, sum_logpf, log_reward
        )

        # Select best candidate based on accuracy
        best_idx = acc.argmax().item()
        self.best_candidate = decoded_responses[best_idx]
        self.best_accuracy = acc[best_idx].item()
        self.best_log_reward = log_reward[best_idx].item()

        self.current_prompt = self.best_candidate

        # Save all generated prompts to prompt_buffer (full info for logging)
        for i, prompt_text in enumerate(decoded_responses):
            self.prompt_buffer.append({
                "global_step": global_step,
                "prompt": prompt_text,
                "accuracy": acc[i].item(),
                "log_reward": log_reward[i].item(),
                "log_ll": log_ll[i].item(),
                "log_prior": log_prior[i].item(),
                "sum_logpf": sum_logpf[i].item(),
            })
        
        # Save to train_buffer (온라인 샘플링일 경우에만 저장)
        if not use_offline:
            batch_samples = []
            for i, prompt_text in enumerate(decoded_responses):
                batch_samples.append({
                    "prompt": prompt_text,
                    "accuracy": acc[i].item(),
                    "log_reward": log_reward[i].item(),
                    "log_ll": log_ll[i].item(),
                    "log_prior": log_prior[i].item(),
                })
            self.add_to_train_buffer(batch_samples)

        # Metrics (온라인/오프라인 통합)
        step_prefix = 'offline_step' if use_offline else 'e_step'
        metrics = {
            f'{step_prefix}/tb_loss': loss.item(),
            f'{step_prefix}/log_z': self.log_z_ema,
            f'{step_prefix}/sum_logpf': sum_logpf.mean().item(),
            f'{step_prefix}/log_prior': log_prior.mean().item(),
            f'{step_prefix}/log_reward': log_reward.mean().item(),
            f'{step_prefix}/log_accuracy': log_accuracy.mean().item(),
            f'{step_prefix}/accuracy': acc.mean().item(),
            'sampling/use_offline': 1.0 if use_offline else 0.0,
            'sampling/buffer_size': len(self.train_buffer),
        }

        source_tag = "[OFFLINE]" if use_offline else "[ONLINE]"
        decoded_text = f"{source_tag} GFN meta prompt:\n {full_meta_prompt}\nGenerated responses:\n"
        for i, r in enumerate(decoded_responses):
            decoded_text += f"{i+1}. {r} (train accuracy: {acc[i].item()})\n"
        
        # Prompts info for JSON logging
        prompts_info = [
            {"prompt": decoded_responses[i], "train_acc": acc[i].item(), "log_reward": log_reward[i].item(), "source": "offline" if use_offline else "online"}
            for i in range(len(decoded_responses))
        ]
        
        # Wandb logging every 10 steps (HTML format)
        if global_step % 10 == 0:
            sampling_mode = "Offline (Buffer)" if use_offline else "Online (Generate)"
            prompts_html = f"<h3>Step {global_step} - {sampling_mode}</h3>"
            
            # Meta Prompt를 더 명확하게 표시 (reference instructions 포함)
            prompts_html += f"<details><summary>Full Meta Prompt (click to expand)</summary>"
            prompts_html += f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{html.escape(full_meta_prompt)}</pre>"
            prompts_html += "</details>"
            
            # Reference Instructions를 별도로 표시 (있는 경우)
            if global_step > 10 and condition_text:
                prompts_html += f"<details><summary>Reference Instructions Used (click to expand)</summary>"
                prompts_html += f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{html.escape(condition_text)}</pre>"
                prompts_html += "</details>"
            
            # 각 샘플링된 프롬프트의 validation accuracy 계산
            sampled_val_acc_list = [self.compute_val_accuracy(prompt) for prompt in decoded_responses]
            
            prompts_html += "<table border='1'><tr><th>Idx</th><th>Prompt</th><th>Train Acc</th><th>Val Acc</th><th>LogR</th></tr>"
            for idx, prompt in enumerate(decoded_responses):
                prompts_html += f"<tr><td>{idx}</td><td>{html.escape(prompt)}</td><td>{acc[idx].item():.4f}</td><td>{sampled_val_acc_list[idx]:.4f}</td><td>{log_reward[idx].item():.4f}</td></tr>"
            prompts_html += "</table>"
            
            wandb.log({
                "sampling/prompts_html": wandb.Html(prompts_html),
                "sampling/temperature": sampling_temp,
                "sampling/mean_val_acc": sum(sampled_val_acc_list) / len(sampled_val_acc_list) if sampled_val_acc_list else 0.0,
            }, step=global_step)
        
        return loss, metrics, decoded_text, prompts_info
    

    @torch.inference_mode(True)
    def sample_prompts(self, prompt_batch):
        temp = random.uniform(self.args.temp_low, self.args.temp_high)
        prompt_responses = self.model.generate(
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            do_sample=True,
            max_new_tokens=self.args.max_prompt_length,
            temperature=temp,
            top_p=0.9,
            # min_new_tokens=self.args.min_len,
            pad_token_id=self.tokenizer.pad_token_id
        )  # [batch_size, prompt_token_len + response_len]  note that no eos_token for minimum length response

        prompt_len = prompt_batch["input_ids"].size(1)
        only_responses = prompt_responses[:, prompt_len:]  # [batch_size, response_len] no eos_token for minimum length response
        decoded_responses = self.tokenizer.batch_decode(only_responses, skip_special_tokens=True)
        return prompt_responses, only_responses, decoded_responses, temp


    def get_logpf(self, prompt_batch, response_batch):
        prompt_len = prompt_batch["input_ids"].size(1)

        # gpu allocation
        prompt_batch = {k: v.to(self.device)
                        for k, v in prompt_batch.items()}
        response_batch = {k: v.to(self.device)
                          for k, v in response_batch.items()}

        concat_inputs = dict()
        for k in prompt_batch.keys():
            concat_inputs[k] = torch.cat(
                [prompt_batch[k], response_batch[k]], 1)
        
        # outputs = self.model(**concat_inputs, output_hidden_states=True)
        outputs = self.model(**concat_inputs)
        
        logits = outputs.logits[:, prompt_len-1:-1] 
        responses = response_batch["input_ids"]

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = torch.gather(
            log_prob, -1, responses.unsqueeze(-1)).squeeze(-1)
        log_prob = log_prob.masked_fill(
            response_batch["attention_mask"] == 0, 0.0)
        sum_logpf = torch.sum(log_prob, dim=1)
        
        return sum_logpf

    
    @torch.inference_mode(True)
    def get_log_prior(self, prompt_responses, prompt_len, attention_mask):
        only_responses = prompt_responses[:, prompt_len:]
        # the first pad token is EOS / note that pad_token_id == eos_token_id
        pad_mask = (only_responses == self.tokenizer.pad_token_id).cumsum(1) > 1
        attention_mask = torch.cat([attention_mask, (~pad_mask).long()], 1)
        # llh from reference model
        lora_to_base(self.model)
        outputs = self.model(
            input_ids=prompt_responses,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[:, prompt_len-1:-1]
        log_prob = F.log_softmax(logits, dim=-1)
        labels = prompt_responses[:, prompt_len:]

        log_prior = torch.gather(log_prob, -1, labels.unsqueeze(2)).squeeze(2)
        log_prior = torch.where(pad_mask, 0.0, log_prior)
        log_prior = torch.sum(log_prior, 1)  # [batch_size,]
        base_to_lora(self.model)
        del outputs, logits, log_prob
        return log_prior


    @torch.inference_mode(True)
    def compute_ll_tc(self, prompts, dataset):
        def _format_prompts(prompts, inputs,side):
            if side == 'First':
                template = "{prompt} Input : {sentence_1} Output:"
            else:
                template = "{sentence_1} {prompt}"
            return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]
        
        verbalizer = self.verbalizer.values() if self.verbalizer else []
        verbalizer_ids = self.eval_model_tokenizer.convert_tokens_to_ids(verbalizer)
        dataset_len = len(dataset)
        dataloader = DataLoader(dataset, batch_size=dataset_len, shuffle=False, drop_last=False)
        batch = next(iter(dataloader))
        if 'text' in batch.keys():
            inputs = batch['text']
        else:
            inputs = batch['sentence']
        targets = batch['label']

        all_templates = []
        for prompt in prompts:
            current_prompts = [prompt for _ in range(dataset_len)]
            formatted_templates = _format_prompts(current_prompts, inputs, side="First")
            all_templates.extend(formatted_templates)
        # print(len(all_templates))
        
        with torch.no_grad():
            torch.cuda.synchronize()
            logits = self.params4logprobs_logits.flush()
            outputs = self.llm.generate(
                all_templates, 
                self.params4logprobs,
                use_tqdm=False
            )
            logits = self.params4logprobs_logits.flush()  # list of (vocab_size,) tensors
            all_logits = torch.stack(logits, dim=0)  # (batch_size, vocab_size)
        # print(all_logits.shape)

        verbalizer_logits = all_logits[..., verbalizer_ids]
        log_probs = F.log_softmax(verbalizer_logits, dim=1).reshape(len(prompts), dataset_len, -1)
        preds = torch.argmax(log_probs, dim=-1).cpu()
        targets_expanded = targets.unsqueeze(0).expand(len(prompts), -1)
        correct_per_prompt = torch.sum(preds == targets_expanded, dim=1)
        acc_per_prompt = correct_per_prompt / dataset_len
        # print(acc_per_prompt)

        targets_idx = targets.to(log_probs.device).long()          # (D,)
        targets_idx = targets_idx.unsqueeze(0).expand(len(prompts), -1)  # (P, D)
        selected_logp = log_probs.gather(dim=-1, index=targets_idx.unsqueeze(-1)).squeeze(-1)
        ll_per_prompt = selected_logp.sum(dim=1)                  # (P,)
        del outputs, logits, log_probs
        # Ensure acc_per_prompt is on the same device as ll_per_prompt
        acc_per_prompt = acc_per_prompt.to(ll_per_prompt.device)
        return ll_per_prompt, acc_per_prompt


    @torch.inference_mode(True)
    def compute_ll_ii(self, prompts, dataset):
        """
        Instruction Induction 태스크용 log-likelihood 및 accuracy 계산.
        
        TC와 달리:
        - Verbalizer 기반이 아닌 전체 vocab 기준 log-prob 계산
        - Ground truth label 전체 시퀀스에 대한 log-likelihood 계산
        - Metric 기반 accuracy (f1, em, es, contains)
        
        Note: reward='acc'인 경우 log-likelihood 계산을 스킵하여 메모리 절약
        
        Args:
            prompts: 프롬프트 리스트
            dataset: 데이터셋 (text, label 포함)
            
        Returns:
            ll_per_prompt: 프롬프트별 log-likelihood (shape: (P,))
            acc_per_prompt: 프롬프트별 accuracy (shape: (P,))
        """
        from junmo.ii_utils import TASK_TO_METRIC, default_metric, get_f1_score, get_em_score, get_exact_set_score, get_contains_score
        
        dataset_len = len(dataset)
        dataloader = DataLoader(dataset, batch_size=dataset_len, shuffle=False, drop_last=False)
        batch = next(iter(dataloader))
        inputs = batch['text'] if 'text' in batch else batch['sentence']
        labels = batch['label']  # ground truth (문자열 리스트)
        
        num_prompts = len(prompts)
        
        # 메트릭 결정
        metric = TASK_TO_METRIC.get(self.args.dataset, default_metric)
        
        # ========== 1. Log-Likelihood 계산 ==========
        # reward='acc'인 경우 log-likelihood 계산을 스킵 (메모리 절약)
        if self.args.reward == 'acc':
            # log_ll은 사용되지 않으므로 0으로 설정
            ll_per_prompt = torch.zeros(num_prompts, dtype=torch.float32, device=self.device)
        else:
            # Ground truth를 포함한 전체 시퀀스의 log-prob 계산
            # 템플릿: "{prompt}\nInput : {input}\nOutput : {label}"
            
            all_templates_with_label = []
            label_start_positions = []  # 각 템플릿에서 label이 시작하는 토큰 위치
            
            for prompt in prompts:
                for i in range(dataset_len):
                    # label 없는 템플릿 (label 시작 위치 계산용)
                    template_without_label = f"{prompt}\nInput : {inputs[i]}\nOutput : "
                    # label 포함 템플릿
                    template_with_label = f"{prompt}\nInput : {inputs[i]}\nOutput : {labels[i]}"
                    
                    all_templates_with_label.append(template_with_label)
                    
                    # label 시작 위치 계산 (토큰 수)
                    tokens_without = self.eval_model_tokenizer.encode(template_without_label, add_special_tokens=False)
                    label_start_positions.append(len(tokens_without))
            
            # vLLM prompt_logprobs를 사용하여 각 토큰의 log-prob 계산
            sampling_params_logprobs = SamplingParams(
                max_tokens=1,  # 생성은 최소로
                temperature=0.0,
                prompt_logprobs=1,  # 프롬프트 토큰들의 logprobs 계산
            )
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    all_templates_with_label,
                    sampling_params_logprobs,
                    use_tqdm=False
                )
            
            # 각 샘플의 label 부분 log-prob 합계 계산
            ll_list = []
            for idx, output in enumerate(outputs):
                prompt_logprobs = output.prompt_logprobs  # List[Dict] or None
                label_start = label_start_positions[idx]
                
                if prompt_logprobs is None:
                    ll_list.append(0.0)
                    continue
                
                # label 부분의 log-prob 합산 (label_start 이후)
                label_ll = 0.0
                for pos in range(label_start, len(prompt_logprobs)):
                    if prompt_logprobs[pos] is not None:
                        # prompt_logprobs[pos]는 Dict[token_id, Logprob] 형태
                        # 해당 위치의 실제 토큰의 log-prob 가져오기
                        logprob_dict = prompt_logprobs[pos]
                        if logprob_dict:
                            # 가장 높은 확률의 토큰 (실제 입력된 토큰)의 log-prob
                            label_ll += list(logprob_dict.values())[0].logprob
                
                ll_list.append(label_ll)
            
            # (P * D) -> (P, D) -> sum over D -> (P,)
            ll_tensor = torch.tensor(ll_list, dtype=torch.float32).reshape(num_prompts, dataset_len)
            ll_per_prompt = ll_tensor.sum(dim=1).to(self.device)  # (P,)
        
        # ========== 2. Accuracy 계산 (텍스트 생성 + metric 평가) ==========
        # 생성용 SamplingParams
        sampling_params_gen = SamplingParams(
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
        )
        
        all_templates = []
        for prompt in prompts:
            for i in range(dataset_len):
                template = f"{prompt}\nInput : {inputs[i]}\nOutput : "
                all_templates.append(template)
        
        with torch.no_grad():
            gen_outputs = self.llm.generate(
                all_templates,
                sampling_params_gen,
                use_tqdm=False
            )
        
        # 메트릭 계산
        scores_per_prompt = torch.zeros(num_prompts, dtype=torch.float32)
        
        for idx, output in enumerate(gen_outputs):
            p = idx // dataset_len
            d = idx % dataset_len
            
            generated_text = output.outputs[0].text
            ground_truth = labels[d]
            
            if metric == 'f1':
                score = get_f1_score(generated_text, ground_truth)
            elif metric == 'em':
                score = get_em_score(generated_text, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(generated_text, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(generated_text, ground_truth)
            else:
                score = get_em_score(generated_text, ground_truth)  # default
            
            scores_per_prompt[p] += score
        
        acc_per_prompt = scores_per_prompt / float(dataset_len)
        
        # device 맞추기
        ll_per_prompt = ll_per_prompt.to(self.device)
        acc_per_prompt = acc_per_prompt.to(self.device)
        
        return ll_per_prompt, acc_per_prompt
    

    @torch.inference_mode(True)
    def compute_val_accuracy(self, prompt: str) -> float:
        """
        단일 프롬프트에 대해 validation dataset에서 accuracy를 계산합니다.
        
        Args:
            prompt: 평가할 프롬프트
            
        Returns:
            val_accuracy: validation accuracy (0.0 ~ 1.0)
        """
        if self.args.task == 'ii' or self.args.task == 'bbii' or self.args.task == 'bbii_tg':
            return self._compute_val_accuracy_ii(prompt)
        else:
            return self._compute_val_accuracy_tc(prompt)
    
    @torch.inference_mode(True)
    def _compute_val_accuracy_tc(self, prompt: str) -> float:
        """Text Classification 방식으로 validation accuracy 계산"""
        def _format_prompts(prompt, inputs, side):
            if side == 'First':
                template = "{prompt} Input : {sentence_1} Output:"
            else:
                template = "{sentence_1} {prompt}"
            return [template.format(sentence_1=s_1, prompt=prompt) for s_1 in inputs]
        
        verbalizer = self.verbalizer.values() if self.verbalizer else []
        verbalizer_ids = self.eval_model_tokenizer.convert_tokens_to_ids(verbalizer)
        dataset_len = len(self.validation_dataset)
        dataloader = DataLoader(self.validation_dataset, batch_size=dataset_len, shuffle=False, drop_last=False)
        batch = next(iter(dataloader))
        if 'text' in batch.keys():
            inputs = batch['text']
        else:
            inputs = batch['sentence']
        targets = batch['label']
        
        formatted_templates = _format_prompts(prompt, inputs, side="First")
        
        with torch.no_grad():
            torch.cuda.synchronize()
            _ = self.params4logprobs_logits.flush()
            outputs = self.llm.generate(
                formatted_templates, 
                self.params4logprobs,
                use_tqdm=False
            )
            logits = self.params4logprobs_logits.flush()
            all_logits = torch.stack(logits, dim=0)
        
        verbalizer_logits = all_logits[..., verbalizer_ids]
        log_probs = F.log_softmax(verbalizer_logits, dim=1)
        preds = torch.argmax(log_probs, dim=-1).cpu()
        correct = torch.sum(preds == targets).item()
        val_accuracy = correct / dataset_len
        
        del outputs, logits, log_probs
        return val_accuracy
    
    @torch.inference_mode(True)
    def _compute_val_accuracy_ii(self, prompt: str) -> float:
        """Instruction Induction 방식으로 validation accuracy 계산 (텍스트 생성 + metric)"""
        from junmo.ii_utils import TASK_TO_METRIC, default_metric, get_f1_score, get_em_score, get_exact_set_score, get_contains_score
        
        dataset_len = len(self.validation_dataset)
        dataloader = DataLoader(self.validation_dataset, batch_size=dataset_len, shuffle=False, drop_last=False)
        batch = next(iter(dataloader))
        inputs = batch['text'] if 'text' in batch else batch['sentence']
        labels = batch['label']
        
        metric = TASK_TO_METRIC.get(self.args.dataset, default_metric)
        
        # 생성용 SamplingParams
        sampling_params_gen = SamplingParams(
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
        )
        
        all_templates = [f"{prompt}\nInput : {inputs[i]}\nOutput : " for i in range(dataset_len)]
        
        with torch.no_grad():
            gen_outputs = self.llm.generate(
                all_templates,
                sampling_params_gen,
                use_tqdm=False
            )
        
        # 메트릭 계산
        total_score = 0.0
        for idx, output in enumerate(gen_outputs):
            generated_text = output.outputs[0].text
            ground_truth = labels[idx]
            
            if metric == 'f1':
                score = get_f1_score(generated_text, ground_truth)
            elif metric == 'em':
                score = get_em_score(generated_text, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(generated_text, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(generated_text, ground_truth)
            else:
                score = get_em_score(generated_text, ground_truth)
            
            total_score += score
        
        val_accuracy = total_score / float(dataset_len)
        return val_accuracy


    def get_lm_reward_temp(self, step: int) -> float:
        diff = self.args.lm_sched_end - self.args.lm_sched_start
        temp = self.args.lm_sched_start + diff * min(1, step / self.args.lm_sched_horizon)
        return temp

    def compute_log_acc_reward(self, acc: torch.Tensor) -> torch.Tensor:
        """
        Compute log accuracy reward with epsilon for numerical stability.
        
        Args:
            acc: (batch_size,) accuracy values in [0, 1]
            
        Returns:
            log_acc: (batch_size,) log(acc + epsilon)
        """
        epsilon = self.args.reward_epsilon
        log_acc = torch.log(acc + epsilon)
        return log_acc


    def compute_tb_loss(
        self,
        log_z: torch.Tensor,
        sum_logpf: torch.Tensor,
        log_reward: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Trajectory Balance loss.
        
        Args:
            log_z: (batch_size,) partition function estimate
            sum_logpf: (batch_size,) sum of log probabilities from policy
            log_reward: (batch_size,) log reward (already computed as log_prior/gamma + log_accuracy/beta)
        """
        delta = log_z + sum_logpf - log_reward
        losses = delta ** 2
        return losses.mean()
    

    