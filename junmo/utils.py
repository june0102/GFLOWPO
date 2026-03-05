import gzip
import heapq
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from pathlib import Path

# II (Instruction Induction) 평가용 함수들
from junmo.ii_utils import (
    TASK_TO_METRIC,
    default_metric,
    get_f1_score,
    get_em_score,
    get_exact_set_score,
    get_contains_score,
)


class JsonlLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def append(self, filename: str, record: dict):
        path = self.log_dir / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def lora_to_base(model):
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()


def base_to_lora(model):
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()


def seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_eval_model_config(args):
    if args.eval_model == "google/gemma-1.1-2b-it":
        args.eval_tokenizer_paths = args.eval_model
        args.eval_tokenizer_kwargs = {"use_fast": False, "add_bos_token": False}
        args.eval_model_paths = args.eval_model
        args.eval_model_kwargs = {"low_cpu_mem_usage": False, "use_cache": False}
        args.eval_conversation_templates = "gemma-1.1"   
        args.eval_gpu_memory_utilization = 0.9
    elif args.eval_model == "google/gemma-1.1-7b-it":
        args.eval_tokenizer_paths = args.eval_model
        args.eval_tokenizer_kwargs = {"use_fast": False, "add_bos_token": False}
        args.eval_model_paths = args.eval_model
        args.eval_model_kwargs = {"low_cpu_mem_usage": False, "use_cache": False}
        args.eval_conversation_templates = "gemma-1.1"   
        args.eval_gpu_memory_utilization = 0.9


@torch.inference_mode()
def evaluate_prompts_chunked(
    prompts,
    dataset,
    model,  # vLLM LLM
    tokenizer,
    params4logprobs_logits,
    params4logprobs,
    verbalizer=('Yes', 'No', 'Maybe'),
    side='First',
    chunk_size=512,   # ⭐ 여기 조절 (256/512/1024 등)
):
    def _format_one(prompt, sentence_1, side):
        if side == 'First':
            return f"{prompt} Input : {sentence_1} Output:"
        else:
            return f"{sentence_1} {prompt}"

    verbalizer_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(verbalizer)),
        dtype=torch.long
    )  # (V,)  V=3
    
    # 한번에 dataset 다 꺼내는 건 OK (텍스트라 CPU)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
    batch = next(iter(dataloader))
    inputs = batch['text'] if 'text' in batch else batch['sentence']
    targets = batch['label']  # (D,)
    dataset_len = len(inputs)
    num_prompts = len(prompts)

    # prompt별 정답 개수 누적
    correct = torch.zeros(num_prompts, dtype=torch.long)

    # (P, D)로 targets broadcast용
    targets_expanded = targets.unsqueeze(0).expand(num_prompts, -1)  # (P, D)

    # 전체 요청을 "prompt 먼저, 그 다음 dataset 순서"로 쌓아둔다고 가정
    # idx -> (p, d) 매핑: global_idx = p*D + d
    total = num_prompts * dataset_len

    # chunk 단위로 templates 만들고 바로 generate
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)

        # 이번 chunk에 해당하는 templates 생성
        templates = []
        pd_pairs = []  # (p, d) 저장해서 나중에 정답 비교에 씀
        for g in range(start, end):
            p = g // dataset_len
            d = g % dataset_len
            templates.append(_format_one(prompts[p], inputs[d], side))
            
            pd_pairs.append((p, d))
        # print(templates)
        # exit()
        # 이전에 남아있을 수 있는 logits 비우기
        _ = params4logprobs_logits.flush()

        # vLLM generate (max_tokens=1 세팅이라 1토큰 logit만 찍힘)
        _ = model.generate(templates, params4logprobs, use_tqdm=False)

        # 이번 chunk에서 나온 logits만 가져오기
        logits_list = params4logprobs_logits.flush()
        # logits_list 길이 == len(templates)
        # logits shape: (vocab_size,)  (대부분 GPU tensor)

        # ⭐ 전체 vocab을 stack하지 말고, verbalizer 3개만 뽑아서 바로 softmax
        # (B, V)
        v_logits = torch.stack([lg[verbalizer_ids] for lg in logits_list], dim=0)  # GPU에 잠깐
        probs = torch.softmax(v_logits, dim=1)  # (B, V)
        pred = torch.argmax(probs, dim=1).cpu()  # (B,)

        # pred와 targets 비교해서 prompt별로 누적
        for i, (p, d) in enumerate(pd_pairs):
            if int(pred[i]) == int(targets_expanded[p, d]):
                correct[p] += 1

        # GPU 텐서 빨리 해제 유도
        del logits_list, v_logits, probs, pred

    acc_per_prompt = correct.float() / float(dataset_len)
    return acc_per_prompt


@torch.inference_mode(True)
def evaluate_prompts(prompts,
                     dataset,
                     model,
                     tokenizer,
                     params4logprobs_logits,
                     params4logprobs,
                     verbalizer=['Yes', 'No', 'Maybe'],
                     side='First',
                     soft_diff=False,
                     ):
    def _format_prompts(prompts, inputs,side):
        if side == 'First':
            template = "{prompt} Input : {sentence_1} Output:"
        else:
            template = "{sentence_1} {prompt}"
        return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]
    
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
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
        formatted_templates = _format_prompts(current_prompts, inputs, side=side)
        all_templates.extend(formatted_templates)
    # print(len(all_templates))
    
    with torch.no_grad():
        torch.cuda.synchronize()
        logits = params4logprobs_logits.flush()
        outputs = model.generate(
            all_templates, 
            params4logprobs,
            use_tqdm=True
        )
        logits = params4logprobs_logits.flush()  # list of (vocab_size,) tensors
        all_logits = torch.stack(logits, dim=0)  # (batch_size, vocab_size)
    # print(all_logits.shape)

    verbalizer_logits = all_logits[..., verbalizer_ids]
    log_probs = F.softmax(verbalizer_logits, dim=1).reshape(len(prompts), dataset_len, -1)
    preds = torch.argmax(log_probs, dim=-1).cpu()
    targets_expanded = targets.unsqueeze(0).expand(len(prompts), -1)
    correct_per_prompt = torch.sum(preds == targets_expanded, dim=1)
    acc_per_prompt = correct_per_prompt / dataset_len
    # print(acc_per_prompt)
    return acc_per_prompt
    
    # dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
    # accuracys = []
    # wrongs = []
    # sds = []
    # for prompt in prompts:
    #     total = 0
    #     correct = 0
    #     sd = 0
    #     with torch.no_grad():
    #         for batch in tqdm(dataloader):
    #             if 'text' in batch.keys():
    #                 inputs = batch['text']
    #             else:
    #                 inputs = batch['sentence']
    #             targets = batch['label']
    #             softmax_diff,acc = evaluation_soft(
    #                 [prompt],
    #                 inputs,
    #                 targets,
    #                 model,
    #                 tokenizer,
    #                 device,
    #                 params4logprobs_logits,
    #                 params4logprobs,
    #                 verbalizer,
    #                 side=side,
    #             )
    #             batch_size = len(targets)
    #             correct += acc[0] * batch_size
    #             total += batch_size
    #             sd += softmax_diff[0]
    #     accuracy = correct / total
    #     soft_diff = sd / total
    #     accuracys.append(torch.Tensor([accuracy]))
    #     sds.append(torch.Tensor([soft_diff]))
    # return accuracys



# def evaluation_soft(prompts,
#                     inputs,
#                     targets,
#                     model,
#                     tokenizer,
#                     device,
#                     params4logprobs_logits,
#                     params4logprobs,
#                     verbalizer, 
#                     Fail_coefficient=1,
#                     Success_coefficient=1,
#                     return_reward= False,
#                     side = 'First',
#                     ):
#     def _format_prompts(prompts, inputs,side):
#         if side == 'First':
#             template = "{prompt} Input : {sentence_1} Output:"
#         else:
#             template = "{sentence_1} {prompt}"
#         return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]
    
#     # def _get_next_token_index(input_ids):
#     #     # 입력의 마지막 토큰 다음 위치 반환
#     #     return input_ids.shape[1] - 1

#     # def _get_logits(texts, tokenizer, model, device):
#     #     batch_size = len(texts)
#     #     encoded_inputs = tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
#     #     token_logits = model(**encoded_inputs.to(device)).logits
#     #     next_token_indices = _get_next_token_index(encoded_inputs['input_ids'])
#     #     out_logits = token_logits[range(batch_size), next_token_indices, :]
#     #     return out_logits

#     accuracies = []
#     rewards = []
#     # model.eval()
#     verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
#     # print(verbalizer_ids)
#     batch_size = targets.size(0)
#     for prompt in prompts:
#         #Get logits
#         current_prompts = [prompt for _ in range(batch_size)]
#         formatted_templates = _format_prompts(current_prompts, inputs, side=side)
        
#         torch.cuda.synchronize()
#         logits = params4logprobs_logits.flush()
#         outputs = model.generate(
#             formatted_templates, 
#             params4logprobs,
#             use_tqdm=False
#         )
#         logits = params4logprobs_logits.flush()  # list of (vocab_size,) tensors
#         all_logits = torch.stack(logits, dim=0)  # (batch_size, vocab_size)
#         # print(all_logits.shape)
#         # for logit in logits:
#         #     print(logit.shape)
#         # exit()
        
#         # all_logits = _get_logits(formatted_templates, tokenizer, model, device)
        
#         #Get verbalizer logits
#         verbalizer_logits = all_logits[:, verbalizer_ids]
#         log_probs = F.softmax(verbalizer_logits, dim=1)
#         #print(log_probs)
#         #Get accuracy
#         preds = torch.argmax(log_probs, dim=1).cpu()
#         correct_predictions = torch.sum(preds == targets)
#         accuracy = correct_predictions.item() / batch_size
#         accuracies.append(accuracy)
        
#         #Get reward
#         reward = get_reward(all_logits, targets, Fail_coefficient=Fail_coefficient, Success_coefficient=Success_coefficient)
#         mean_reward = reward.mean().cpu()
#         rewards.append(mean_reward)

#     z_scaled_reward = rewards

#     if return_reward:
#         return z_scaled_reward, accuracies, rewards
#     else:
#         return z_scaled_reward, accuracies


# def get_reward(
#     logits,
#     labels,
#     Fail_coefficient=1,
#     Success_coefficient=1,
# ):
#     #TODO
#     # Inputs :
#     #     logits : 모델의 출력 logits
#     #     targets : 정답 레이블
#     # Outputs :
#     #     reward : 정답 로짓 - 최대 로짓의 값 * 성공/실패 계수
#     with torch.no_grad():
#         labels = labels.to('cpu')
#         logits = logits.to('cpu')
#         correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

#         # 정답 레이블의 로짓 값을 0으로 만들어 최대값 계산에 영향을 주지 않도록 합니다.
#         mask = torch.ones_like(logits)
#         mask.scatter_(1, labels.unsqueeze(1), 0)
#         masked_logits = logits * mask

#         # 마스킹된 로짓 중 최대값을 찾습니다.
#         max_other_logits = masked_logits.max(dim=1)[0]
#         differences = correct_logits - max_other_logits
        
#     reward = torch.where(differences > 0, differences * Success_coefficient, differences * Fail_coefficient)
#     return reward


@torch.inference_mode()
def evaluate_prompts_chunked_II(
    prompts: List[str],
    dataset,
    model,  # vLLM LLM 객체
    tokenizer,  # HuggingFace tokenizer
    sampling_params,  # vLLM SamplingParams (텍스트 생성용)
    task: str,
    chunk_size: int = 256,
    return_details: bool = False,
):
    """
    Instruction Induction 태스크용 vLLM 평가 함수.
    
    TC와 달리 verbalizer 기반이 아닌, 텍스트 생성 후 metric(f1, em, es, contains)으로 평가.
    
    Args:
        prompts: 평가할 프롬프트 리스트
        dataset: 테스트 데이터셋 (text, label 포함)
        model: vLLM LLM 객체
        tokenizer: HuggingFace tokenizer
        sampling_params: vLLM SamplingParams (텍스트 생성용, max_tokens 등 설정)
        task: 태스크 이름 (메트릭 결정용)
        chunk_size: 한번에 처리할 요청 개수 (메모리 관리용)
        return_details: 상세 결과 반환 여부
        
    Returns:
        acc_per_prompt: 프롬프트별 accuracy (torch.Tensor, shape: (num_prompts,))
        details: (return_details=True일 때) 상세 결과 리스트
    """
    # 전체 데이터셋을 한번에 로드
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
    batch = next(iter(dataloader))
    inputs = batch['text'] if 'text' in batch else batch['sentence']
    labels = batch['label']
    dataset_len = len(inputs)
    num_prompts = len(prompts)
    
    # 메트릭 결정
    metric = TASK_TO_METRIC.get(task, default_metric)
    
    # 전체 요청: (prompt_idx, data_idx) 쌍
    # global_idx = p * dataset_len + d
    total = num_prompts * dataset_len
    
    # 프롬프트별 점수 누적
    scores_per_prompt = torch.zeros(num_prompts, dtype=torch.float32)
    details = [] if return_details else None
    
    # 청크 단위로 처리
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        
        # 이번 청크에 해당하는 templates 생성
        templates = []
        pd_pairs = []  # (prompt_idx, data_idx) 저장
        
        for g in range(start, end):
            p = g // dataset_len
            d = g % dataset_len
            template = f"{prompts[p]}\nInput : {inputs[d]}\nOutput : "
            templates.append(template)
            pd_pairs.append((p, d))
        
        # vLLM generate
        outputs = model.generate(templates, sampling_params, use_tqdm=False)
        
        # 결과 추출 및 점수 계산
        for i, output in enumerate(outputs):
            p, d = pd_pairs[i]
            generated_text = output.outputs[0].text
            ground_truth = labels[d]
            
            # 메트릭별 점수 계산
            if metric == 'f1':
                score = get_f1_score(generated_text, ground_truth)
            elif metric == 'em':
                score = get_em_score(generated_text, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(generated_text, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(generated_text, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')
            
            scores_per_prompt[p] += score
            
            if return_details:
                details.append({
                    'prompt_idx': p,
                    'data_idx': d,
                    'input': inputs[d],
                    'prediction': generated_text,
                    'ground_truth': ground_truth,
                    'score': score
                })
    
    # 평균 accuracy 계산
    acc_per_prompt = scores_per_prompt / float(dataset_len)
    
    if return_details:
        return acc_per_prompt, details
    return acc_per_prompt