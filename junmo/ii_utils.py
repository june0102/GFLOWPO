import os
import json
import random
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
# from trl.core import LengthSampler  # Removed: not available in newer TRL versions
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader,TensorDataset
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
from PIL import Image
import os
from peft import LoraConfig
import warnings
import numpy as np
import wandb
import copy
from collections import deque
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import random
import torch
import heapq
import torch.nn.functional as F
from collections import Counter
import string
# from datasets import load_metric  # Deprecated, use evaluate library instead if needed
from typing import List
import re
induce_data_path = os.path.join(os.path.dirname(__file__), 'automatic_prompt_engineer/experiments/data/instruction_induction/raw/induce/')
eval_data_path = os.path.join(os.path.dirname(__file__), 'automatic_prompt_engineer/experiments/data/instruction_induction/raw/execute/')
annotation_data_path = os.path.join(os.path.dirname(__file__), 'automatic_prompt_engineer/experiments/data/instruction_induction/annotations/')

# Get a list of tasks (by looking at the names of the files in the induced directory)
tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]
TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains',
                    'dyck_languages': 'f1',
                    'gender_inclusive_sentences_german': 'f1',
                    'object_counting': 'f1',
                    'operators': 'f1',
                    'tense': 'f1',
                    'word_sorting': 'f1',
                    'word_unscrambling': 'f1',
                    'linguistics_puzzles': 'f1',}
default_metric = 'em'

def load_data(type, task):
    base_path = induce_data_path if type == 'induce' else eval_data_path
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        if task == 'cause_and_effect':
            cause, effect = data['cause'], data['effect']
            # Pick an order randomly
            if random.random() < 0.5:
                input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
            else:
                input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
            output_ = cause
        elif task == 'common_concept':
            items = data['items']
            # Make comma separated list of items
            input_ = ', '.join(items[:-1])
            output_ = data['all_common_concepts']
        elif task == 'rhymes':
            input_, output_ = data['input'], data['other_rhymes']
        elif 'translation' in task:
            input_, output_ = data['input'], data['possible_translations']
        else:
            input_, output_ = data['input'], [data['output']]
        if isinstance(output_, list):
            output_ = output_[0]
        if isinstance(input_, list):
            input_ = input_[0]
        inputs.append(input_)
        outputs.append(output_)
    return inputs, outputs

def load_annotation(task):
    path = annotation_data_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    return annotations[0]





def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction




def load_ii_data(task, seed=42):
    train_inputs, train_outputs = load_data('induce', task)
    test_inputs, test_outputs = load_data('execute', task)
    
    # Random shuffle train data to avoid biased splits
    random.seed(seed)
    indices = list(range(len(train_inputs)))
    random.shuffle(indices)
    train_inputs = [train_inputs[i] for i in indices]
    train_outputs = [train_outputs[i] for i in indices]
    
    if len(train_inputs) > 32:
        train_ds = Dataset.from_dict({'text': train_inputs[:32], 'label': train_outputs[:32]})
        validation_ds = Dataset.from_dict({'text': train_inputs[32:64], 'label': train_outputs[32:64]})
    else:
        train_ds = Dataset.from_dict({'text': train_inputs, 'label': train_outputs})
        validation_ds = train_ds
    test_ds = Dataset.from_dict({'text': test_inputs, 'label': test_outputs})
    return train_ds, test_ds, validation_ds


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1
    else:
        return 0

def _format_prompts(prompts,inputs):
    return [prompt + '\n' + 'Input : ' + inputss + '\nOutput : ' for prompt,inputss in zip(prompts,inputs)]

def _format_prompt(prompt,inputs):
    template = "{prompt} Input : {sentence_1} Output : "
    return [template.format(prompt=prompt,sentence_1=inputss) for inputss in inputs]

def _format_prompt_tta(prompt,inputs):
    template = "{prompt} Current Input : {sentence_1} \n \nRewritten Instruction : "
    return [template.format(prompt=prompt,sentence_1=inputss) for inputss in inputs]

def _get_only_generated(outputs,trigger):
    #print(outputs,trigger)
    new_output = []
    for output in outputs:
        if trigger in output:
            new_output.append(output.split(trigger)[1].strip())
        else:
            print(output)
            new_output.append(output)
    #print(outputs)
    return new_output

def _get_generated_text(inputs,outputs):
    new_outputs = []
    for i in range(len(inputs)):
        current_input = inputs[i]
        current_output = outputs[i]
        length = len(current_input)
        if len(current_output) > length:
            new_outputs.append(current_output[length:])
        else:
            new_outputs.append(current_output)
    return new_outputs


def ii_tta_evaluation(
    dataset,
    agent_model,
    agent_tokenizer,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    prompt_generation_kwargs,
    task,
    batch_size= 8,
    
):
    total = 0
    scores = 0
    check_manual = False
    if 'Instruction : ' in meta_prompt:
        check_manual = True
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    for batch in tqdm(dataloader):
        inputs = batch['text']
        labels = batch['label']
        meta_prompted_inputs = _format_prompt_tta(meta_prompt,inputs)
        input_ids = agent_tokenizer(meta_prompted_inputs,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=512,).input_ids.to(device)
        with torch.no_grad():
            outputs= agent_model.generate(input_ids,
                                          **prompt_generation_kwargs)
        generated_texts = target_tokenizer.batch_decode(outputs,
                                                         skip_special_tokens=True)
        generated_texts = _get_generated_text(meta_prompted_inputs,generated_texts)
        prompted_inputs = _format_prompts(generated_texts,inputs)
        prompted_input_ids = target_tokenizer(prompted_inputs,
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        generated_texts_ = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = _get_generated_text(prompted_inputs,generated_texts_)
        metric = TASK_TO_METRIC.get(task, default_metric)
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')     
            scores += score
            total += 1
        print('Generated Text : ',generated_texts_[-1])
        print('-------------------')
        print('Prediction : ',prediction)
        print('-------------------')
        print('Ground Truth : ',ground_truth)
        print('-------------------')
        print('Score : ',score)
        print('-------------------')
    return scores / total
        
    
def ii_tta_evaluation_test(
    dataset,
    agent_model,
    agent_tokenizer,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    prompt_generation_kwargs,
    task,
    batch_size= 8,
    
):
    total = 0
    scores = 0
    check_manual = False
    if 'Instruction : ' in meta_prompt:
        check_manual = True
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    for batch in tqdm(dataloader):
        inputs = batch['text']
        labels = batch['label']
        prompted_inputs = _format_prompts([meta_prompt for i in range(len(inputs))],inputs)
        #print(prompted_inputs)
        prompted_input_ids = target_tokenizer(prompted_inputs,
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        generated_texts_ = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = _get_generated_text(prompted_inputs,generated_texts_)
        #print(generated_texts)
        #print(generated_texts_)
        metric = TASK_TO_METRIC.get(task, default_metric)
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')     
            scores += score
            total += 1
        print('Generated Text : ',generated_texts_[-1])
        print('-------------------')
        print('Prediction : ',prediction)
        print('-------------------')
        print('Ground Truth : ',ground_truth)
        print('-------------------')
        print('Score : ',score)
        print('-------------------')
    return scores / total
        
    
    
    
    
    
def evaluation_ii_batch(
    prompt,
    dataset,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    task,
    batch_size= 8,
    return_details=False,
    
):
    total = 0
    scores = 0
    details = []  # List to store details for each sample
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    for batch in tqdm(dataloader):
        inputs = batch['text']
        labels = batch['label']
        prompts = [prompt] * len(inputs)
        prompted_inputs = _format_prompts(prompts,inputs)
        prompted_input_ids = target_tokenizer(prompted_inputs,
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        generated_texts_ = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = _get_only_generated(generated_texts_, 'Output : ')
        metric = TASK_TO_METRIC.get(task, default_metric)
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')     
            scores += score
            total += 1
            
            # Store details for this sample
            if return_details:
                details.append({
                    'input': inputs[i],
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'score': score
                })
        print('\nPrompt : \n',prompt)
        print('-------------------')        
        print('\nGiven Input : \n',inputs[-1])
        print('-------------------')          
        print('\nTemplate : \n',prompted_inputs[-1])
        print('-------------------')   
        print('\nGenerated Text : \n',generated_texts_[-1])
        print('-------------------')
        print('\nPrediction : \n',prediction)
        print('-------------------')        
        print('\nGround Truth : \n',ground_truth)
        print('-------------------')        
        print('\nScore : \n',score)
        print('-------------------')  
            

    accuracy = scores / total if total > 0 else 0.0
    if return_details:
        return accuracy, details
    return accuracy

def lora_to_base(model):
    """Disable LoRA adapter layers to use base model"""
    try:
        # For AutoModelForCausalLMWithValueHead, LoRA is in pretrained_model
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'disable_adapter_layers'):
            model.pretrained_model.disable_adapter_layers()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'disable_adapter_layers'):
            model.base_model.disable_adapter_layers()
        else:
            print("No adapter layers to disable")
    except Exception as e:
        print(f"Could not disable adapter layers: {e}")
    model.eval()

def base_to_lora(model):
    """Enable LoRA adapter layers"""
    try:
        # For AutoModelForCausalLMWithValueHead, LoRA is in pretrained_model
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'enable_adapter_layers'):
            model.pretrained_model.enable_adapter_layers()
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'enable_adapter_layers'):
            model.base_model.enable_adapter_layers()
        else:
            print("No adapter layers to enable")
    except Exception as e:
        print(f"Could not enable adapter layers: {e}")
    model.train()

def evaluation_ii_batch_with_log_reward(
    prompt,
    dataset,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    task,
    agent_model,
    agent_tokenizer,
    initial_ref_model_state_dict,
    step,
    beta=1.0,
    gamma=1.0,
    reward_sched_start=1.0,
    reward_sched_end=1.0,
    reward_sched_horizon=1000,
    lm_sched_start=1.0,
    lm_sched_end=1.0,
    lm_sched_horizon=1000,
    batch_size=8,
    return_details=False,
    query_encoded=None,
    response_tensor=None,
):
    """
    Calculate log_reward = (lm_log_reward / gamma) + (c_log_reward / beta)
    
    Args:
        prompt: Generated prompt text
        dataset: Dataset for evaluation
        target_model: Target model for task evaluation
        target_tokenizer: Target tokenizer
        device: Device
        meta_prompt: Meta prompt (not used but kept for compatibility)
        generation_kwargs: Generation kwargs
        task: Task name
        agent_model: Agent model (for LM log reward calculation)
        agent_tokenizer: Agent tokenizer
        initial_ref_model_state_dict: Initial reference model state_dict
        step: Current training step
        beta: C reward temperature (fixed)
        gamma: LM reward temperature (scheduled)
        reward_sched_start: Reward temperature schedule start
        reward_sched_end: Reward temperature schedule end
        reward_sched_horizon: Reward temperature schedule horizon
        lm_sched_start: LM reward temperature schedule start
        lm_sched_end: LM reward temperature schedule end
        lm_sched_horizon: LM reward temperature schedule horizon
        batch_size: Batch size
        return_details: Whether to return details
        
    Returns:
        log_reward: Final log reward
        details: Details if return_details=True
    """
    # Calculate scheduled temperatures
    def get_lm_reward_temp(step):
        diff = lm_sched_end - lm_sched_start
        temp = lm_sched_start + diff * min(1, step / lm_sched_horizon)
        return temp
    
    def get_total_reward_temp(step):
        diff = reward_sched_end - reward_sched_start
        temp = reward_sched_start + diff * min(1, step / reward_sched_horizon)
        return temp
    
    gamma_scheduled = get_lm_reward_temp(step)
    rew_temp = get_total_reward_temp(step)
    
    # ========== 1. Calculate LM Log Reward ==========
    # Use base model (LoRA disabled) for LM log reward calculation
    # Note: We just disable LoRA instead of loading initial state_dict
    #       because the state_dict keys don't match between agent_model and ref_model
    lora_to_base(agent_model)
    with torch.no_grad():
        try:
            if query_encoded is not None and response_tensor is not None:
                # Use provided query and response tensors
                query_prompt_len = query_encoded.size(1)
                # Concatenate query and response
                prompts_responses = torch.cat([query_encoded, response_tensor], dim=1)
                
                # Create attention mask
                attention_mask = torch.ones_like(prompts_responses)
                
                outputs = agent_model(
                    input_ids=prompts_responses,
                    attention_mask=attention_mask,
                )
                
                # Handle both AutoModelForCausalLM and AutoModelForCausalLMWithValueHead
                if hasattr(outputs, 'logits'):
                    model_logits = outputs.logits
                else:
                    model_logits = outputs[0]  # Some models return tuple
                
                # Extract logits for response tokens
                model_logits = model_logits[:, query_prompt_len-1:-1]
                response_labels = prompts_responses[:, query_prompt_len:]
                
                # Calculate pad mask (EOS token is pad_token_id)
                pad_mask = (response_labels == agent_tokenizer.pad_token_id).cumsum(1) > 1
                
                log_prob = F.log_softmax(model_logits, dim=-1)
                lm_logreward = torch.gather(
                    log_prob, -1, response_labels.unsqueeze(2)).squeeze(2)
                lm_logreward = torch.where(pad_mask, 0.0, lm_logreward)
                lm_logreward = torch.sum(lm_logreward, 1)
            else:
                # Fallback: reconstruct from prompt text
                # This is less accurate but works if tensors are not provided
                query_text = [
                    {"role": "user", "content": meta_prompt},
                    {"role": "assistant", "content": "The Instruction is : " + prompt}
                ]
                
                full_encoded = agent_tokenizer.apply_chat_template(
                    query_text,
                    return_tensors='pt',
                    add_generation_prompt=False
                ).view(1, -1).to(device)
                
                # Get prompt length (user message part)
                prompt_only = agent_tokenizer.apply_chat_template(
                    [{"role": "user", "content": meta_prompt}],
                    return_tensors='pt',
                    add_generation_prompt=False
                ).view(1, -1)
                
                query_prompt_len = prompt_only.size(1)
                
                outputs = agent_model(
                    input_ids=full_encoded,
                    attention_mask=torch.ones_like(full_encoded),
                )
                
                # Handle both AutoModelForCausalLM and AutoModelForCausalLMWithValueHead
                if hasattr(outputs, 'logits'):
                    model_logits = outputs.logits
                else:
                    model_logits = outputs[0]
                
                if full_encoded.size(1) > query_prompt_len:
                    model_logits = model_logits[:, query_prompt_len-1:-1]
                    response_labels = full_encoded[:, query_prompt_len:]
                    
                    pad_mask = (response_labels == agent_tokenizer.pad_token_id).cumsum(1) > 1
                    
                    log_prob = F.log_softmax(model_logits, dim=-1)
                    lm_logreward = torch.gather(
                        log_prob, -1, response_labels.unsqueeze(2)).squeeze(2)
                    lm_logreward = torch.where(pad_mask, 0.0, lm_logreward)
                    lm_logreward = torch.sum(lm_logreward, 1)
                else:
                    lm_logreward = torch.tensor(0.0).to(device)
        except Exception as e:
            print(f"Warning: Error calculating LM log reward: {e}")
            lm_logreward = torch.tensor(0.0).to(device)
    
    base_to_lora(agent_model)
    
    # ========== 2. Calculate C Log Reward (Log Probability-based) ==========
    total = 0
    scores = 0
    details = []
    sum_log_probs = 0.0  # Sum of log probabilities for c_log_reward
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in tqdm(dataloader, desc="Calculating C log reward"):
        inputs = batch['text']
        labels = batch['label']
        prompts_list = [prompt] * len(inputs)
        prompted_inputs = _format_prompts(prompts_list, inputs)
        
        # ===== Log Probability Calculation =====
        # Create prompt + input + ground_truth sequences for log prob calculation
        prompted_inputs_with_labels = [
            pi + labels[i] for i, pi in enumerate(prompted_inputs)
        ]
        
        # Tokenize the full sequences (prompt + input + label)
        full_encoded = target_tokenizer(
            prompted_inputs_with_labels,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Tokenize just the prompt + input part to get the length
        prompt_input_encoded = target_tokenizer(
            prompted_inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        prompt_input_len = prompt_input_encoded.input_ids.size(1)
        
        with torch.no_grad():
            # Get logits from target model
            outputs = target_model(
                input_ids=full_encoded.input_ids,
                attention_mask=full_encoded.attention_mask,
            )
            logits = outputs.logits
            
            # Extract logits for the label tokens (after prompt+input)
            # logits[:, prompt_input_len-1:-1] corresponds to predictions for tokens at positions prompt_input_len onwards
            label_logits = logits[:, prompt_input_len-1:-1, :]
            label_tokens = full_encoded.input_ids[:, prompt_input_len:]
            
            # Create mask for valid label tokens (not padding)
            label_attention_mask = full_encoded.attention_mask[:, prompt_input_len:]
            
            # Calculate log probabilities
            log_probs = F.log_softmax(label_logits, dim=-1)
            
            # Gather log probs for the actual label tokens
            token_log_probs = torch.gather(
                log_probs, -1, label_tokens.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask out padding tokens
            token_log_probs = token_log_probs * label_attention_mask.float()
            
            # Sum log probs for each sample
            sample_log_probs = token_log_probs.sum(dim=1)
            
            # Add to total sum
            sum_log_probs += sample_log_probs.sum().item()
        
        # ===== Accuracy Calculation (for logging/display) =====
        prompted_input_ids = prompt_input_encoded.input_ids
        
        with torch.no_grad():
            gen_outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        
        generated_texts_ = target_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
        generated_texts = _get_only_generated(generated_texts_, 'Output : ')
        metric = TASK_TO_METRIC.get(task, default_metric)
        
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')
            
            scores += score
            total += 1
            
            if return_details:
                details.append({
                    'input': inputs[i],
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'score': score,
                    'log_prob': sample_log_probs[i].item() if i < len(sample_log_probs) else 0.0
                })
    
    accuracy = scores / total if total > 0 else 0.0
    # C Log Reward: Sum of log probabilities (log probability-based)
    c_log_reward = torch.tensor(sum_log_probs, dtype=torch.float32, device=device)
    
    # ========== 3. Calculate Final Log Reward ==========
    # log_reward = (lm_log_reward / gamma) + (c_log_reward / beta)
    # Ensure lm_logreward is on the same device as c_log_reward
    if lm_logreward.device != c_log_reward.device:
        lm_logreward = lm_logreward.to(c_log_reward.device)
    log_reward = (lm_logreward / gamma_scheduled) + (c_log_reward / beta)
    
    # Apply final temperature
    tempered_log_reward = log_reward / rew_temp
    
    # Handle tensor shape: if multi-dimensional, take mean
    if tempered_log_reward.dim() > 0:
        tempered_log_reward = tempered_log_reward.mean()
    
    # Return log_reward, accuracy, c_log_reward, lm_log_reward
    # log_reward is used for PPO training, accuracy is used for logging/display
    # c_log_reward and lm_log_reward are for wandb logging
    lm_log_reward_value = lm_logreward.mean().item() if lm_logreward.dim() > 0 else lm_logreward.item()
    c_log_reward_value = c_log_reward.item()
    
    if return_details:
        return tempered_log_reward.item(), accuracy, details, c_log_reward_value, lm_log_reward_value
    return tempered_log_reward.item(), accuracy, c_log_reward_value, lm_log_reward_value

def evaluation_ii(
    prompts,
    dataset,
    model,
    tokenizer,
    device,
    task,
    generation_kwargs=None,
    show=False,
    must_show=False,
):
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
    rewardss = []
    accs = []
    if generation_kwargs is None:
        generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 30,
        "min_length": -1,
        }
    with torch.no_grad():
        for prompt in prompts:
            loss = 0
            acc = 0
            total = 0
            scores = 0
            reward =0
            for batch in dataloader:
                inputs = batch['text']
                labels = batch['label']
                template = prompt + '\nInput : ' + inputs[0] + '\nOutput : '
                prompt_encoded = tokenizer(template, return_tensors='pt').to(device)
                label_encoded = tokenizer(labels[0], return_tensors='pt')
                length_label = label_encoded['input_ids']
                #print(' LNEGTH  : ',len(length_label[0]))
                #print(length_label.size())
                outputs = model.generate(**prompt_encoded,**generation_kwargs)
                prediction_ = tokenizer.decode(outputs[0],skip_special_tokens=True)
                prediction = _get_generated_text([template], [prediction_])[0]
                ground_truth = labels[0]
                rewards = get_f1_score(prediction,ground_truth)
                metric = TASK_TO_METRIC.get(task, default_metric)
                if metric == 'f1':
                    score = get_f1_score(prediction, ground_truth)
                elif metric == 'em':
                    score = get_em_score(prediction, ground_truth)
                elif metric == 'es':
                    score = get_exact_set_score(prediction, ground_truth)
                elif metric == 'contains':
                    score = get_contains_score(prediction, ground_truth)
                else:
                    raise ValueError(f'Invalid metric {metric}')
                             
                reward += rewards
                scores += score
                total += 1
            acc = scores / total
            accs.append(acc)
            rewardss.append(reward)
            if must_show==True:
                print('\nPrompt : \n',prompt)
                print('-------------------')        
                print('\nGiven Input : \n',inputs[0])
                print('-------------------')        
                print('\nTemplate : \n' ,template)
                print('-------------------')        
                print('\nGenerated Text : \n',prediction_)
                print('-------------------')
                print('\nPrediction : \n',prediction)
                print('-------------------')        
                print('\nGround Truth : \n',ground_truth)
                print('-------------------')        
                print('\nScore : \n',score)
                print('-------------------')        
                print('\nReward : \n',rewards)
                print('-------------------')   
    return rewardss, accs

def got_example_ii(dataset,shot=5):
    examples = ''
    for i in range(shot):
        idx = random.randint(0,len(dataset)-1)
        example = dataset[idx]
        #print('Input : ',example['text'])
        #print('Output : ',example['label'])
        a = 'Input : ' + example['text'] + '\nOutput : ' + example['label']
        examples += a + '\n'
    return examples


# ==============================================================================
# Common Log-Likelihood Functions for GFN (II/TC)
# ==============================================================================

def compute_log_likelihood_ii(
    prompts: List[str],
    dataset,
    target_model,
    target_tokenizer,
    device: str,
    batch_size: int = 16,
    pair_batch_size: int = None,
    enable_length_bucketing: bool = True,
    subsample_size: int = None,
    subsample_seed: int = None,
    subsample_with_replacement: bool = False,
    scale_log_likelihood: bool = True,
) -> tuple:
    """
    Compute token-level log-likelihood for Instruction Induction (II) tasks.
    
    REFACTORED VERSION:
    - Removes prompt-for-loop by flattening (prompt, sample) pairs
    - Single batched forward pass instead of num_prompts × num_batches forwards
    - Per-sample prefix_lens calculation (fixes padding bug)
    - Truncation detection with warnings
    - Optional length bucketing for reduced padding waste
    - Optional dataset subsampling for faster approximate evaluation
    
    Full log-likelihood:
        L = Σ_{j=1..N} log p(y_j | prompt, x_j)
    
    With subsampling (m samples from N):
        L_hat = (N/m) * Σ_{j∈S} log p(y_j | prompt, x_j), |S|=m
    
    This provides an unbiased estimate of the full log-likelihood.
    Accuracy is computed on the subset only (no scaling).
    
    Args:
        prompts: List of prompt candidates θ′
        dataset: Dataset with 'text' (inputs) and 'label' (outputs)
        target_model: Target LLM for evaluation
        target_tokenizer: Target tokenizer
        device: Device string
        batch_size: (DEPRECATED) Legacy parameter, use pair_batch_size instead.
                    If pair_batch_size is None, this value is used.
        pair_batch_size: Batch size for flattened (prompt, sample) pairs.
                         Recommended: 128-256 for optimal GPU utilization.
                         Total pairs = num_prompts × num_samples.
        enable_length_bucketing: If True, sort pairs by estimated length before
                                 batching to reduce padding waste. This only
                                 changes processing order, NOT the results.
        subsample_size: If None, use all N samples. If int, sample m=min(subsample_size, N)
                        samples per call. All prompts use the SAME sampled indices
                        for fair comparison and variance reduction.
        subsample_seed: If None, random sampling (non-reproducible).
                        If int, use this seed for reproducible sampling.
        subsample_with_replacement: If False, sample without replacement (default).
                                    If True, sample with replacement.
        scale_log_likelihood: If True (default), multiply by (N/m) to get unbiased estimate.
                              If False, return raw subset sum (for debugging).
        
    Returns:
        log_likelihoods: (num_prompts,) log likelihood for each prompt
                         If subsampling with scale_log_likelihood=True: L_hat = (N/m) * Σ_{j∈S} log p(...)
        accuracies: (num_prompts,) accuracy for each prompt (subset-based, no scaling)
    
    Note:
        Unlike the legacy implementation where batch_size meant "samples per batch",
        here batch_size (or pair_batch_size) means "(prompt, sample) pairs per batch".
        For 100 prompts × 32 samples = 3200 pairs:
        - pair_batch_size=16  → 200 forward calls (no speedup)
        - pair_batch_size=256 → 13 forward calls (15x fewer)
        
        Length bucketing only changes the order of processing, not the final
        aggregated results. This is because log_likelihoods are accumulated
        via scatter_add which is order-independent.
    """
    import random
    
    num_prompts = len(prompts)
    N = len(dataset)  # Total dataset size
    
    # =========================================================================
    # Subsampling: select m indices from N
    # =========================================================================
    if subsample_size is not None:
        m = min(subsample_size, N)
        
        # Create RNG with optional seed for reproducibility
        if subsample_seed is not None:
            rng = random.Random(subsample_seed)
        else:
            rng = random.Random()
        
        # Sample indices
        if subsample_with_replacement:
            sampled_indices = rng.choices(range(N), k=m)
        else:
            sampled_indices = rng.sample(range(N), m)
    else:
        m = N
        sampled_indices = list(range(N))
    
    num_samples = m  # Actual number of samples used
    total_pairs = num_prompts * num_samples
    
    # Resolve batch size: prefer pair_batch_size if provided
    effective_batch_size = pair_batch_size if pair_batch_size is not None else batch_size
    
    # Performance warning for small batch sizes
    if effective_batch_size < 64 and total_pairs > 100:
        print(f"[INFO] compute_log_likelihood_ii: batch_size={effective_batch_size} is small "
              f"for {total_pairs} pairs. Consider pair_batch_size>=128 for speedup.")
    
    # Initialize accumulators on GPU
    log_likelihoods = torch.zeros(num_prompts, device=device)
    correct_counts = torch.zeros(num_prompts, device=device)
    sample_counts = torch.zeros(num_prompts, device=device)
    
    # Truncation statistics
    truncation_count = 0
    max_length = 512
    
    # Get pad token id
    pad_token_id = target_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = target_tokenizer.eos_token_id
    
    # =========================================================================
    # B1) Flatten (prompt, sample) pairs into a dataset
    # =========================================================================
    # Note: All prompts use the SAME sampled_indices for fair comparison
    flattened_pairs = []
    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in sampled_indices:  # Use sampled indices instead of range(N)
            sample = dataset[sample_idx]
            input_text = sample['text']
            label_text = sample['label']
            # Estimate sequence length for length bucketing (character-based)
            est_len = len(prompt) + len(input_text) + len(label_text)
            flattened_pairs.append({
                'prompt_idx': prompt_idx,
                'sample_idx': sample_idx,
                'prompt_str': prompt,
                'input_text': input_text,
                'label_text': label_text,
                'est_len': est_len,
            })
    
    # =========================================================================
    # B1.5) Length Bucketing: Sort by estimated length to reduce padding waste
    # =========================================================================
    # Sorting by est_len groups similar-length sequences together,
    # reducing padding within each batch. This does NOT affect results
    # because scatter_add aggregation is order-independent.
    if enable_length_bucketing:
        flattened_pairs = sorted(flattened_pairs, key=lambda x: x['est_len'])
    
    # Custom collate function to keep data as lists
    def collate_fn(batch):
        return {
            'prompt_idx': [item['prompt_idx'] for item in batch],
            'sample_idx': [item['sample_idx'] for item in batch],
            'prompt_str': [item['prompt_str'] for item in batch],
            'input_text': [item['input_text'] for item in batch],
            'label_text': [item['label_text'] for item in batch],
        }
    
    # Create DataLoader for flattened pairs
    pair_dataloader = DataLoader(
        flattened_pairs, 
        batch_size=effective_batch_size,  # Use effective_batch_size (was using wrong variable)
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Padding statistics tracking
    total_tokens_actual = 0
    total_tokens_padded = 0
    batch_idx = 0
    
    # =========================================================================
    # B2-B5) Batched teacher forcing log-likelihood calculation
    # =========================================================================
    target_model.eval()
    
    for batch in pair_dataloader:
        batch_prompt_idx = torch.tensor(batch['prompt_idx'], device=device, dtype=torch.long)
        prompts_batch = batch['prompt_str']
        inputs_batch = batch['input_text']
        labels_batch = batch['label_text']
        current_batch_size = len(prompts_batch)
        
        # -----------------------------------------------------------------
        # Step 1: Construct prefix and full sequences
        # -----------------------------------------------------------------
        # Format: f"{prompt}\nInput: {inp}\nOutput: " (unchanged)
        prompted_inputs = [
            f"{prompt}\nInput: {inp}\nOutput: " 
            for prompt, inp in zip(prompts_batch, inputs_batch)
        ]
        full_sequences = [
            pi + label 
            for pi, label in zip(prompted_inputs, labels_batch)
        ]
        
        # -----------------------------------------------------------------
        # Step 2: Tokenize with per-sample prefix_lens calculation
        # -----------------------------------------------------------------
        # Tokenize prefix (for length calculation)
        prefix_enc = target_tokenizer(
            prompted_inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        ).to(device)
        
        # Tokenize full sequence
        full_enc = target_tokenizer(
            full_sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        ).to(device)
        
        # Calculate per-sample prefix lengths (NOT padded size!)
        # prefix_lens[i] = number of non-pad tokens in prefix_enc for sample i
        prefix_lens = (prefix_enc.input_ids != pad_token_id).sum(dim=1)  # [B]
        
        # -----------------------------------------------------------------
        # Step 2.5: Truncation detection & Padding statistics
        # -----------------------------------------------------------------
        full_seq_lens = full_enc.attention_mask.sum(dim=1)  # [B]
        truncated_mask = (full_seq_lens >= max_length)
        batch_truncation_count = truncated_mask.sum().item()
        truncation_count += batch_truncation_count
        
        # Accumulate padding statistics
        batch_max_len = full_enc.input_ids.size(1)
        batch_actual_tokens = full_seq_lens.sum().item()
        batch_padded_tokens = current_batch_size * batch_max_len
        total_tokens_actual += batch_actual_tokens
        total_tokens_padded += batch_padded_tokens
        
        # Log padding stats for first batch only
        if batch_idx == 0:
            batch_mean_len = full_seq_lens.float().mean().item()
            batch_padding_ratio = 1 - (batch_actual_tokens / batch_padded_tokens)
            print(f"[PERF] Batch 0: max_seq={batch_max_len}, mean_seq={batch_mean_len:.1f}, "
                  f"padding_ratio={batch_padding_ratio:.1%}")
        
        batch_idx += 1
        
        # -----------------------------------------------------------------
        # Step 3: Forward pass (single batched forward)
        # -----------------------------------------------------------------
        with torch.no_grad():
            outputs = target_model(
                input_ids=full_enc.input_ids,
                attention_mask=full_enc.attention_mask,
            )
            logits = outputs.logits  # [B, T, V]
        
        # -----------------------------------------------------------------
        # Step 4: Compute next-token log probabilities
        # -----------------------------------------------------------------
        # logits[:, t, :] predicts token at position t+1
        # So we use logits[:, :-1, :] to predict targets[:, 1:]
        seq_len = full_enc.input_ids.size(1)
        
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
        targets = full_enc.input_ids[:, 1:]  # [B, T-1]
        
        # Gather log probabilities for target tokens
        token_log_probs = torch.gather(
            log_probs, -1, targets.unsqueeze(-1)
        ).squeeze(-1)  # [B, T-1]
        
        # -----------------------------------------------------------------
        # Step 5: Create mask for label tokens only
        # -----------------------------------------------------------------
        # positions in token_log_probs: 0, 1, ..., T-2
        # token_log_probs[i, t] = log p(full_ids[i, t+1] | full_ids[i, :t+1])
        # Label starts at full_ids[:, prefix_lens]
        # So we need positions where (t+1) >= prefix_lens, i.e., t >= prefix_lens - 1
        
        positions = torch.arange(seq_len - 1, device=device).unsqueeze(0)  # [1, T-1]
        prefix_lens_expanded = prefix_lens.unsqueeze(1)  # [B, 1]
        
        # Mask for label token predictions: position t >= prefix_lens - 1
        label_mask = (positions >= (prefix_lens_expanded - 1))  # [B, T-1]
        
        # Combine with attention mask to exclude padding
        nonpad_mask = full_enc.attention_mask[:, 1:].bool()  # [B, T-1]
        final_mask = label_mask & nonpad_mask  # [B, T-1]
        
        # -----------------------------------------------------------------
        # Step 6: Compute per-sample log-likelihood
        # -----------------------------------------------------------------
        masked_log_probs = token_log_probs * final_mask.float()
        sample_log_probs = masked_log_probs.sum(dim=1)  # [B]
        
        # -----------------------------------------------------------------
        # B3) Aggregate by prompt using scatter_add
        # -----------------------------------------------------------------
        log_likelihoods.scatter_add_(0, batch_prompt_idx, sample_log_probs)
        
        # -----------------------------------------------------------------
        # Accuracy calculation (string match, vectorized aggregation)
        # -----------------------------------------------------------------
        # For accuracy, we need to generate and compare
        # This is still sequential per batch but much faster than per-prompt loop
        
        # Get the prefix lengths for slicing generated output
        prefix_input_ids = prefix_enc.input_ids
        prefix_attention_mask = prefix_enc.attention_mask
        
        gen_outputs = target_model.generate(
            prefix_input_ids,
            attention_mask=prefix_attention_mask,
            max_new_tokens=30,
            pad_token_id=target_tokenizer.eos_token_id,
            do_sample=False,
        )
        
        # Decode generated text (excluding prefix)
        # Use per-sample prefix_lens for correct slicing
        generated_texts = []
        for i in range(current_batch_size):
            plen = prefix_lens[i].item()
            gen_ids = gen_outputs[i, plen:]
            gen_text = target_tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_texts.append(gen_text)
        
        # Compute correctness for each sample
        correct_tensor = torch.zeros(current_batch_size, device=device)
        for i, (gen, label) in enumerate(zip(generated_texts, labels_batch)):
            if label.strip().lower() in gen.strip().lower():
                correct_tensor[i] = 1.0
        
        # Aggregate correct counts by prompt
        correct_counts.scatter_add_(0, batch_prompt_idx, correct_tensor)
        sample_counts.scatter_add_(0, batch_prompt_idx, torch.ones(current_batch_size, device=device))
    
    # =========================================================================
    # Final accuracy computation
    # =========================================================================
    accuracies = correct_counts / sample_counts.clamp(min=1)
    
    # =========================================================================
    # Truncation warning
    # =========================================================================
    if truncation_count > 0:
        truncation_rate = truncation_count / total_pairs * 100
        print(f"[WARNING] compute_log_likelihood_ii: {truncation_count}/{total_pairs} "
              f"({truncation_rate:.1f}%) samples were truncated at max_length={max_length}")
    
    # =========================================================================
    # Overall padding statistics (with length bucketing)
    # =========================================================================
    if total_tokens_padded > 0:
        overall_padding_ratio = 1 - (total_tokens_actual / total_tokens_padded)
        bucketing_status = "ON" if enable_length_bucketing else "OFF"
        print(f"[PERF] Overall: {total_pairs} pairs, {batch_idx} batches, "
              f"padding_ratio={overall_padding_ratio:.1%} (bucketing={bucketing_status})")
    
    # =========================================================================
    # Subsampling scaling: L_hat = (N/m) * Σ_{j∈S} log p(y_j | ...)
    # =========================================================================
    if subsample_size is not None and m < N and scale_log_likelihood:
        scale_factor = N / m
        log_likelihoods = log_likelihoods * scale_factor
        # Note: accuracy is NOT scaled (it's a ratio on the subset)
    
    return log_likelihoods, accuracies


def compute_log_likelihood_ii_legacy(
    prompts: List[str],
    dataset,
    target_model,
    target_tokenizer,
    device: str,
    batch_size: int = 16,
) -> tuple:
    """
    LEGACY VERSION - kept for testing/comparison.
    Original implementation with prompt-for-loop structure.
    
    Known issues:
    - O(num_prompts × num_batches) forward calls
    - prefix_len uses padded batch size (correctness bug risk)
    """
    num_prompts = len(prompts)
    log_likelihoods = torch.zeros(num_prompts, device=device)
    accuracies = torch.zeros(num_prompts, device=device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for prompt_idx, prompt in enumerate(prompts):
        total_log_prob = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            inputs = batch['text']
            labels = batch['label']
            
            # Format: prompt + input + label
            prompted_inputs = [f"{prompt}\nInput: {inp}\nOutput: " for inp in inputs]
            full_sequences = [pi + label for pi, label in zip(prompted_inputs, labels)]
            
            # Tokenize prompt + input
            prompt_input_encoded = target_tokenizer(
                prompted_inputs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Tokenize full sequence (prompt + input + label)
            full_encoded = target_tokenizer(
                full_sequences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            prompt_input_len = prompt_input_encoded.input_ids.size(1)
            
            with torch.no_grad():
                outputs = target_model(
                    input_ids=full_encoded.input_ids,
                    attention_mask=full_encoded.attention_mask,
                )
                logits = outputs.logits
                
                # Extract logits for label tokens
                # logits[:, prompt_input_len-1:-1] corresponds to predictions for label tokens
                label_logits = logits[:, prompt_input_len-1:-1, :]
                label_tokens = full_encoded.input_ids[:, prompt_input_len:]
                label_mask = full_encoded.attention_mask[:, prompt_input_len:]
                
                # Compute log probabilities over FULL VOCABULARY
                log_probs = F.log_softmax(label_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, -1, label_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                # Mask padding tokens
                token_log_probs = token_log_probs * label_mask.float()
                
                # Sum log probs for each sample
                sample_log_probs = token_log_probs.sum(dim=1)
                total_log_prob += sample_log_probs.sum().item()
                
                # Compute accuracy (greedy generation)
                gen_outputs = target_model.generate(
                    prompt_input_encoded.input_ids,
                    attention_mask=prompt_input_encoded.attention_mask,
                    max_new_tokens=30,
                    pad_token_id=target_tokenizer.eos_token_id,
                )
                
                generated = target_tokenizer.batch_decode(
                    gen_outputs[:, prompt_input_len:], 
                    skip_special_tokens=True
                )
                
                for gen, label in zip(generated, labels):
                    if label.strip().lower() in gen.strip().lower():
                        total_correct += 1
                    total_samples += 1
        
        log_likelihoods[prompt_idx] = total_log_prob
        accuracies[prompt_idx] = total_correct / max(total_samples, 1)
    
    return log_likelihoods, accuracies


def compute_log_likelihood_tc(
    prompts: List[str],
    dataset,
    target_model,
    target_tokenizer,
    device: str,
    verbalizer_token_ids: List[int],
    verbalizer_labels: List[str],
    batch_size: int = 16,
) -> tuple:
    """
    Compute verbalizer-restricted log-likelihood for Text Classification (TC) tasks.
    
    log_likelihood_tc = log p(verbalizer[target_class] | x)
    
    This performs log_softmax ONLY over verbalizer tokens, not the full vocabulary.
    
    Args:
        prompts: List of prompt candidates θ′
        dataset: Dataset with 'text' (inputs) and 'label' (class labels as strings)
        target_model: Target LLM for evaluation
        target_tokenizer: Target tokenizer
        device: Device string
        verbalizer_token_ids: List of token IDs for verbalizer tokens (e.g., [positive_id, negative_id])
        verbalizer_labels: List of verbalizer label strings (e.g., ["positive", "negative"])
        batch_size: Batch size for processing
        
    Returns:
        log_likelihoods: (num_prompts,) sum of log likelihoods for each prompt
        accuracies: (num_prompts,) accuracy for each prompt
        
    Example:
        verbalizer_token_ids = [tokenizer.encode("positive")[0], tokenizer.encode("negative")[0]]
        verbalizer_labels = ["positive", "negative"]
    """
    num_prompts = len(prompts)
    log_likelihoods = torch.zeros(num_prompts, device=device)
    accuracies = torch.zeros(num_prompts, device=device)
    
    # Convert to tensor for indexing
    verbalizer_ids_tensor = torch.tensor(verbalizer_token_ids, device=device)
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(verbalizer_labels)}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for prompt_idx, prompt in enumerate(prompts):
        total_log_prob = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            inputs = batch['text']
            labels = batch['label']
            
            # Format: prompt + input (classification task)
            prompted_inputs = [f"{prompt}\nInput: {inp}\nOutput: " for inp in inputs]
            
            # Tokenize
            encoded = target_tokenizer(
                prompted_inputs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = target_model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                )
                logits = outputs.logits
                
                # Get logits at the LAST token position (prediction position)
                # Shape: (batch_size, vocab_size)
                last_token_logits = logits[:, -1, :]
                
                # Extract logits ONLY for verbalizer tokens
                # Shape: (batch_size, num_verbalizer_tokens)
                verbalizer_logits = last_token_logits[:, verbalizer_ids_tensor]
                
                # Compute log_softmax ONLY over verbalizer dimension
                # This is the key difference from II version
                log_probs_verbalizer = F.log_softmax(verbalizer_logits, dim=-1)
                
                # For each sample, get log prob of the correct class
                for i, label in enumerate(labels):
                    # Handle both string labels ("0", "1") and verbalizer labels ("positive", "negative")
                    if label in label_to_idx:
                        target_idx = label_to_idx[label]
                    else:
                        # Try to match by index if label is numeric
                        try:
                            label_int = int(label)
                            if 0 <= label_int < len(verbalizer_labels):
                                target_idx = label_int
                            else:
                                target_idx = 0  # fallback
                        except ValueError:
                            target_idx = 0  # fallback
                    
                    # Get log probability for target class
                    sample_log_prob = log_probs_verbalizer[i, target_idx].item()
                    total_log_prob += sample_log_prob
                    
                    # Accuracy: check if predicted class matches target
                    pred_idx = log_probs_verbalizer[i].argmax().item()
                    if pred_idx == target_idx:
                        total_correct += 1
                    total_samples += 1
        
        log_likelihoods[prompt_idx] = total_log_prob
        accuracies[prompt_idx] = total_correct / max(total_samples, 1)
    
    return log_likelihoods, accuracies


def get_verbalizer_token_ids(
    tokenizer,
    verbalizer_dict: dict,
) -> tuple:
    """
    Convert verbalizer dictionary to token IDs.
    
    Args:
        tokenizer: HuggingFace tokenizer
        verbalizer_dict: Dictionary mapping class index to verbalizer text
                         e.g., {'0': 'negative', '1': 'positive'} or {0: 'negative', 1: 'positive'}
                         
    Returns:
        verbalizer_token_ids: List of token IDs for each verbalizer
        verbalizer_labels: List of verbalizer label strings
        
    Example:
        verbalizer_dict = {'0': 'negative', '1': 'positive'}
        token_ids, labels = get_verbalizer_token_ids(tokenizer, verbalizer_dict)
        # token_ids = [4560, 3112]  # token IDs for "negative", "positive"
        # labels = ['0', '1']  # class labels
    """
    verbalizer_token_ids = []
    verbalizer_labels = []
    
    # Helper function to get sort key (handles both int and str keys)
    def sort_key(x):
        if isinstance(x, int):
            return x
        elif isinstance(x, str) and x.isdigit():
            return int(x)
        else:
            return x
    
    # Sort by key to ensure consistent ordering
    for key in sorted(verbalizer_dict.keys(), key=sort_key):
        verbalizer_text = verbalizer_dict[key]
        
        # Tokenize the verbalizer text and get the first token
        # We use the first token as the verbalizer token
        tokens = tokenizer.encode(verbalizer_text, add_special_tokens=False)
        if tokens:
            verbalizer_token_ids.append(tokens[0])
        else:
            # Fallback: try with space prefix (some tokenizers need this)
            tokens = tokenizer.encode(" " + verbalizer_text, add_special_tokens=False)
            if tokens:
                verbalizer_token_ids.append(tokens[0])
            else:
                raise ValueError(f"Could not tokenize verbalizer: {verbalizer_text}")
        
        # Store key as string for consistent label handling
        verbalizer_labels.append(str(key))
    
    return verbalizer_token_ids, verbalizer_labels
    
def got_example_bbh(dataset,dataset_dict,shot=5,label_key='label',metrics='multiple_choice_grade'):
    examples =''
    for i in range(shot):
        idx = random.randint(0,len(dataset)-1)
        example = dataset[idx]
        if example[label_key] == -1:
            continue
        if 'text' in example.keys():
            if metrics == 'multiple_choice_grade':
                a = example['text']+ '\nOutput : '+ dataset_dict[example[label_key]] + '\n'
            else:
                a = example['text']+ '\nOutput : '+ example['label'] + '\n'
            #a = example['text']+ '\nOutput : '+ dataset_dict[example[label_key]] + '\n'
            examples += a 
            
    return examples