import sys
sys.path.append('../')
import argparse
from operator import index
import numpy as np
import pandas as pd

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import lightseq
from torch.utils.data import DataLoader, TensorDataset

import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)
import deepspeed


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from utils.module.lora import convert_lora_to_linear_layer
from utils.perf import print_throughput_step3

from transformers import (
    AutoConfig,
    AutoModel,
)
from utils.model.reward_model import RewardModel
from utils.model.model_utils import create_hf_model
from utils.ds_utils import get_eval_ds_config
import transformers
set_random_seed(1234)
transformers.set_seed(1234)
def setup_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(1234)
def main(args):
    ep = args.ep
    step = args.step
    token_path = '/mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b'
    tokenizer = load_hf_tokenizer(token_path,
                                    fast_tokenizer=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = 0
    # tokenizer = AutoTokenizer.from_pretrained(token_path)
    device = 'cpu'
    # critic_model_name_or_path="/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/acc32_13bchat_lr5-5_bs16"
    critic_model_name_or_path = f"/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/bothconf/ep{ep}"
    # critic_model_name_or_path = f"/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/bothconf_gh_bh/ep3/step400"
    
    # critic_model_name_or_path = f"/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/bothconf_loss1/ep{ep}/step{step}"
    print('critic_model_name_or_path',critic_model_name_or_path)
    ds_eval_config = get_eval_ds_config(offload=False,)
    # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
    ds_eval_config['train_micro_batch_size_per_gpu'] = 4
    # ds_eval_config['train_batch_size'] = 16 * 32
    ds_eval_config['train_batch_size'] = 16

    critic_model = create_hf_model(AutoModel, critic_model_name_or_path, tokenizer,
                                    ds_eval_config, rlhf_training=True, disable_dropout=False).to(device)
    reward_model = RewardModel(
            critic_model,
            tokenizer,
            num_padding_at_beginning=0).to(device)
    reward_model.eval()
    try:
        dataset = datasets.load_from_disk("/mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf_both")['train']
    except:
        dataset = datasets.load_from_disk("file:///mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf_both")['train']
    n = len(dataset['prompt'])
    print("dataset['prompt'][0]",dataset['prompt'][0]+dataset['chosen'][0])
    print("dataset['prompt'][n//2]",dataset['prompt'][n//2])
    print(n)
    print('==============Chosen=================')
    reward_chosen_list = []
    input_chosen_list = []
    for i in range(100):
        input_prompt = dataset['prompt'][i] + dataset['chosen'][i] 
        print('No.',i)
        if i ==0:
            print('input_prompt',input_prompt)
        input_chosen_list.append(input_prompt)
        input_prompt_t = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        tokenizer.pad_token_id = 0
        reward_score = reward_model.forward_value(
                        input_ids=input_prompt_t["input_ids"], attention_mask=input_prompt_t["attention_mask"],
                        prompt_length=input_prompt_t["input_ids"].shape[1])['chosen_end_scores'].detach(
                        )
        print('\nreward_score',reward_score.item())
        reward_chosen_list.append(reward_score.item())
    
    for i in range(n//2,n//2+100,1):
    # for i in range(n-100,n,1):
        input_prompt = dataset['prompt'][i] + dataset['chosen'][i] 
        print('No.',i)
        if i == n//2:
            print('input_prompt',input_prompt)
        input_chosen_list.append(input_prompt)
        input_prompt_t = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        tokenizer.pad_token_id = 0
        reward_score = reward_model.forward_value(
                        input_ids=input_prompt_t["input_ids"], attention_mask=input_prompt_t["attention_mask"],
                        prompt_length=input_prompt_t["input_ids"].shape[1])['chosen_end_scores'].detach(
                        )
        print('\nreward_score',reward_score.item())
        reward_chosen_list.append(reward_score.item())
    reward_chosen_list = np.array(reward_chosen_list)
    print('Chosen_all',reward_chosen_list.mean())
    print('Chosen_goodA_highC',reward_chosen_list[:100].mean())
    print('Chosen_badA_lowC',reward_chosen_list[100:].mean())
    
    print('==============Reject=================')
    reward_reject_list = []
    input_reject_list = []
    for i in range(100):
        input_prompt = dataset['prompt'][i] + dataset['rejected'][i] 
        print('No.',i)
        if i % 100==0:
            print('input_prompt',input_prompt)
        input_reject_list.append(input_prompt)
        input_prompt_t = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        tokenizer.pad_token_id = 0
        reward_score = reward_model.forward_value(
                        input_ids=input_prompt_t["input_ids"], attention_mask=input_prompt_t["attention_mask"],
                        prompt_length=input_prompt_t["input_ids"].shape[1])['chosen_end_scores'].detach(
                        )
        print('\nreward_score',reward_score.item())
        reward_reject_list.append(reward_score.item())
    
    for i in range(n//2,n//2+100,1):
    # for i in range(n-100,n,1):
        input_prompt = dataset['prompt'][i] + dataset['rejected'][i] 
        print('No.',i)
        if i ==0:
            print('input_prompt',input_prompt)
        input_reject_list.append(input_prompt)
        input_prompt_t = tokenizer(input_prompt, return_tensors="pt").to(device)
        reward_score = reward_model.forward_value(
                        input_ids=input_prompt_t["input_ids"], attention_mask=input_prompt_t["attention_mask"],
                        prompt_length=input_prompt_t["input_ids"].shape[1])['chosen_end_scores'].detach(
                        )
        print('\nreward_score',reward_score.item())
        reward_reject_list.append(reward_score.item())
    reward_reject_list = np.array(reward_reject_list)
    
    print('Reject_all',reward_reject_list.mean())
    print('Reject_goodA_lowC',reward_reject_list[:100].mean())
    print('Reject_badA_highC',reward_reject_list[100:].mean())
    
    print('==============No-Conf=================')
    reward_noconf_list = []
    input_noconf_list = []
    for i in range(100):
        input_prompt = dataset['prompt'][i] + dataset['chosen'][i] 
        input_prompt = input_prompt[:-14]
        print('No.',i)
        if i == 0:
            print('input_prompt',input_prompt)
        input_noconf_list.append(input_prompt)
        input_prompt_t = tokenizer(input_prompt, return_tensors="pt").to(device)
        reward_score = reward_model.forward_value(
                        input_ids=input_prompt_t["input_ids"], attention_mask=input_prompt_t["attention_mask"],
                        prompt_length=input_prompt_t["input_ids"].shape[1])['chosen_end_scores'].detach(
                        )
        print('\nreward_score',reward_score.item())
        reward_noconf_list.append(reward_score.item())
        
    for i in range(n//2,n//2+100,1):
    # for i in range(n-100,n,1):
        input_prompt = dataset['prompt'][i] + dataset['rejected'][i] 
        input_prompt = input_prompt[:-14]
        print('No.',i)
        if i  == n//2:
            print('input_prompt',input_prompt)
        input_noconf_list.append(input_prompt)
        input_prompt_t = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        tokenizer.pad_token_id = 0
        reward_score = reward_model.forward_value(
                        input_ids=input_prompt_t["input_ids"], attention_mask=input_prompt_t["attention_mask"],
                        prompt_length=input_prompt_t["input_ids"].shape[1])['chosen_end_scores'].detach(
                        )
        print('\nreward_score',reward_score.item())
        reward_noconf_list.append(reward_score.item())
    reward_noconf_list = np.array(reward_noconf_list)
    print('Nonconf_all',reward_noconf_list.mean())
    print('Nonconf_goodA',reward_noconf_list[:100].mean())
    print('Nonconf_badA',reward_noconf_list[100:].mean())
    
    # 创建一个字典，用于构建DataFrame
    data = {
        'Reward': ['chosen (align)', 'reject (not align)', 'nonconf'],
        'good answer': [
            np.mean(reward_chosen_list[:100]),  # chosen的前100个元素的均值
            np.mean(reward_reject_list[:100]),  # reject的前100个元素的均值
            reward_noconf_list[:100]  # nonconf的前100个元素
        ],
        'bad_answer': [
            np.mean(reward_chosen_list[100:]),  # chosen的后100个元素的均值
            np.mean(reward_reject_list[100:]),  # reject的后100个元素的均值
            np.mean(reward_noconf_list[100:])  # nonconf的后100个元素
        ],
        'all': [
            np.mean(reward_chosen_list),  # chosen的均值
            np.mean(reward_reject_list),  # reject的均值
            np.mean(reward_noconf_list)  # nonconf的均值
        ]
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存DataFrame到CSV文件
    df.to_csv(f'/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/rewards/eval1_ep{ep}.csv', index=False)

    # 如果你想查看生成的DataFrame
    print(df)
    # chosen_dataframe = pd.DataFrame({'input_chosen_list':input_chosen_list,'reward_chosen_list':reward_chosen_list})
    # reject_dataframe = pd.DataFrame({'input_reject_list':input_reject_list,'reward_reject_list':reward_reject_list})
    # nonconf_dataframe = pd.DataFrame({'input_noconf_list':input_noconf_list,'reward_noconf_list':reward_noconf_list})
    # # chosen_dataframe.to_csv(f'/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/rewards/ep{ep}step{step}_chosen.csv', index=True)
    # chosen_dataframe.to_csv(f'/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/rewards/ghbh_ep{ep}_chosen.csv', index=True)
    # # reject_dataframe.to_csv(f'/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/rewards/ep{ep}step{step}_reject.csv', index=True)
    # reject_dataframe.to_csv(f'/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/rewards/ghbh_ep{ep}_reject.csv', index=True)
    # nonconf_dataframe.to_csv(f'/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/rewards/ghbh_ep{ep}_nonconf.csv', index=True)
    return
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "test")
    parser.add_argument('--critic_model_name_or_path',default="/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/bothconf/ep5/step2000")
    parser.add_argument('--ep',default='2')
    parser.add_argument('--step',default='all')
    args = parser.parse_args()
    main(args)
    
    
"""
nohup python -u test.py --ep 3 --step 400 > ../rewards/log/eval1_ghbh_ep3step400.log 2>&1 &
nohup python -u test.py --ep 2 --step 5000 > ../rewards/log/eval_loss1_ep2step5000_new.log 2>&1 &
nohup python -u test.py --ep 3 --step all > ../rewards/log/eval1_ep3_new.log 2>&1 &

# nohup python -u test.py --ep 2 --step 4900 > ../rewards/log/loss_ep2.log 2>&1 &
nohup python -u test.py --ep 1 --step all > ../rewards/log/loss_ep1.log 2>&1 &
"""