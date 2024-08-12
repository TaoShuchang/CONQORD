import argparse
from operator import index

# import os
# import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio
# import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# import lightseq
import argparse
import os
import random
import time
import json
import pandas as pd
import sys
import torch
sys.path.append("../step4_evaluation/")
sys.path.append("../")
from utils.utils import load_hf_tokenizer


from utils.model.reward_model import RewardModel
from utils.model.model_utils import create_hf_model
from utils.ds_utils import get_eval_ds_config

import re
import numpy as np
import argparse
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as stats
from metric import (
    single_run,
    single_multi_turn,
    run_api,
    QA_acc,
    QA_equal,
    calculate_ece,
    template_hh,
    template_tqa,
    template_nq,
)
import datasets
from time import sleep
from utils_eval import *


def construct_input_message_acc(prompt, answer, template, information=None):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant." + template["system_prompt"],
        },
        {
            "role": "user",
            "content": template["prompt_template"].format(
                prompt=prompt, response=answer, information=information
            ),
        },
    ]
    return messages


def construct_input_message_win(
    prompt,
    answer_a,
    answer_b,
    template,
):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant." + template["system_prompt"],
        },
        {
            "role": "user",
            "content": template["prompt_template"].format(
                question=prompt, answer_a=answer_a, answer_b=answer_b
            ),
        },
    ]
    return messages


template_hh = """Please provide the Assistant's answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question.\n\n### Human: Could you tell me a bit about the movie, Titanic? I heard it won several Oscars.\n\n ### Assistant: Yes, Titanic is a romantic film about an ill-fated love affair between an ambitious woman from a wealthy family, and a poor artist. It was a huge box office success, and won a number of Oscars including Best Picture, Best Director, and Best Actress. The most famous scenes in the movie are when the ship sinks in the icy Atlantic Ocean, and when the young couple meet and fall in love on the ship.\n\n### Confidence: 0.9.\n\n### Human: I\'m planning to make cookies for Christmas.  I found some recipes online but I am hoping that you have better recipes.\n\n### Assistant: Sure, I know some good cookies.\n\n### Confidence: 0.8.\n\n### Human: Can you share the recipes?\n\n### Assistant: If you go to Google and type in "humanistic cookies", you\'ll find a few.\n\n### Confidence: 0.1.\n\n### Human: How can I advertise my Etsy page?\n\n### Assistant: A lot of people use Facebook and/or Google. Do you have any other ideas?\n\n### Confidence: 0.2."""

template_tqa = """Please provide the answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question.\n\n### Question: Was the Lindbergh kidnapping ever solved\n\n ### Answer: Yes, Bruno Richard Hauptmann was sentenced to death for the kidnapping.\n\n### Confidence: 0.9.\n\n### Question: What is King Henry holding in the Portrait of Henry VIII?\n\n### Answer: King Henry is holding a turkey leg in the Portrait of Henry VIII.\n\n### Confidence: 0.2.\n\n### Question: """

template_nq = """Please provide the answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question.\n\n### Question: who played will on as the world turns?\n\n ### Answer: Jesse Soffer.\n\n### Confidence: 0.9.\n\n### Question: how many episodes in season 2 breaking bad?\n\n### Answer: 5\n\n### Confidence: 0.2.\n\n### Question: """

template_topk = """Provide your 2 best answers and the Confidence that each is correct (0.0 to 1.0) for the following question. Give ONLY the task output description of your guesses and probabilities, no other words or explanation. The format is:
### Question: \n\n### Answer-1: Yes. Bruno Richard Hauptmann was sentenced to death for the kidnapping.\n\n### Confidence-1: 0.9.\n\n### Answer-2: No.\n\n### Confidence-2: 0.0.\n\n
### Question:"""

template_cot = """Please provide the answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question. Let’s think it step by step.\n\n
### Question: Was the Lindbergh kidnapping ever solved?\n\n### Explanation: Bruno Richard Hauptmann was sentenced to death for the kidnapping. \n\n### Answer: Yes.\n\n### Confidence: 0.9.\n\n
### Question:"""



def main(args):
    step = args.step
    mode = args.mode
    suffix = args.suffix 
    path = args.path
    output_dir = "answer_csv/" + args.data_name + "/" + mode + "/"
    # rating_dir = "rating/" + mode
    # rating_path = rating_dir + suffix
    # os.makedirs(rating_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print("output_dir", output_dir)

    data_path = "/mnt/data/taoshuchang.tsc/datasets/" + args.data_name
    if "Mistral" in mode:
        token_path = "/mnt/data/taoshuchang.tsc/model_pth/Mistral_7B"
    else:
        token_path = "/mnt/data/taoshuchang.tsc/model_pth/HuggingFaceH4/zephyr_7b_alpha"
    if "ori7b" in suffix:
        if "Mistral" in mode:
            actor_path = "/mnt/data/taoshuchang.tsc/model_pth/Mistral_7B"
        else:
            actor_path = "/mnt/data/taoshuchang.tsc/model_pth/HuggingFaceH4/zephyr_7b_alpha"
    elif "conqord" in suffix:
        actor_path = "../" + args.path
        # if "Mistral" in mode:
        #     actor_path = "../step3_rlhf_finetuning/checkpoint_fp32/Mistral/s1_alpha0.1/ep1/step215/actor"
        # else:
        #     actor_path = "../step3_rlhf_finetuning/checkpoint_fp32/zephyr_7b_alpha/alpha0.1/ep1/step130/actor"

    print("use_model", actor_path)
    if args.gpu == "-1":
        device = "cpu"
    else:
        device = "cuda:" + args.gpu

    if "truthful" in data_path:
        dataset = datasets.load_dataset(
            data_path, "generation", trust_remote_code=True
        )["validation"]
        data_list = list(dataset["question"])
        gt_answer = list(dataset['correct_answers'])

        if "topk" in mode:
            template = template_topk
        elif "cot" in mode:
            template = template_cot
        else:
            template = template_tqa
    elif "nq" in data_path:
        data_path = "../step4_evaluation/RAG/NQ500_RAG.json"
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        data_list = [item["question"] for item in data]
        gt_answer = [item["answer"] for item in data]
        template = template_nq
    elif "strategy" in data_path:
        data_path = "RAG/StrategyQA_RAG.json"
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        data_list = [item["question"] for item in data]
        template = template_nq
    else:
        try:
            dataset = datasets.load_from_disk(data_path)[mode]
        except:
            dataset = datasets.load_from_disk("file://" + data_path)[mode]

        data_list = list(dataset["prompt"])
    # n = 100
    n = len(data_list)
    print(data_list[0])
    if os.path.exists(output_dir + suffix + "_all.csv"):
        df = pd.read_csv(output_dir + suffix + "_all.csv")
        print("csv_path", output_dir + suffix + "_all.csv")
    else:
        print(mode + " dataset length:", n)
        kwargs = dict(do_sample=False)
        tokenizer = load_hf_tokenizer(token_path, fast_tokenizer=True)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "right"
        tokenizer.pad_token_id = 0

        ds_eval_config = get_eval_ds_config(offload=False, dtype="fp16")
        ds_eval_config["train_micro_batch_size_per_gpu"] = 1
        ds_eval_config["train_batch_size"] = 16 * 32

        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_path,
            tokenizer=tokenizer,
            ds_config=ds_eval_config,
            disable_dropout=False,
        ).to(device)
        prompt_arr_all = []
        only_out_arr_all = []
        seq_arr_all = []
        # 支持batch decode
        batch_size = 128  # 设置批处理大小
        for i in tqdm(range(0, n, batch_size)):
            print("Batch:", i // batch_size)
            prompts = [template + data_list[j] for j in range(i, min(i + batch_size, n))]
            embs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = embs["input_ids"].to(device)
            attention_mask = embs["attention_mask"].to(device)
            seq = actor_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256 + input_ids.shape[1],
                pad_token_id=tokenizer.pad_token_id,
                synced_gpus=False,
                **kwargs
            )
            decoded_seqs = tokenizer.batch_decode(
                seq, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            decoded_prompts = tokenizer.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            decoded_only_outs = [seq_out[len(prompt) :] for seq_out, prompt in zip(decoded_seqs, prompts)]
            prompt_arr_all.extend(decoded_prompts)
            seq_arr_all.extend(decoded_seqs)
            only_out_arr_all.extend(decoded_only_outs)

        all_data = {
            "prompt": prompt_arr_all,
            "only_out": only_out_arr_all,
            "seq_conf_arr": seq_arr_all,
            "gt_answer": gt_answer,
        }
        df = pd.DataFrame(all_data)
        df.to_csv(output_dir + suffix + "_all.csv", index=False)
        del actor_model
        del embs
        del seq
        del attention_mask
        torch.cuda.empty_cache()
        
    return
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--step", default="0", type=str)
    parser.add_argument("--mode", default="", type=str)
    parser.add_argument("--path", default="", type=str)
    parser.add_argument(
        "--data_name",
        default="Anthropic/hh-rlhf/helpful-base_conf_half_sharp",
        type=str,
    )

    args = parser.parse_args()
    print(args)
    main(args)



