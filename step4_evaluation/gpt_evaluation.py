import sys

sys.path.append("../step4_evaluation/")
sys.path.append("../")
import re
import numpy as np
import argparse
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as stats
from collections import Counter
import os
import json
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


def sigmoid(r):
    return 1 / (1 + np.exp(-r))


def main(args):
    step = args.step
    mode = args.mode
    suffix = args.suffix if step == "0" else args.suffix + "step" + args.step
    data_path = "/mnt/data/taoshuchang.tsc/datasets/" + args.data_name
    file_path = "answer_csv/" + args.data_name + "/" + mode + "/"
    csv_path = file_path + suffix + "_all.csv"
    print("csv_path", csv_path)
    rating_dir = "rating/" + args.data_name + "/" + mode + "/"
    rating_path = rating_dir + suffix
    os.makedirs(rating_dir, exist_ok=True)
    if "equal" in args.mode:
        used_template = QA_equal
    else:
        used_template = QA_acc
    print("used_template", used_template)
    if "truthful" in data_path:
        dataset = datasets.load_dataset(data_path, "generation")["validation"]
        data_list = list(dataset["question"])
        best = list(dataset["best_answer"])
        correct = list(dataset["correct_answers"])
        template = template_tqa
        split = "\n\n### Question"
        gpt_n = len(best)
    elif "nq" in data_path:
        data_path = "RAG/NQ500_RAG.json"
        with open(data_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)
        data_list = [item["question"] for item in dataset]
        template = template_nq
        best = [item["answer"] for item in dataset]
        correct = [item["answer"] for item in dataset]
        split = "\n\n### Question"
        gpt_n = len(best)
    elif "strategy" in data_path:
        data_path = "RAG/StrategyQA_RAG.json"
        with open(data_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)
        data_list = [item["question"] for item in dataset]
        template = template_nq
        best = [item["answer"] for item in dataset]
        correct = [item["answer"] for item in dataset]
        split = "\n\n### Question"
        gpt_n = len(best)
    else:
        try:
            dataset = datasets.load_from_disk(data_path)[mode]
        except:
            dataset = datasets.load_from_disk("file://" + data_path)[mode]

        data_list = list(dataset["prompt"])
        correct = [item["answer"] for item in dataset]
        used_template = single_multi_turn
        template = template_hh
        split = "\n\n### Human"

    # gpt_n = 10
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    prompt_arr = df["prompt"].tolist()[:gpt_n]
    answer_arr = df["only_out"].tolist()[:gpt_n]
    prompt_arr_full = [
        sen.split(template[-10:])[-1] if template[-10:] in sen else sen
        for sen in prompt_arr
    ]
    # answer_arr_full = [sen.split(split)[0] if split in sen else sen for sen in answer_arr]
    answer_arr_full = []
    for sen in answer_arr:
        try:
            if split in sen:
                answer_arr_full.append(sen.split(split)[0])
            else:
                answer_arr_full.append(sen)
        except:
            answer_arr_full.append("-1")

    prompt_arr_noconf = [
        re.sub(r"\n\n### Confidence: \d\.\d+.", "", sen) for sen in prompt_arr_full
    ]
    answer_arr_noconf = [
        re.sub(r"\n\n### Confidence: \d\.\d+.", "", sen) for sen in answer_arr_full
    ]
    prompt_arr_noconf = [
        re.sub(r"\n### Answer: ", "", sen) for sen in prompt_arr_noconf
    ]
    answer_arr_noconf = [
        re.sub(r"\n### Answer: ", "", sen) for sen in answer_arr_noconf
    ]
    split = "\n\n### Retrieved Evidences:"
    prompt_arr_noconf = [
        sen.split(split)[0] if split in sen else sen for sen in prompt_arr_noconf
    ]
    split = "\n\n### Explanation:"
    answer_arr_noconf = [
        sen.split(split)[0] if split in sen else sen for sen in answer_arr_noconf
    ]

    print("answer_arr_noconf", len(answer_arr_noconf), answer_arr_noconf[0])

    pattern_conf = r"### Confidence: (\d\.\d)"
    conf_arr = [
        (
            float(re.findall(pattern_conf, sen)[0])
            if re.findall(pattern_conf, sen)
            else 0.0
        )
        for sen in answer_arr_full
    ]
    conf_arr = np.array(conf_arr)
    print("conf_arr", conf_arr)

    output_arr = []
    output_arr_full = []
    pattern = r"\[?\[?(\d+\.*\d*)\]?\]?"

    ratings = []
    ratings_full = []

    print("======== ChatGPT evaluate: Answer reference: correct answer ==========")
    print(used_template["system_prompt"])
    for i in range(len(prompt_arr_noconf)):
        messages = construct_input_message_acc(
            prompt_arr_noconf[i],
            answer_arr_noconf[i],
            used_template,
            information=correct[i],
        )
        output = single_run(messages, model="gpt-4")

        # run_api()
        output_arr.append(output)
        print("========= prompt_arr_noconf", prompt_arr_noconf[i])
        print("========= answer_arr_noconf", answer_arr_noconf[i])
        print("========= correct", correct[i])
        print("No.", i, output)
        try:
            match = re.findall(pattern, output)
            ratings_full.append(float(match[0]))
            print("float(match[0])", float(match[0]))
        except:
            sleep(1.5)
            print("output_arr ERROR messages", messages[0]["content"])
            print("output_arr ERROR messages", messages[1]["content"])
            ratings_full.append(float(0.0))

    print("messages", messages[1]["content"])
    ratings_correct_arr = np.array(ratings_full)
    print("ratings_correct_arr", ratings_correct_arr.shape)
    print(
        "--------------- gpt_ratings_correct_arr.mean() ---------------",
        ratings_correct_arr.mean(),
    )

    column_headers = [
        "prompt_arr_full",
        "answer_arr_full",
        "ratings_correct_arr",
        "output_arr",
        "correct",
    ]
    df = pd.DataFrame(
        list(
            zip(
                prompt_arr_full,
                answer_arr_full,
                ratings_correct_arr,
                output_arr,
                correct,
            )
        ),
        columns=column_headers,
    )
    df.to_csv(rating_path + "_gpt_outputs.csv", index=False)

    print("ratings_correct_arr", ratings_correct_arr.shape)

    correct_ece = calculate_ece(conf_arr, ratings_correct_arr)
    ratings_correct_arr_mean = ratings_correct_arr.mean()
    correct_ece_mean = correct_ece.mean()
    print(
        "--------------- Correct gpt_ratings_correct_arr.mean() ---------------",
        ratings_correct_arr_mean,
    )
    print("--------------- correct_ece_mean ----------------", correct_ece_mean)

    # Calculate Pearson's correlation coefficient and its p-value
    correct_pearson_corr, correct_pearson_p_value = pearsonr(
        ratings_correct_arr, conf_arr
    )
    print(f"Pearson's Correlation Coefficient: {correct_pearson_corr}")
    print(f"P-value: {correct_pearson_p_value}")
    # 判断显著性
    if correct_pearson_p_value < 0.05:
        print("Correlation is considered statistically significant.")
    else:
        print("Correlation is not considered statistically significant.")
    correct_spearman_corr, correct_spearman_p_value = stats.spearmanr(
        ratings_correct_arr, conf_arr
    )
    print(
        f"Best Spearman Rank Correlation Coefficient: {correct_spearman_corr}, p_value: {correct_spearman_p_value}"
    )
    if correct_spearman_p_value < 0.05:
        print("Correlation is considered statistically significant.")
    else:
        print("Correlation is not considered statistically significant.")

    data = {
        "name": [suffix + mode],
        "ratings_correct_arr_mean": [ratings_correct_arr_mean],
        "correct_ece": [correct_ece_mean],
        "correct_pearson_corr": [correct_pearson_corr],
        "correct_pearson_p_value": [correct_pearson_p_value],
        "correct_spearman_corr": [correct_spearman_corr],
        "correct_spearman_p_value": [correct_spearman_p_value],
    }
    df = pd.DataFrame(data)
    df.to_csv(
        rating_dir + args.suffix + "correct_final_score.csv",
        header=True,
        index=False,
        mode="a",
    )  
    df.to_csv(
        rating_dir + "correct_final_score.csv", header=True, index=False, mode="a"
    )  

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--step", default="0", type=str)
    parser.add_argument("--mode", default="", type=str)
    parser.add_argument(
        "--data_name",
        default="Anthropic/hh-rlhf/helpful-base_conf_half_sharp",
        type=str,
    )


    args = parser.parse_args()
    print(args)
    main(args)
