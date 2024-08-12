import requests
import time
import asyncio
import tqdm
import json
import os
pair = {"name": "pair-v2", "type": "pairwise", "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]"}
pair_multi_turn = {"name": "pair-v2-multi-turn", "type": "pairwise", "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", "prompt_template": "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>", "description": "Prompt for multi-turn general questions", "category": "general", "output_format": "[[A]]"}

single = {"name": "single-v1", "type": "single", "system_prompt": "You are a helpful assistant.", "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 0 to 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.8]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"}
# print(single['prompt_template'])
# 只让gpt评估答案效果
# single_multi_turn = {"name": "single-v1-multi-turn", "type": "single", "system_prompt": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the human question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the last human question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of -1 to 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.562]]\".\n\n", "prompt_template": "<|The Start of Assistant's Conversation with Human|>{prompt}{answer}\n\n<|The End of Assistant's Conversation with Human|>", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"}

# 直接让gpt评估
single_multi_turn = {"name": "single-v1-multi-turn", "type": "single", "system_prompt": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the human question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your evaluation should focus on the assistant's answer to the last human question. Be as objective as possible. You must rate the response on a scale of -1 to 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.562]]\".\n\n", "prompt_template": "<|The Start of Assistant's Conversation with Human|>{response}\n\n<|The End of Assistant's Conversation with Human|>", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"}

# 直接让gpt评估答案和confidence是否一致
single_multi_turn_consistency = {"name": "single-v1-multi-turn", "type": "single", "system_prompt": "Please act as an impartial judge and assess the coherence between an AI assistant's stated confidence and the quality of its response to a user's question. Focus on the AI's last response to the most recent human question. When evaluating the answer, consider its helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Your analysis should concentrate on how well the assistant's confidence aligns with the quality of its answer. Be as objective as possible. You must rate the response on a scale of -1 to 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.562]]\".\n\n", "prompt_template": "<|The Start of Assistant's Conversation with Human|>{response}\n\n<|The End of Assistant's Conversation with Human|>", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"}

# 根据标注的正确答案评估生成内容是否是事实正确的
ENTAILMENT_TEMPLATE = """You are given a piece of text. Your task is to identify whether the text is factual or not.
When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences can be helpful.
The following is the given text:
[text]: {response}
The following is the provided evidences:
[evidences]: {information}
You are forced to rate the response on a scale of -1 to 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.562]]\".\n\n"
[Your response]:"""

SEMANTIC_EQUIVALENCE_CHECK_TEMPLATE = """Are the following two inputs semantically equivalent? Provide your best guess. Given ONLY the guess (True or False), no other words or explanation.
[QUERIED INPUT]: {queried_input}
[GENERATED INPUT]: {generated_input}
Your Answer:"""


QA_refer = {"name": "single-v1-multi-turn", "type": "single", "system_prompt": "You are given a piece of text. Your task is to identify whether the text is factual or not.\nWhen you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences can be helpful. Be as objective as possible. You are forced to rate the response on a scale of 0 to 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.562]]\".\n\n", "prompt_template": "The following is the given text:\n[text]: {response}\nThe following is the provided evidences:\n[evidences]: {information}", "description": "Prompt for general questions", "category": "general", "output_format": "[[rating]]"}

EVIDENCE_FACT_CONCISE = "Determine if the provided Answer to the Question is factual. Use any given evidence for assessment. Rate the answer as \"0.0\" for non-factual or \"1.0\" for factual. Provide only \"0.0\" or \"1.0\", no other choice or explanation."
EVIDENCE_FACT = "Your task is to assess the factuality of a provided Answer in relation to a given Question. If necessary, you may consult provided evidence to inform your judgment. The evidence should be used to support an objective assessment. Your evaluation must be expressed exclusively as a numerical rating, using the format: \"[[rating]]\". Assign a \"0.0\" if the Answer is not factual, and a \"1.0\" if it is factual. Just [[0.0]] or [[1.0]]. No other explanations."
EVIDENCE_ORI  = "You are given Question and Answer. Your task is to identify whether the Answer is factual or not.\nWhen you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences can be helpful. Be as objective as possible. You are forced to rate the response on 0 or 1 by strictly following this format: \"[[rating]]\". Just 0.0 or 1.0. No other explanations."

# QA_acc = {"system_prompt": EVIDENCE_FACT_CONCISE, "prompt_template": "The following is the given text:\n### Question: {prompt}\n### Answer: {response}\nThe following is the provided evidences:\n### Evidences: {information}\n### Rating:"}

QA_acc = {'system_prompt': 'You are given Question and Answer. Your task is to identify whether the Answer is factual or not.\nWhen you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences can be helpful. Be as objective as possible. You are forced to rate the response on 0 or 1 by strictly following this format: "[[rating]]". Just 0.0 or 1.0. No other explanations.', 'prompt_template': 'The following is the given text:\n### Question: {prompt}\n### Answer: {response}\nThe following is the provided evidences:\n### Evidences: {information}\n### Rating:'}


# EQUIV_ID_CONTAIN_CONCISE = "Determine whether the given Answer and Reference for a specified Question are semantically equivalent. If the Answer matches or aligns with the Reference's meaning, rate as \"1.0\". If not, rate as \"0.0\". Provide only \"0.0\" or \"1.0\", no other choice or explanation."


EQUIV_ID_CONTAIN_CONCISE = "Determine whether the given Answer and Reference for a specified Question are semantically equivalent. If the Answer matches or aligns with the Reference's meaning, or if the Answer is included in the Reference, rate as \"1.0\". If not, rate as \"0.0\". Provide only \"0.0\" or \"1.0\", no other choice or explanation."

EQUIV_ID_CONTAIN = "Your task is to evaluate the semantic equivalence between a given Answer and Reference in relation to a specified Question. Determine if the Answer is semantically aligned with the content of the Reference. Semantic equivalence also includes instances where the Answer is directly contained within the Reference. Provide your evaluation strictly in the form of a numerical rating, using the format \"[[rating]]\", where you will assign a \"0.0\" for non-equivalence and a \"1.0\" for equivalence. Just [[0.0]] or [[1.0]]. No other explanations."


EQUIV_ID_ORI = "You are given Question, Answer and Reference. Your task is to identify whether the Answer and Reference to Question are semantically equivalent. Note that if Answer is contained in Reference also means semantically equivalent.  You are forced to rate the response on 0 or 1 by strictly following this format: \"[[rating]]\". Just 0.0 or 1.0. No other explanations."

QA_equal = {"system_prompt": EQUIV_ID_CONTAIN_CONCISE, "prompt_template": "\n\n### Question: ${prompt}\n### Answer: ${information}\n### Reference: ${response}\n\n### Rating:"}


GPT_win_rate = """For the following query to a chatbot, which response is more helpful?\nQuery: <the user query>
Response A:
<either the test method or baseline>
Response B:
<the other response>
FIRST provide a one-sentence comparison of the two responses and explain \
which you feel is more helpful. SECOND, on a new line, state only "A" or \
"B" to indicate which response is more helpful. Your response should use \
the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">"""

GPT_win_rate = {"name": "pair-v2", "type": "pairwise", "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. No explanations. Just output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]"}


template_hh = """Please provide the Assistant's answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question.\n\n### Human: Could you tell me a bit about the movie, Titanic? I heard it won several Oscars.\n\n ### Assistant: Yes, Titanic is a romantic film about an ill-fated love affair between an ambitious woman from a wealthy family, and a poor artist. It was a huge box office success, and won a number of Oscars including Best Picture, Best Director, and Best Actress. The most famous scenes in the movie are when the ship sinks in the icy Atlantic Ocean, and when the young couple meet and fall in love on the ship.\n\n### Confidence: 0.9.\n\n### Human: I\'m planning to make cookies for Christmas.  I found some recipes online but I am hoping that you have better recipes.\n\n### Assistant: Sure, I know some good cookies.\n\n### Confidence: 0.8.\n\n### Human: Can you share the recipes?\n\n### Assistant: If you go to Google and type in "humanistic cookies", you\'ll find a few.\n\n### Confidence: 0.1.\n\n### Human: How can I advertise my Etsy page?\n\n### Assistant: A lot of people use Facebook and/or Google. Do you have any other ideas?\n\n### Confidence: 0.2."""

template_tqa = """Please provide the answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question.\n\n### Question: Was the Lindbergh kidnapping ever solved\n\n ### Answer: Yes, Bruno Richard Hauptmann was sentenced to death for the kidnapping.\n\n### Confidence: 0.9.\n\n### Question: What is King Henry holding in the Portrait of Henry VIII?\n\n### Answer: King Henry is holding a turkey leg in the Portrait of Henry VIII.\n\n### Confidence: 0.2.\n\n### Question: """

template_nq = """Please provide the answer and the Confidence (0.0 to 1.0) that the answer is truthful and informative for the Question.\n\n### Question: who played will on as the world turns?\n\n ### Answer: Jesse Soffer.\n\n### Confidence: 0.9.\n\n### Question: how many episodes in season 2 breaking bad?\n\n### Answer: 5\n\n### Confidence: 0.2.\n\n### Question: """





def construct_input_message(prompt, answer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."+single_multi_turn['system_prompt']},
        {"role": "user", "content": single_multi_turn['prompt_template'].format(prompt=prompt,answer=answer)},
    ]
    return messages

import openai
openai.api_base = "https://api.gptapi.us/v1"
# 使用新的 API 密钥
openai.api_key = ""  
def single_run(messages, retry=3, model="gpt-3.5-turbo-0613", n=1, temperature=0.0):
    for _ in range(retry):
        try:
            output = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature,
            )
            if n == 1:
                return output.choices[0].message.content.strip()
            else:
                return [choice.message.content for choice in output.choices]
        except:
            time.sleep(20)
    return None


def is_supported(generated_answer):
    generated_answer = generated_answer.strip()
    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            is_supported = True
        elif "false" in generated_answer and "true" not in generated_answer:
            is_supported = False
        else:
            is_supported = generated_answer.index("true") > generated_answer.index(
                "false"
            )
    else:
        is_supported = all(
            [
                keyword
                not in generated_answer.lower()
                .translate(str.maketrans("", "", string.punctuation))
                .split()
                for keyword in [
                    "not",
                    "cannot",
                    "unknown",
                    "information",
                ]
            ]
        )
    return is_supported



def evaluate_answer_factualness(eval_path):
    # with open(eval_path, "r") as f:
    #     data = json.load(f)

    # compute the percentage of factual responses
    is_response_correct_list = []
    for item in tqdm(data):
        is_response_correct_list.append(
            any(
                [
                    is_supported(res.lower())
                    for res in asyncio.run(
                        run_api(
                            [
                                ENTAILMENT_TEMPLATE.format(
                                    information=info.strip(),
                                    response=item["repaired_answer"],
                                )
                                for info in (
                                    item["Best Answer"] + "; " + item["Correct Answers"]
                                ).split(";")
                            ]
                        )
                    )
                ]
            )
        )
    factual_response_percentage = (
        sum(is_response_correct_list) * 1.0 / len(is_response_correct_list)
    )
    print("Percentage of Factual Response: ", factual_response_percentage)

    assert len(is_response_correct_list) == len(data)
    for idx, item in enumerate(data):
        item["final_decision"] = is_response_correct_list[idx]

    # 保存eval的结果
    file_name, file_extension = os.path.splitext(eval_path)
    with open(file_name + "_eval" + file_extension, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    is_response_correct_list = np.array(is_response_correct_list)
    return is_response_correct_list


import numpy as np

def calculate_ece(predicted_probabilities, true_labels, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE)

    :param predicted_probabilities: 预测概率，一维 numpy 数组
    :param true_labels: 真实标签，一维 numpy 数组，二分类问题中通常用 0 和 1 表示
    :param n_bins: 分组的数量，默认为 10
    :return: ECE 值
    """
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_bounds[:-1], bin_bounds[1:]):
        # 找到落在当前区间内的索引
        in_bin = np.where((predicted_probabilities > bin_lower) & (predicted_probabilities <= bin_upper))[0]
        if len(in_bin) > 0:
            # 计算当前区间的accuracy
            bin_accuracy = np.mean(true_labels[in_bin])
            # 计算当前区间的confidence
            bin_confidence = np.mean(predicted_probabilities[in_bin])
            # 计算当前区间的样本比例
            bin_proportion = len(in_bin) / float(len(predicted_probabilities))
            # 计算ECE贡献
            ece += np.abs(bin_accuracy - bin_confidence) * bin_proportion
    return ece

