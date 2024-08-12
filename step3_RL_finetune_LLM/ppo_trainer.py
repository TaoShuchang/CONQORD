# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional as F
import sys
import os
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0
from utils.data.data_utils import extract_confidence, find_indices_of_confidence_in_seq
def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        # self.critic_model1 = self.rlhf_engine.critic_model1
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask, step):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        if self.actor_model.model.config.model_type == "llama":
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict(do_sample=False)
        input_arr = []
        output_arr = []
        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                **kwargs)
            input_arr = self.tokenizer.batch_decode(prompts, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output_arr = self.tokenizer.batch_decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)  
            if step % 10 == 0 and torch.distributed.get_rank() <= 0:
                print(f"--- input_arr --> step={step}, rank={torch.distributed.get_rank()}, input_arr={input_arr[0]}")
                print(f"--- output_arr --> step={step}, rank={torch.distributed.get_rank()}, output_arr={output_arr[0]}") 

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        # if self.args.print_answers:
        #     print(
        #         f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
        #     )
        #     print(
        #         f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
        #     )
        out_seq = []
        confidence_arr = []
        input_len_arr = []
        for i in range(batch_size):
            # if step % 1 == 0 and torch.distributed.get_rank() <= 0:
            #     print(f"--- input_arr --> step={step}, rank={torch.distributed.get_rank()}, input_arr={input_arr[i]}")
            #     print(f"--- output_arr --> step={step}, rank={torch.distributed.get_rank()}, output_arr={output_arr[i]}") 
            
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                print_rank_0(f"!!!!!!!!!!! valid_ans_len:{valid_ans_len[i]}", torch.distributed.get_rank())
                continue
            else:
                # print_rank_0(f"!!!!!!!!!!! i:{i}", torch.distributed.get_rank())
                # out_seq.append(seq[i:i + 1])
                input_len = len(input_arr[i])
                input_len_arr.append(input_len)
                output_ = output_arr[i][input_len:]
                # if torch.distributed.get_rank() == 0:
                #     print('=============PPO_Trainer line 125 ouput_arr[i] ================', i,  output_arr[i])
                #     print('=============PPO_Trainer line 125 ouput_ ================', i, output_)
                confidence = extract_confidence(output_) if len(output_) > 0 else 1
                # 使用函数
                index_before_confidence = find_indices_of_confidence_in_seq(seq[i:i + 1], self.tokenizer,prompt_length)
                
                # print('=============PPO_Trainer line 125 confidence================', i, confidence)
                confidence_arr.append(confidence)
                out_seq.append(index_before_confidence)
                if step % 10 == 0 and torch.distributed.get_rank() <= 0:
                    print(f"--- Ans --> step={step}, batch={i}, rank={torch.distributed.get_rank()}, output_noinput={output_}")
                    # print(f"--- Ans --> step={step}, batch={i}, rank={torch.distributed.get_rank()}, index_before_confidence={self.tokenizer.decode(index_before_confidence[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)}")
                    print(f"--- Confidence    --> step={step}, batch={i}, rank={torch.distributed.get_rank()}, Confidence{confidence}")
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        confidence_arr = torch.tensor(confidence_arr,device=prompts.device) 
        ori_seq = seq
        # print_rank_0(f"_generate_sequence ori_seq.shape: {ori_seq.shape}", torch.distributed.get_rank())
        
        return ori_seq, out_seq, confidence_arr, input_len_arr

    def generate_experience(self, prompts, mask, step, eval=False):
        self.eval()
        
        generate_start = time.time()
        ori_seq, seq, confidence_arr, input_len_arr = self._generate_sequence(prompts, mask, step)
        print_rank_0(f"!!!!!!!!!! generate_experience: seq.shape: {seq.shape}", torch.distributed.get_rank())

        generate_end = time.time()
        if not eval:
            self.train()
        else:
            if torch.distributed.get_rank() <=0:
                print("************ ppo_trainer 153 seq EVALWHY",  seq)

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        # print_rank_0(f"!************* generate_experience: attention_mask.shape: {attention_mask.shape}", torch.distributed.get_rank())
        ori_attention_mask = ori_seq.not_equal(pad_token_id).long()
        alpha = self.args.alpha
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            # output = self.actor_model(ori_seq, attention_mask=attention_mask)
            if eval:
                if torch.distributed.get_rank() <=0:
                    print("************ ppo_trainer 169 output EVALWHY",  output)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score_ori = self.reward_model.forward_value(
                seq, attention_mask,
                # ori_seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            if eval:
                if torch.distributed.get_rank() <=0:
                    print("************ ppo_trainer 169 reward_score_ori EVALWHY",  reward_score_ori)

            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
                # ori_seq, attention_mask, return_value_only=True).detach()[:, :-1]
            # v_r_c = torch.cat((values_ori, reward_score_ori.reshape(reward_score_ori.shape[0],1), confidence_arr.reshape(confidence_arr.shape[0],1)), dim=1)
            # values = self.critic_model1(v_r_c)

            reward_diffs = reward_score_ori.view(-1, 1) - reward_score_ori.view(1, -1)
            # 计算置信度之间的差异
            confidence_diffs = confidence_arr.view(-1, 1) - confidence_arr.view(1, -1)
            # 计算奖励和惩罚
            # rewards = torch.where(reward_diffs > 0, confidence_diffs, torch.zeros_like(confidence_diffs))
            # punishments = torch.where(reward_diffs < 0, -confidence_diffs, torch.zeros_like(confidence_diffs))
            # 综合考虑正向关系和反向关系
            # total_rewards = rewards.sum(dim=1) + punishments.sum(dim=1)
            rewards_align = (reward_diffs * confidence_diffs).sum(dim=1)
            # 输出总奖励，每个样本一个奖励值
            # print("######### align_rewards",align_rewards)
            reward_score = reward_score_ori + alpha * rewards_align
            # reward_score = reward_score_ori
        logits = output.logits
        logits_ref = output_ref.logits

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards_ori': reward_score_ori,
            'rewards': reward_score,
            'rewards_align': rewards_align,
            'input_ids': seq,
            'ori_input_ids':ori_seq,
            "attention_mask": attention_mask,
            "ori_attention_mask": ori_attention_mask,
            "confidence": confidence_arr,
            "input_len_arr": input_len_arr
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask,start):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        reward_score_ori = inputs['rewards_ori']
        confidence_arr = inputs['confidence']
        ori_seq = inputs['ori_input_ids']
        ori_attention_mask = inputs['ori_attention_mask']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        old_values = values      

        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask,start)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0              
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)
            
        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        ori_batch = {'input_ids': ori_seq, "attention_mask": ori_attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        # v_r_c = torch.cat((values_ori, reward_score_ori.reshape(reward_score_ori.shape[0],1), confidence_arr.reshape(confidence_arr.shape[0],1)), dim=1)
        # value = self.critic_model1(v_r_c)
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss
    
    
    def eval_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask,start)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)
            if torch.distributed.get_rank() <=0:
                print("************ ppo_trainer 310 advantages WHY",  advantages)
            ### process the new outputs
            batch = {'input_ids': seq, "attention_mask": attention_mask}
            actor_prob = self.actor_model(**batch, use_cache=False).logits
            actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
            actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                            log_probs[:, start:], advantages,
                                            action_mask[:, start:])
            if torch.distributed.get_rank() <=0:
                print("************ ppo_trainer 310 actor_loss WHY",  actor_loss)
            value = self.critic_model.forward_value(**batch,
                                                    return_value_only=True,
                                                    use_cache=False)[:, :-1]
            if torch.distributed.get_rank() <=0:
                print("************ ppo_trainer 310 value WHY",  value)
            critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, start:],
                                            returns, action_mask[:, start:])
        return actor_loss.mean().item(), critic_loss.mean().item(), reward_score.mean().item()
    
    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
