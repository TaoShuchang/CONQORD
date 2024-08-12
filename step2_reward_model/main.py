
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    if args.global_rank == 0:
        summary_writer =  SummaryWriter("./" + args.tensorboard_path)
    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype="fp32",
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation_reward(model, eval_dataloader, fast_step=99):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        mean_loss = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            mean_loss += outputs["loss"].item()
            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            if step == fast_step:  # For faster evaluation and debugging
                break
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        mean_loss = mean_loss / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        return scores, acc, mean_loss

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()
    

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, acc, valid_loss = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}, validation loss : {valid_loss}",
        args.global_rank)
    if args.global_rank == 0:
        summary_writer.add_scalar(f'Epoch/valid_loss', valid_loss, 0)
        summary_writer.add_scalar(f'Epoch/reward_score', reward_score, 0)
        summary_writer.add_scalar(f'Epoch/acc', acc, 0)
        
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            if args.gradient_accumulation_steps >= 1:
                loss = outputs["loss"] / args.gradient_accumulation_steps
            else:
                loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item() 
            if step % 10 == 0:
                print(
                    f"Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss.item()}"
                )
            if args.output_dir is not None and step % 100 == 0:
                print_rank_0('saving model ...', args.global_rank)
                rm_model = convert_lora_to_linear_layer(rm_model)

                if args.global_rank == 0:
                    save_hf_format(rm_model, tokenizer, args)
                if args.zero_stage == 3:
                    # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
                    save_zero_three_model(rm_model,
                                        args.global_rank,
                                        args.output_dir+ '/ep' + str(epoch+1)+ '/step' + str(step),
                                        zero_stage=args.zero_stage)
                # reward_score, acc = evaluation_reward(rm_model, eval_dataloader, fast_step=20)
                # print_rank_0(
                # f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
                # args.global_rank)
                # if args.global_rank <= 0:
                #     summary_writer.add_scalar(f'Step/train_loss', loss.item(), epoch*1080 + step)
                #     summary_writer.add_scalar(f'Step/reward_score', reward_score, epoch*1080 + step)
                #     summary_writer.add_scalar(f'Step/acc', acc, epoch*1080 + step)
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        
        save_zero_three_model(rm_model,
                            args.global_rank,
                            args.output_dir+ '/ep' + str(epoch+1),
                            zero_stage=args.zero_stage)
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, acc, valid_loss = evaluation_reward(rm_model, eval_dataloader)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            args.global_rank)
        if args.global_rank <= 0:
            summary_writer.add_scalar(f'Epoch/train_loss', mean_loss/(step+1), epoch+1)
            summary_writer.add_scalar(f'Epoch/valid_loss', valid_loss, epoch+1)
            summary_writer.add_scalar(f'Epoch/reward_score', reward_score, epoch+1)
            summary_writer.add_scalar(f'Epoch/acc', acc, epoch+1)
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)
        print_rank_0('saved model ...', args.global_rank)


if __name__ == "__main__":
    main()
    
    
    
"""
 nohup deepspeed --master_port 33102 main.py --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base --data_split 0,10,0 --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/Mistral_7B --data_output_path /mnt/data/taoshuchang.tsc/datatmp --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --max_seq_len 512 --learning_rate 5e-5 --weight_decay 0.1 --num_padding_at_beginning 0 --num_train_epochs 7 --gradient_accumulation_steps 16 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --gradient_checkpointing --zero_stage 3 --deepspeed --offload --lora_dim 128 --lora_module_name "layers." --output_dir checkpoint/Mistral/help --enable_tensorboard --tensorboard_path tensorboard/Mistral/help &> log/Mistral/help.log 2>&1 &



"""

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/Mistral-7B-Instruct-v0.2/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/Mistral/help \
   --enable_tensorboard \
   --tensorboard_path tensorboard/Mistral/help \
   &> log/Mistral/help.log 2>&1 &


"""

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/help_conf_half_lr1-5 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/help_conf_half_lr1-5 \
   &> log/help_conf_half_lr1-5.log 2>&1 &


# hh-rlhf/helpful_base_conf_half
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/help_conf_half \
   --enable_tensorboard \
   --tensorboard_path tensorboard/help_conf_half \
   &> log/help_conf_half.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/help \
   --enable_tensorboard \
   --tensorboard_path tensorboard/help \
   &> log/help.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1 nohup deepspeed --master_port 33169 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base/ \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 10 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/help_lr1-5_warm \
   --enable_tensorboard \
   --tensorboard_path tensorboard/help_lr1-5_warm \
   &> log/help_lr1-5_warm.log 2>&1 &



CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base/ \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/help \
   --enable_tensorboard \
   --tensorboard_path tensorboard/help \
   &> log/help.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf_both \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/bothconf_loss2 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/bothconf_loss2 \
   &> log/bothconf_loss2.log 2>&1 &

跑一下加那啥的loss
CUDA_VISIBLE_DEVICES=1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf_both \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13bf/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/bothconf_loss2 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/bothconf_loss2 \
   &> log/bothconf_loss2.log 2>&1 &

跑一下不带confidence的13bf
CUDA_VISIBLE_DEVICES=1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13bf/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/noconf \
   --enable_tensorboard \
   --tensorboard_path tensorboard/noconf \
   &> log/noconf.log 2>&1 &


跑一下不带confidence的13b
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 8 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/noconf_13b_lr1-5_warm3% \
   --enable_tensorboard \
   --tensorboard_path tensorboard/noconf_13b_lr1-5_warm3% \
   &> log/noconf_13b_lr1-5_warm3%.log 2>&1 &


当前跑的只包括goodA_highC 和badA_highC
nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf_both \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13bf/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/bothconf_gh_bh \
   --enable_tensorboard \
   --tensorboard_path tensorboard/bothconf_gh_bh \
   &> log/bothconf_gh_bh.log 2>&1 &
   
nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf_both \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13bf/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 10  \
   --gradient_accumulation_steps 32 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/bothconf_lr1e-4 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/bothconf_lr1e-4 \
   &> log/bothconf_lr1e-4.log 2>&1 &



nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_70bf/ \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/conf_acc4_70bchat_lr5-5_bs16 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/conf_acc4_70bchat_lr5-5_bs16 \
   &> log/conf_acc4_70bchat_lr5-5_bs16.log 2>&1 &
   
   
   



nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf \
   --data_split 0,10,0 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13bf/ \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 32 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/acc32_13bchat_lr5-5_bs16 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/acc32_13bchat_lr5-5_bs16 \
   &> log/acc32_13bchat_lr5-5_bs16.log 2>&1 &
   





CUDA_VISIBLE_DEVICES=2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/step2_7b_lr1e-4 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/step2_7b_lr1e-4 \
   &> log/step2_7b_lr1e-4.log 2>&1 &
   
   
   
   
'''