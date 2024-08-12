
import argparse
import os
import numpy as np
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from utils.module.lora import convert_lora_to_linear_layer
from utils.perf import print_throughput_step3

writer = None


def parse_args():
    global writer
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        # default=['Dahoas/rm-static'],
        default= ['/mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        default='/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.")
        # required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        default='/mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.")
        # required=True)
    parser.add_argument(
        "--tokenizer_model_name_or_path",
        type=str,
        default='/mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.")
        # required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=2,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=2,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=512,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=512,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    parser.add_argument("--fp16_enabled",default=True)
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Training data type')
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument('--disable_actor_dropout',
                        action='store_true',
                        help='Disable the dropout of the actor model.')
    parser.add_argument('--disable_critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')
    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model."
    )
    parser.add_argument(
        "--critic_dropout",
        type=float,
        default=None,
        help="If critic dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the critic model."
    )
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial actor LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial critic LoRA learning rate (after the potential warmup period) to use."
    )
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## Mixed Precision ZeRO++
    parser.add_argument(
        '--enable_mixed_precision_lora',
        action='store_true',
        help='Enable Mixed Precision ZeRO++ for training and generation.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step3_tensorboard")
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    ## Print actor model answers during training
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')
    ## Testing
    parser.add_argument(
        '--enable_test_mode',
        action='store_true',
        help=
        'Enable a testing mode that terminates training based on args.test_stop_step'
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help=
        "Training non-overflow step at which to terminate training during testing."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.enable_tensorboard:
        print(
            f"Tensorboard logs going to: {args.tensorboard_path}/step3_tensorboard_logs"
        )
        writer = SummaryWriter(
            f"{args.tensorboard_path}/step3_tensorboard_logs")

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if args.actor_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.actor_lora_dim == 0:
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    return args


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    print('args.data_output_path',args.data_output_path)
    prompt_train_dataset, prompt_eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len,reload=False)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        prompt_eval_sampler = DistributedSampler(prompt_eval_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size)
    prompt_eval_dataloader = DataLoader(
        prompt_eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=args.per_device_generation_batch_size)
    
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_generation_batch_size / args.per_device_training_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, prompt_eval_dataloader, unsupervised_train_dataloader, num_total_iters

def evaluation(trainer, eval_dataloader, device, fast_step=10):
    trainer.eval()
    with torch.no_grad():
        actor_loss_sum = 0
        critic_loss_sum = 0
        for step, batch_prompt in enumerate(eval_dataloader):
            if torch.distributed.get_rank() == 0:
                print("^^^^^^^^^^^^^ Evaluation_step", step, "len(batch_prompt)", len(batch_prompt['prompt']), "^^^^^^^^^^^^^")
            batch_prompt = to_device(batch_prompt, device)
            
            out = trainer.generate_experience(batch_prompt['prompt'],
                                                batch_prompt['prompt_att_mask'],
                                                step, eval=True)
            if torch.distributed.get_rank() == 0:
                print("^^^^^^^^^^^^^ Evaluation_out", out['rewards'], "^^^^^^^^^^^^^")
            actor_loss, critic_loss, reward_score = trainer.eval_rlhf(out)
            if torch.distributed.get_rank() == 0:
                print("^^^^^^^^^^^^^ Evaluation_actor_loss", actor_loss, "^^^^^^^^^^^^^")
            actor_loss_sum += actor_loss
            critic_loss_sum += critic_loss
            reward_score_sum += reward_score
            if step == fast_step:  # For faster evaluation and debugging
                break
        actor_loss_mean = actor_loss_sum / (step + 1)
        critic_loss_mean = critic_loss_sum / (step + 1)
        reward_score_mean = reward_score_sum / (step + 1)

    return actor_loss_mean, critic_loss_mean, reward_score_mean

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        torch.cuda.set_device(local_rank)
        print(f"Global rank: {rank}, Local rank: {local_rank}, Current device: {torch.cuda.current_device()}")
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.tokenizer_model_name_or_path,
                                  fast_tokenizer=True)
    prompt_train_dataloader, prompt_eval_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)
    # print('prompt_train_dataloader',prompt_train_dataloader)
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)
    # if args.dtype == "fp16":
    #     print("++++++++++++++ Changing to fp16 ++++++++++++++++")
    #     rlhf_engine.actor = rlhf_engine.actor.half
    #     rlhf_engine.ref = rlhf_engine.ref.half
    #     rlhf_engine.critic = rlhf_engine.critic.half
    #     rlhf_engine.ref = rlhf_engine.ref.half
    # eval_dataset = torch.load(eval_file)
    # data_collator = DataCollatorReward()
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset,
    #                              collate_fn=data_collator,
    #                              sampler=eval_sampler,
    #                              batch_size=8)

    # Mixed Precision ZeRO++
    if args.enable_mixed_precision_lora:
        assert args.actor_lora_dim > 0, "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    args.end_of_conversation_token = "<|endoftext|>"

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)
    try:
        print("trainer",trainer.device)
        print("trainer.actor",trainer.actor_model.device)
        print("trainer.critic_model1",trainer.critic_model1.device)
    except:
        print('Cannot print device in trainer')
    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batches,
                                   args.per_device_training_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batches,
                                     args.per_device_training_batch_size)

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    non_overflow_step_count = 0
    max_reward = -100
    max_reward_val = -100
    save_max = False
    save_max_val = False
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            batch_prompt = to_device(batch_prompt, device)
            # if torch.distributed.get_rank() == 0:
            #     print("^^^^^^^^^^^^^ Step", step, "len(batch_prompt)", len(batch_prompt['prompt']), "^^^^^^^^^^^^^")
            #     print("^^^^^^^^^^^^^ Step", step, "batch_prompt[prompt].shape", batch_prompt['prompt'].shape, "^^^^^^^^^^^^^")
            #     print("^^^^^^^^^^^^^ Step", step, "batch_prompt[prompt_att_mask].shape", batch_prompt['prompt_att_mask'].shape, "^^^^^^^^^^^^^")
            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step)
            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_generation_batch_size])

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0
                average_reward_ori = 0
                average_reward_align = 0
                confidence = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()
                
                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()
                        average_reward_ori += exp_data["rewards_ori"].mean()
                        average_reward_align += exp_data["rewards_align"].mean()
                        confidence += exp_data["confidence"].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)
                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generation_batches  # it is an approximation, we did not include, e.g., rw forward time etc
                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | PPO Epoch: {ppo_ep+1} | Actor Loss: {actor_loss_sum/inner_iter} | Critic Loss: {critic_loss_sum/inner_iter} | Unsupervised Loss: {unsup_loss_sum/inner_iter}',
                    args.global_rank)
                print_throughput_step3(rlhf_engine.actor.model,
                                       rlhf_engine.critic, args, e2e_time,
                                       trainer.generate_time, training_time,
                                       args.global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                average_reward_ori = get_all_reduce_mean(average_reward_ori).item()
                average_reward_align = get_all_reduce_mean(average_reward_align).item()
                confidence = get_all_reduce_mean(confidence).item()
                print_rank_0(
                    f"Average reward score: {average_reward/inner_iter} Average reward score_ori: {average_reward_ori/inner_iter},Average confidence: {confidence/inner_iter}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

                if args.enable_tensorboard and torch.distributed.get_rank(
                ) == 0:
                    writer.add_scalar('train/reward', average_reward / inner_iter, global_step=step)
                    writer.add_scalar('train/reward_ori', average_reward_ori / inner_iter, global_step=step)
                    writer.add_scalar('train/reward_align_rewards', average_reward_ori / inner_iter, global_step=step)
                    writer.add_scalar('train/confidence', confidence / inner_iter, global_step=step)
                    writer.add_scalar('train/actor_loss',
                                      actor_loss,
                                      global_step=step)
                    # writer.add_scalar('train/actor_loss_sum',
                    #                   actor_loss_sum,
                    #                   global_step=step)
                    writer.add_scalar('train/critic_loss',
                                      critic_loss,
                                      global_step=step)
                    # writer.add_scalar('train/critic_loss_sum',
                    #                   critic_loss_sum,
                    #                   global_step=step)
                    writer.flush()

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            actor_overflow, critic_overflow = trainer.get_overflow()

            if not actor_overflow and not critic_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break
            if average_reward / inner_iter > max_reward:
                max_reward = average_reward / inner_iter
                save_max = True
            
            # if args.enable_tensorboard and torch.distributed.get_rank(
            #         ) == 0 and step % 100 == 0:
            #     actor_loss_mean, critic_loss_mean, reward_score = evaluation(trainer,prompt_eval_dataloader,device=device, fast_step=20)
            #     if reward_score > max_reward_val:
            #         max_reward_val = reward_score
            #         save_max_val = True
            #     writer.add_scalar('validation/reward',
            #                         reward_score,
            #                         global_step=step)
            #     writer.add_scalar('validation/actor_loss',
            #                         actor_loss_mean,
            #                         global_step=step)
            #     writer.add_scalar('validation/critic_loss',
            #                         critic_loss_mean,
            #                         global_step=step)
            #     writer.flush()
            if step % 5 == 0 or save_max or save_max_val:
                print_rank_0('saving model ...')
                rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
                rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
                if args.enable_ema:
                    rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                        rlhf_engine.actor_ema)

                if torch.distributed.get_rank() == 0:
                    save_hf_format(rlhf_engine.actor,
                                tokenizer,
                                args,
                                sub_folder='actor')
                    save_hf_format(rlhf_engine.critic,
                                tokenizer,
                                args,
                                sub_folder='critic')
                    if args.enable_ema:
                        save_hf_format(rlhf_engine.actor_ema,
                                    tokenizer,
                                    args,
                                    sub_folder='actor_ema')

                if args.actor_zero_stage == 3:
                    if save_max:
                        save_zero_three_model(rlhf_engine.actor,
                                            global_rank=args.global_rank,
                                            save_dir=os.path.join(
                                                args.output_dir, 'actor'),
                                            zero_stage=args.actor_zero_stage)
                    save_zero_three_model(rlhf_engine.actor,
                                        global_rank=args.global_rank,
                                        save_dir=os.path.join(
                                            args.output_dir + '/ep' + str(epoch+1)+ '/step'+ str(step), 'actor'),
                                        zero_stage=args.actor_zero_stage)
                    if args.enable_ema:
                        save_zero_three_model(rlhf_engine.actor_ema,
                                            global_rank=args.global_rank,
                                            save_dir=os.path.join(
                                                args.output_dir, 'actor_ema'),
                                            zero_stage=args.actor_zero_stage)
                if args.critic_zero_stage == 3:
                    if save_max:
                        save_zero_three_model(rlhf_engine.critic,
                                            global_rank=args.global_rank,
                                            save_dir=os.path.join(
                                                args.output_dir, 'critic'),
                                            zero_stage=args.critic_zero_stage)
                    save_zero_three_model(rlhf_engine.critic,
                                        global_rank=args.global_rank,
                                        save_dir=os.path.join(
                                            args.output_dir + '/ep' + str(epoch+1)+ '/step'+ str(step), 'critic'),
                                        zero_stage=args.critic_zero_stage)
                print_rank_0('saved model ...', args.global_rank)
                save_max = False
                save_max_val = False
        if args.enable_test_mode:
            break

    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)
        print_rank_0('saved model ...', args.global_rank)


if __name__ == "__main__":
    main()

""" 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
nohup deepspeed --master_port 33110 --include localhost:4,5,6,7 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/conf_acc32_13bchat_lr5-5_bs16/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 2 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --dtype fp32 \
   --alpha 0 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_fp32/acc32_13bchat_lr5-5_bs16 \
   --enable_tensorboard \
   --tensorboard_path tensorboard_fp32/acc32_13bchat_lr5-5_bs16 \
   &> log_fp32/acc32_13bchat_lr5-5_bs16.log 2>&1 &
   
   
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 12366 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/conf_acc32_13bchat_lr5-5_bs16/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-4 \
   --critic_learning_rate 1e-4 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_fp32/conf_acc32_13bchat_lr5-5_bs16 \
   --enable_tensorboard \
   --tensorboard_path tensorboard_fp32/conf_acc32_13bchat_lr5-5_bs16 \
   &> log_fp32/conf_acc32_13bchat_lr5-5_bs16.log 2>&1 &
"""




"""
nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 2 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --dtype fp16 \
   --alpha 10 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_rdiff_cdiff/13b_alpha10 \
   --enable_tensorboard \
   --tensorboard_path tensorboard_rdiff_cdiff/13b_alpha10 \
   &> log_rdiff_cdiff/13b_alpha10.log 2>&1 &
   
   
nohup deepspeed --master_port 33110 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/13b/noexample_real_alpaca_rand_lr1e-4/  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_13b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 2 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 12346 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --dtype fp16 \
   --alpha 0 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_fp16/13b_s1_ \
   --enable_tensorboard \
   --tensorboard_path tensorboard_fp16/13b_s1_ori \
   &> log_fp16/13b_s1_ori.log 2>&1 &
   

   
nohup deepspeed --master_port 33110 --include localhost:4,5,6,7 \
   main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 4 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --dtype fp32 \
   --alpha 0.6 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_/s1_alpha0.6 \
   --enable_tensorboard \
   --tensorboard_path tensorboard_fp32/s1_alpha0.6 \
   &> log_fp32/s1_alpha0.6.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 2 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --dtype fp32 \
   --alpha 10 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_rdiff_cdiff/alpha10 \
   --enable_tensorboard \
   --tensorboard_path tensorboard_rdiff_cdiff/alpha10 \
   &> log_rdiff_cdiff/alpha10.log 2>&1 &
   
nohup deepspeed --master_port 33110 --include localhost:4,5,6,7 \
   main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 4 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --dtype fp32 \
   --alpha 0.6 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint_rdiff_cdiff/alpha0.6 \
   --enable_tensorboard \
   --tensorboard_path tensorboard_rdiff_cdiff/alpha0.6 \
   &> log_rdiff_cdiff/alpha0.6.log 2>&1 &
   

   
   
   


# enable_mixed_precision_lora, fp16, lora 64
 我们方法: reward = reward_score_ori + alpha1 * -torch.abs(torch.sigmoid(reward_score_ori) - confidence_arr)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 128 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 8 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --fp16_enabled True \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1_r_alpha1_data \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1_r_alpha1_data \
   &> log_fp16/s1_r_alpha1_data.log 2>&1 &
"""



"""
 我们方法: reward = reward_score_ori + alpha1 * -torch.abs(torch.sigmoid(reward_score_ori) - confidence_arr)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 128 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 8 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1_r_alpha1_data \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1_r_alpha1_data \
   &> log/s1_r_alpha1_data.log 2>&1 &


# 我们方法: reward = eward_score_ori * torch.exp(-torch.abs(torch.sigmoid(reward_score_ori) - confidence_arr))
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33169 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 2 \
   --per_device_training_batch_size 2 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 16 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/7b_r_sig \
   --enable_tensorboard \
   --tensorboard_path tensorboard/7b_r_sig \
   &> log/7b_r_sig.log 2>&1 &
   

# Baseline: reward model只对 answer做评估 初始位置就是7b
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 35137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final  \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 8 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/ori_r_noconf \
   --enable_tensorboard \
   --tensorboard_path tensorboard/ori_r_noconf \
   &> log/ori_r_noconf3.log 2>&1 &
   
   

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1alpaca_rand_2help_hh512_r_ori \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1alpaca_rand_2help_hh512_rori \
   &> log/s1alpaca_rand_2help_hh512_rori.log 2>&1 &


# reward model只对 answer做评估
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33137 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 768 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps  \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1alpaca_rand_2help_hh512_onlyans \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1alpaca_rand_2help_hh512_onlyans \
   &> log/s1alpaca_rand_2help_hh512_onlyans.log 2>&1 &
   
# 加上validation
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_real_alpaca_rand_lr1e-4/final \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --data_output_path /mnt/data/taoshuchang.tsc/datatmp/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1alpaca_rand_2help_hh512 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1alpaca_rand_2help_hh512 \
   &> log/s1alpaca_rand_2help_hh512.log 2>&1 &
# 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 13102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_alpaca_lr1e-4/final \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 16 \
   --critic_lora_dim 16 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1alpaca_2help_hh512 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1alpaca_2help_hh512 \
   &> log/s1alpaca_2help_hh512_rand.log 2>&1 &
   
   
CUDA_VISIBLE_DEVICES=2,3    nohup deepspeed --master_port 13102 main.py \
   --data_path file:///mnt/data/taoshuchang.tsc/datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/checkpoint/confidence/noexample_alpaca_lr1e-4/final \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/help/ep3/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type constant \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/s1alpaca_2help \
   --enable_tensorboard \
   --tensorboard_path tensorboard/s1alpaca_2help \
   &> log/s1alpaca_2help_hh.log 2>&1 &

nohup deepspeed --master_port 12366 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/output/llama7blora \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/acc32_13bchat_lr5-5_bs16 \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-4 \
   --critic_learning_rate 1e-4 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/acc32_7bppo_conf_full_qacseq \
   --enable_tensorboard \
   --tensorboard_path tensorboard/acc32_7bppo_conf_full_qacseq \
   &> log/acc32_7bppo_conf_full_qacseq.log 2>&1 &
"""



'''
nohup deepspeed --master_port 12366 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/output/llama7blora \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/conf_acc32_13bchat_lr5-5_bs16_60k/ \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-4 \
   --critic_learning_rate 1e-4 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/conf_acc32_13bchat_lr5-5_bs16 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/conf_acc32_13bchat_lr5-5_bs16 \
   &> log/conf_acc32_13bchat_lr5-5_bs16.log 2>&1 &



nohup deepspeed --master_port 12366 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static_conf \
   --data_split 0,0,10 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/output/llama7blora \
   --tokenizer_model_name_or_path /mnt/data/taoshuchang.tsc/model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/acc32_13bchat_lr5-5_bs16 \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-4 \
   --critic_learning_rate 1e-4 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir checkpoint/acc32_7bppo_conf_full \
   --enable_tensorboard \
   --tensorboard_path tensorboard/acc16_7bppo_conf_full \
   &> log/acc32_7bppo_conf_full.log 2>&1 &





nohup deepspeed --master_port 12346 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/output/llama7blora \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/checkpoint/acc16_13bchat_lr-4 \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-4 \
   --critic_learning_rate 1e-4 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir output/step3_7bppo_lract-4_lrcri-4 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/step3_7bppo_lract-4_lrcri-4 \
   &> log/step3_7bppo_lract-4_lrcri-4.log 2>&1 &







nohup deepspeed --master_port 12346 main.py \
   --data_path /mnt/data/taoshuchang.tsc/datasets/Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step1_supervised_finetuning/output/llama7blora \
   --critic_model_name_or_path /mnt/data/taoshuchang.tsc/Code_released/DeepSpeedExamples/applications/DeepSpeed-Chat/SA_rlhf/step2_reward_model_finetuning/output/step2_llama_7b_epoch1_lr9.65e-6 \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-4 \
   --critic_learning_rate 1e-4 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir output/step3_7bppo_lract-4_lrcri-4 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/step3_7bppo_lract-4_lrcri-4 \
   &> log/step3_7bppo_lract-4_lrcri-4.log 2>&1 &
'''