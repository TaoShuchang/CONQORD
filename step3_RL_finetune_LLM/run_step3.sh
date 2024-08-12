

# --------------- Step3 RL finetune LM -------------------------


# The current directory is ./step3_RL_finetune_LLM/ (=CONQORD/step3_RL_finetune_LLM/)
# Step 3.0: Create log, checkpoint, tensorboard folders  
mkdir -p log
mkdir -p checkpoint
mkdir -p tensorboard

# Step 3.1: Prepared the actor model in step1 and the critic model in step2.


# Step 3.2: Run main.py in step3
export CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port 33001 main.py \
   --data_path ../datasets/Anthropic/hh-rlhf/helpful-base_conf_half_sharp \
   --data_split 0,0,10 \
   --actor_model_name_or_path ../step1_supervised_finetuning_LM/checkpoint/step1/final \
   --tokenizer_model_name_or_path ../model_pth/llama2_all_hf/llama2_hf_7b/ \
   --critic_model_name_or_path ../step2_reward_model_finetuning/checkpoint/step2/ep3/ \
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
   --output_dir checkpoint/step3 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/step3 \
   &> log/step3.log 2>&1 &
