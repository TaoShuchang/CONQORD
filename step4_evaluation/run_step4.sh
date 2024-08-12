

# --------------- Step4 Evaluating CONQORD -------------------------

# The current directory is ./step4_evaluation/ (=CONQORD/step4_evaluation/)
# Step 4.1: CONQORD Inference
mkdir -p ./rating/truthful_qa/
mkdir -p ./log/truthful_qa/
nohup python -u test_conqord.py --data_name truthful_qa \
    --mode llama2_7b \
    --suffix conqord_llama2_nq \
    --path ../step3_rlhf_finetuning/checkpoint/step3_RL_finetune_LLM/ep1/step30/actor \
    --gpu 2 > ./log/truthful_qa/conqord_llama2.log 2>&1 &


# Step 4.2: Evaluating performance for CONQORD
nohup python -u gpt_evaluation.py --data_name truthful_qa --suffix conqord_llama2 --mode llama2_7b --gpu -1 > ./log/truthful_qa/conqord_llama2.log 2>&1 &

