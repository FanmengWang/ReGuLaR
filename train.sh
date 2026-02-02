# (Optional) Following Coconut and CoLaR, we also initialize ReGuLaR with weights from CoT-SFT to accelerate training.
dataset_name=$1
python run.py \
    --devices=all \
    --model=ReGuLaR \
    --dataset=qsa \
    --do_test \
    --log_suffix=train_ReGuLaR \
    --load_ckpt_path=./models/CoLaR/logs/cot/qsa-gsm/llama-1b-cot/checkpoints/epoch7__step12056__monitor0.560.ckpt \
    dataset_name=$dataset_name \
    precompute_name=./datasets/${dataset_name}/precomputed_visual_representations \
    model_id=Llama-3.2-1B-Instruct