nproc_per_node=3

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=1,2,3 \
swift sft \
    --model_type qwen2-1_5b-instruct \
    --model_id_or_path /path/to/your/Qwen2-1___5B-Instruct \
    --dataset /path/to/your/recover/train/data/recover_train_chn_eng_swift.jsonl \
    --sft_type full \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir ./exp/Recover/output_qwen2_1_5B_recover_8192_chn_eng_v6 \
    --ddp_backend nccl \
    --train_dataset_sample -1 \
    --num_train_epochs 5 \
    --max_length 8192 \
    --gradient_checkpointing true \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --use_flash_attn true \
    --deepspeed 'default-zero3' \
    --save_only_model true \
