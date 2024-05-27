BASE_MODEL="google/gemma-7b"
# BASE_MODEL="mistralai/Mistral-7B-v0.1"
LORA_RANK=4
NUM_VECTORS=2048
OUTPUT="output"
mkdir $OUTPUT
MERGED_PATH="${OUTPUT}_merged"
mkdir $MERGED_PATH


python intruction_tuning_vblora.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --lora_r $LORA_RANK \
    --num_vectors $NUM_VECTORS \
    --vector_length 256 \
    --save_only_topk_weights True \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]"\
    --dataset_field query response \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 4e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --report_to tensorboard

FT_PATH=$(find $OUTPUT -type d -path "*/ft" | grep $BASE_MODEL | grep rank_$LORA_RANK)
python -m utils.merge_adapter_to_base_model --base_mode $BASE_MODEL --adapter "$FT_PATH" --output_path "$MERGED_PATH"

python instruction_tuning_eval/gsm8k_eval.py --model "$MERGED_PATH"
python instruction_tuning_eval/MATH_eval.py --model "$MERGED_PATH"