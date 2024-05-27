export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/large_qnli_qv"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-large \
--task_name qnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 2e-3 \
--learning_rate_vector_bank 1e-3 \
--learning_rate_logits 1e-2 \
--num_train_epochs 20 \
--output_dir $output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy no \
--warmup_ratio 0.06 \
--vb_module value,query \
--rank 4 \
--topk 2 \
--num_vectors 90 \
--vector_length 256 \
--seed 1 \
--weight_decay 0.1 