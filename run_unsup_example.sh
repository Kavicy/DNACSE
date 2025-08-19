#!/bin/bash



NUM_GPU=2

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path bertmodel \
    --train_file data/pretrain.csv \
    --val_file data/zsc_dna_bin.csv
    --output_dir result/my-sup-dnacse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model AMI \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --global_noise 0.09\
    --local_noise 0.10\
    --lambda 0.2\
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
