#!/bin/bash
model=students
block_size=512
path_teacherAB=teachersAB/teacherAB_ast/gvd_epochs10_lr5e-4_blsize512_wsize5_tempt256/checkpoint-best-acc/model.bin
path_teacherBA=teachersAB/teacherBA/textcnn_epochs10_lr1e-4_blsize512/checkpoint-best-acc/model.bin

ep=10
lr=2e-5
sa=0.7
ta=0.3

printf "\n"
logp="./studentS/${model}_epochs${ep}_lr${lr}_blsize${block_size}_sa${sa}_ta${ta}"
if ! (( -d $logp ))
then
	mkdir $logp
fi
python w_llms_studentS_withast.py \
	--output_dir=$logp \
	--model_type=roberta \
	--tokenizer_name=microsoft/graphcodebert-base \
	--model_name_or_path=microsoft/graphcodebert-base \
	--do_test \
	--train_data_file=./dataset/train.jsonl \
	--eval_data_file=./dataset/valid.jsonl \
	--test_data_file=./dataset/test.jsonl \
	--block_size $block_size \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--max_grad_norm 1.0 \
	--evaluate_during_training \
	--learning_rate $lr \
	--epoch $ep \
	--seed 123456 2>&1 \
	--path_teacherAB=$path_teacherAB \
	--hidden_size_ta 256 \
	--path_teacherBA=$path_teacherBA \
	--model=$model \
	--student_alpha $sa \
	--teacher_beta $ta
printf "\n\n=======\n\n"
