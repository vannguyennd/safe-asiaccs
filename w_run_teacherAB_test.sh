#!/bin/bash
model=gvd
ast_type=detail

block_size=512
ep=10
token=graphcodebert-base
lr=5e-4
ws=5
hs=256

printf "\n"
logp="./teachersAB/teacherAB_ast/${model}_epochs${ep}_lr${lr}_blsize${block_size}_wsize${ws}_tempt${hs}"
if ! (( -d $logp ))
then
	mkdir $logp
fi
python w_llms_teachersAB_ast.py \
	--output_dir=$logp \
	--model_type=roberta \
	--tokenizer_name=microsoft/$token \
	--model_name_or_path=microsoft/$token \
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
	--model=$model \
	--ast_type=$ast_type \
	--window_size $ws \
	--hidden_size_ta $hs
printf "\n\n=======\n\n"
