## SAFE
This repository contains the source code and data samples for reproducing the experiments of our SAFE approach in our paper "SAFE: A Novel Approach For Software Vulnerability Detection from Enhancing The Capability of Large Language Models", published as a conference paper at the ACM ASIA Conference on Computer and Communications Security (ACM ASIACCS), 2025.

## Dependencies
We implement our SAFE approach in Python using Pytorch (version 2.0), Python (version 3.9), and Transformers (version 4.3). Other required packages are tree-sitter, scikit-learn, numpy, scipy, and pickle.

We use tree-sitter to obtain Abstract Syntax Trees (ASTs) and Data Flow Graphs (DFGs) from the source code data used for the teacherB model. After installing tree-sitter with *pip install tree-sitter*, you need to run the *build.py* file located in the parser_ved folder. This step generates the *my-languages.so* file, which is essential for obtaining ASTs and DFGs from the source code data.

## Datasets and Pre-trained models
The *dataset* folder contains source code samples (of the Devign dataset) used in the training, validation, and testing (inference) phases. The *studentS* folder houses the pre-trained model of our SAFE approach when using RoBERTa as the backbone for the Student-S model with ASTs for the Teacher-B model. The *teachersAB* folder contains pre-trained models of the teacherA and teacherB models. Please download these folders including all of their files at (https://drive.google.com/drive/folders/1TJpm5iF2BnDdkrkBfEReJSDtbNAjS8SG?usp=sharing)

## Running SAFE (the training, validation, and testing processes)
To train our SAFE approach, including the training, validation, and testing processes, run the *w_run_studentS.sh* file. This script allows you to easily set the values for the hyperparameters to train the model. Particularly, in your terminal, execute the following command: *bash w_run_studentS.sh*.

```python
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
	--do_eval \
	--do_test \
	--do_train \
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
	--teacher_beta $ta | tee $logp/training_log.txt
printf "\n\n=======\n\n"
```

Note that: The above settings configure the model to perform training, validation, and testing phases sequentially. To enable or disable a specific phase, include or remove the corresponding flags in the script: (i) *do_train* to enable training, (ii) *do_eval* to enable validation, and (iii) *do_test* to enable testing.

## Citation

If you reference our paper (and/or) use our source code samples in your work, please kindly cite our paper.

@inproceedings{nguyen-safe-asiaccs2025,<br/>
      author={Van Nguyen and Surya Nepal and Xingliang Yuan and Tingmin Wu and Carsten Rudolph},<br/>
      title={ SAFE: A Novel Approach For Software Vulnerability Detection from Enhancing The Capability of Large Language Models},<br/>
      booktitle={The 20th ACM ASIA Conference on Computer and Communications Security (ACM ASIACCS)},<br/>
      year={2025}<br/>
}
