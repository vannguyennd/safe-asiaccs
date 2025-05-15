from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

from utils_ast_rc import convert_examples_to_features_ast

from sklearn import metrics as mt
from scipy.special import softmax
from tqdm import tqdm, trange
import multiprocessing
from w_models_AB import *
from w_model_S_withast_sab import *
import warnings
from parser_ved import remove_comments_and_docstrings 
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)
cpu_cont = multiprocessing.cpu_count()

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens_sb,
                 input_ids_sb,
                 att_masks_sb,
                 idx,
                 label,
                 input_tokens_a,
                 input_ids_a,
                 att_masks_a,
    ):
        self.input_tokens_sb = input_tokens_sb
        self.input_ids_sb = input_ids_sb
        self.att_masks_sb = att_masks_sb
        self.idx = str(idx)
        self.label = label
        self.input_tokens_a = input_tokens_a
        self.input_ids_a = input_ids_a
        self.att_masks_a = att_masks_a


def convert_examples_to_features(js, tokenizer, args):
    # for student and teacherB models
    if args.recomments is True:
        func_sb = remove_comments_and_docstrings(js['func'], 'c')
        code_sb =' '.join(func_sb.split())
    else:
        code_sb =' '.join(js['func'].split())

    code_tokens_sb = tokenizer.tokenize(code_sb)[:args.block_size-4]
    source_tokens_sb = [tokenizer.cls_token] + code_tokens_sb + [tokenizer.dia_token] + [tokenizer.dib_token] + [tokenizer.sep_token]
    source_ids_sb =  tokenizer.convert_tokens_to_ids(source_tokens_sb)
    padding_length_sb = args.block_size - len(source_ids_sb)
    
    source_ids_sb += [tokenizer.pad_token_id]*padding_length_sb
    att_masks_sb = (np.array(source_ids_sb).__ne__(tokenizer.pad_token_id)*1).tolist()

    # for teacherA model
    _,_, flatten_ast, flatten_ast_details = convert_examples_to_features_ast(js['func'])
    if args.ast_type == "abstract":
        flatten_ast_str = ' '.join([str(item) for item in flatten_ast])
        code_a =' '.join(flatten_ast_str.split())
    else:
        flatten_ast_details_str = ' '.join([str(item) for item in flatten_ast_details])
        code_a =' '.join(flatten_ast_details_str.split())

    code_tokens_a = tokenizer.tokenize(code_a)[:args.block_size-4]
    source_tokens_a =[tokenizer.cls_token] + code_tokens_a + [tokenizer.dia_token] + [tokenizer.dib_token] + [tokenizer.sep_token]
    source_ids_a =  tokenizer.convert_tokens_to_ids(source_tokens_a)
    padding_length_a = args.block_size - len(source_ids_a)
    
    source_ids_a += [tokenizer.pad_token_id]*padding_length_a
    att_masks_a = (np.array(source_ids_a).__ne__(tokenizer.pad_token_id)*1).tolist()

    return InputFeatures(source_tokens_sb, source_ids_sb, att_masks_sb, js['idx'], js['target'], source_tokens_a, source_ids_a, att_masks_a)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            logger.info("*** Total Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Sample ***")
                    logger.info("Total sample".format(idx))
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens_sb: {}".format([x.replace('\u0120','_') for x in example.input_tokens_sb]))
                    logger.info("input_ids_sb: {}".format(' '.join(map(str, example.input_ids_sb))))
                    logger.info("input_tokens_a: {}".format([x.replace('\u0120','_') for x in example.input_tokens_a]))
                    logger.info("input_ids_a: {}".format(' '.join(map(str, example.input_ids_a))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids_sb), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].att_masks_sb), torch.tensor(self.examples[i].input_ids_a), \
            torch.tensor(self.examples[i].att_masks_a)
    

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    """
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    """

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        # bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        # for step, batch in enumerate(bar):
        for step, batch in enumerate(train_dataloader):
            inputs_sb = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            masks_sb = batch[2].to(args.device)
            inputs_a = batch[3].to(args.device)
            masks_a = batch[4].to(args.device)
            model.train()
            loss, logits, _, _ = model(inputs_sb, masks_sb, labels, inputs_a, masks_a)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            """
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            """

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)

            # bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            # logger.info("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate
                        # when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint

                    if results['eval_f1_binary'] > best_acc:
                        best_acc = results['eval_f1_binary']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
        avg_loss = round(train_loss / tr_num, 5)
        logger.info("epoch {} loss {}".format(idx, avg_loss))


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs_sb = batch[0].to(args.device)
        label = batch[1].to(args.device)
        masks_sb = batch[2].to(args.device) 
        inputs_a = batch[3].to(args.device)
        masks_a = batch[4].to(args.device)
        with torch.no_grad():
            lm_loss, logit, _, _ = model(inputs_sb, masks_sb, label, inputs_a, masks_a)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    
    eval_acc = np.mean(labels == preds)
    eval_f1_binary = mt.f1_score(y_true=labels, y_pred=preds, average='binary')
    
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_f1_binary": round(eval_f1_binary, 4),
    }
    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
    for batch in eval_dataloader:
        inputs_sb = batch[0].to(args.device)
        label = batch[1].to(args.device)
        masks_sb = batch[2].to(args.device)
        inputs_a = batch[3].to(args.device)
        masks_a = batch[4].to(args.device)
        with torch.no_grad():
            logit, _, _ = model(inputs_sb, masks_sb, None, inputs_a, masks_a)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    
    preds = logits[:, 0] > 0.5

    # compute the main metrics
    src_test_pre = mt.precision_score(y_true=labels, y_pred=preds)
    src_test_f1_binary = mt.f1_score(y_true=labels, y_pred=preds, average='binary')
    src_test_re = mt.recall_score(y_true=labels, y_pred=preds)
    
    """additional metrics
    test_acc = np.mean(labels == preds)
    src_test_acc = mt.accuracy_score(y_true=labels, y_pred=preds)
    src_test_f1_weighted = mt.f1_score(y_true=labels, y_pred=preds, average='weighted')
    tn, fp, fn, tp = mt.confusion_matrix(y_true=labels, y_pred=preds).ravel()
    if (fp + tn) == 0:
        fpr = -1.0
    else:
        fpr = float(fp) / (fp + tn)

    if (tp + fn) == 0:
        fnr = -1.0
    else:
        fnr = float(fn) / (tp + fn)
    """
        
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        f.write("mt_test_recall: " + str(round(src_test_re, 4)) + "\n")
        f.write("mt_test_precision: " + str(round(src_test_pre, 4)) + "\n")
        f.write("mt_test_f1_binary: " + str(round(src_test_f1_binary, 4)) + "\n")
        
    result = {
        "src_test_f1_binary": round(src_test_f1_binary, 4),
    }
    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default="./dataset/train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model "
                             "predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--eval_data_file", default="./dataset/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="./dataset/test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 "
                             "(instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, '
                             'does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and "
                             "ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # default="students"
    parser.add_argument("--model", default="students", type=str, help="")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument("--num_classes_s", default=1, type=int,
                        help="num classes.")
    parser.add_argument("--gnn", default="GCN", type=str, help="")

    parser.add_argument("--format", default="uni", type=str,
                        help="idx for index-focused method, uni for unique token-focused method")
    parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percet of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percent of training sample")

    parser.add_argument("--tau_tempt", default=0.07, type=float, help="tau, temperature.")
    parser.add_argument("--trade_off_cl", default=3e-3, type=float, help="trade-off for the contrastive learning term.")
    parser.add_argument("--gamma", default=2.0, type=float, help="gamma used in focal loss")
    parser.add_argument("--trade_off_fl", default=-1.0, type=float, help="trade-off for the focal loss.")
    parser.add_argument("--alpha", default=0.5, type=float, help="alpha used for focal loss")

    parser.add_argument("--hidden_size_projection", default=256, type=int, help="hidden size projection used for contrastive learning")
    parser.add_argument("--roberta_embedding", default=768, type=int, help="roberta embedding size")
    parser.add_argument("--reduction", default="mean", type=str, help="reduction used in focal loss")

    """
    for distillations
    """
    parser.add_argument("--path_teacherAB", default="", type=str, help="The path of the trained teacher B model")
    parser.add_argument("--path_teacherBA", default="", type=str, help="The path of the trained teacher A model")
    parser.add_argument("--distill_tempt", default=1.0, type=float, help="Distiallation's temperature")
    parser.add_argument("--student_alpha", default=0.5, type=float, help="The trade-off hyperparameter for the student")
    parser.add_argument("--teacher_beta", default=0.5, type=float, help="The trade-off hyperparameter for teacher A")
    parser.add_argument("--hidden_size_ta", default=256, type=int, help="hidden size.")

    parser.add_argument("--taugumbel", default=0.5, type=int, help="tau value used in gumbel softmax")
    parser.add_argument("--hardgumbel", default=True, type=bool, help="If hard=True, the returned samples will be one-hot, otherwise they will be probability distributions that sum to 1 across dim.")

    parser.add_argument("--recomments", default=False, type=bool, help="To decide if we want to remove comments from source code or not")
    parser.add_argument("--ast_type", default="detail", type=str, help="abstract or detail")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    """
    adding tokens regarding the distillation tokens A and B
    """
    tokenizer.add_tokens(["<dia>"])
    tokenizer.dia_token_id = tokenizer.encode("<dia>", add_special_tokens=False)[0]
    tokenizer.dia_token = "<dia>"

    tokenizer.add_tokens(["<dib>"])
    tokenizer.dib_token_id = tokenizer.encode("<dib>", add_special_tokens=False)[0]
    tokenizer.dib_token = "<dib>"

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
        # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)
    
    """
    resize the token embeddings of the model with the appended tokenizer
    """
    model.resize_token_embeddings(len(tokenizer))

    """
    load trained teachers B (gvd) and A (textcnn)
    """
    teacherAB = GNNGVD(model, config, tokenizer, args)
    teacherBA = TextCNNVed(model)

    teacherAB.load_state_dict(torch.load(args.path_teacherAB))
    teacherBA.load_state_dict(torch.load(args.path_teacherBA))

    # model oriupdate
    if args.model == "students":
        model = ModelCheck(model, config, tokenizer, args, teacherAB, teacherBA)
    else:
        print("Please set the model argument to 'students'!!!")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training
        # download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    """Checking"""
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training process the dataset,
            # and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)
    """Checking"""

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test_result = test(args, model, tokenizer)

        logger.info("***** Test results *****")
        for key in sorted(test_result.keys()):
            logger.info("  %s = %s", key, str(round(test_result[key], 4)))

    return results


if __name__ == "__main__":
    main()
