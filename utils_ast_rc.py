from __future__ import absolute_import, division, print_function
from tree_sitter import Language, Parser
from utils import (remove_comments_and_docstrings, tree_to_token_index,
                   index_to_code_token, tree_to_variable_index)

import argparse
import glob
import logging
import os
import pickle
import random
import shutil
import json
import numpy as np
import multiprocessing
from tqdm import tqdm, trange
import pandas as pd
import re

cpu_cont = 16
logger = logging.getLogger(__name__)

"""
from nltk import word_tokenize
at the first run: nltk.download('punkt')
import nltk
nltk.download('punkt')
"""


def is_number(s):
    try:
        float(s)  # for int, long and float
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False

    return True


def traverse_tree(tree):
    cursor = tree.walk()

    reached_root = False
    while not reached_root:
        yield cursor.node

        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            if cursor.goto_next_sibling():
                retracing = False


# remove comments, tokenize code and extract ast trees
def extract_tree(code, parser, lang):
    """65.02% (f1) when using this function (removing comments)"""
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    """"""
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser.parse(bytes(code, 'utf8'))
        tokens_index = tree_to_token_index(tree.root_node)
        code_split = code.split('\n')
    except:
        tree = None
        code_split = None
        tokens_index = None
    return tree, code_split, tokens_index


# remove comments, tokenize code and extract ast trees
def extract_tree_afterstr(code, parser, lang):
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser.parse(bytes(code, 'utf8'))
    except:
        tree = None
    return tree


def flatten_AST(tree_root):
    seq = []
    name = tree_root.type
    if len(tree_root.children) == 0:
        seq.append(name)
    else:
        seq.append(name + " :: left")
        for child in tree_root.children:
            seq.extend(flatten_AST(child))
        seq.append(name + " :: right")

    return seq


def flatten_AST_details(tree_root, code_split):
    seq = []
    name = tree_root.type
    if len(tree_root.children) == 0:
        """
        seq.append(name)
        """
        if not ((tree_root.start_point[0] == tree_root.end_point[0]) and (tree_root.start_point[1] == tree_root.end_point[1])):
            i_str = index_to_code_token((tree_root.start_point, tree_root.end_point), code_split)
            if i_str.__contains__('"'):
                node_str = 'str'
            else:
                node_str = i_str
            seq.append(node_str)
    else:
        seq.append(name + " :: left")
        for child in tree_root.children:
            seq.extend(flatten_AST_details(child, code_split))
        seq.append(name + " :: right")

    return seq


def convert_examples_to_features_ast(code):
    lang = 'c'
    language = Language('parser_ved/my-languages.so', lang)
    parser = Parser()
    parser.set_language(language)

    tree, code_split, tokens_index = extract_tree(code, parser, lang)
    """
    flatten_ast = flatten_AST(tree.root_node)
    """

    ast_success = True
    if tree is not None:
        flatten_ast_details = flatten_AST_details(tree.root_node, code_split)
        flatten_ast = flatten_AST(tree.root_node)
        """
        tree_as = extract_tree_afterstr(func_rename, parser, lang)
        if tree_as is not None:
            flatten_ast = flatten_AST(tree_as.root_node)
        else:
            flatten_ast = None
        """
    else:
        flatten_ast_details = None
        ast_success = False
        flatten_ast = None

    return ast_success, code_split, flatten_ast, flatten_ast_details
