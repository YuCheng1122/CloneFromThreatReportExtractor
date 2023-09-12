#!python
# -*- coding: utf-8 -*-

import os
import signal
import argparse
import re
import nltk
import spacy

from nlp_extractor.tokenizer import ThreatTokenizer
from preprocessings import PreProcessor
from role_generator import RoleGenerator
from graph_generator import GraphGenerator
from data_loader.pattern_loader import load_lists
from project_config import SEC_PATTERNS_FILE_PATH


def load_configurations(file_path):
    titles_list = load_lists(file_path)['MS_TITLES']
    titles_list = titles_list.replace("'", "").strip('][').split(', ')
    main_verbs = load_lists(file_path)['verbs']
    main_verbs = main_verbs.replace("'", "").strip('][').split(', ')
    return titles_list, main_verbs


def read_input_file(file_path):
    with open(file_path, encoding='iso-8859-1') as f:
        txt = f.readlines()
        txt = " ".join(txt)
        txt = txt.replace('\n', ' ')
    return txt


def main(args):
    print(nltk.__version__)
    print(args)

    nlp = spacy.load("en_core_web_lg")
    nltk.download('punkt')

    titles_list, main_verbs = load_configurations(SEC_PATTERNS_FILE_PATH)
    threat_tokenizer = ThreatTokenizer(nlp, main_verbs, titles_list)

    if not args.input_file:
        raise ValueError("Input file is required.")

    txt = read_input_file(args.input_file)

    pre_processor = PreProcessor(nlp)
    role_gen = RoleGenerator(nlp, main_verbs)
    graph_gen = GraphGenerator()

    # Tokenization and Preprocessing
    # Assuming you have a method called 'tokenize' in your ThreatTokenizer class
    txt = threat_tokenizer.tokenize(txt)  
    txt = pre_processor.preprocess(txt)  # Replace with your actual method

    # Role Generation
    roles = role_gen.generate_roles(txt)  # Replace with your actual method
    print(type(roles))
    # Graph Generation
    graph_gen.graph_builder(roles, args.gname)
    print(type(roles))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asterisk', type=str, default='true')
    parser.add_argument('--crf', type=str, default='true')
    parser.add_argument('--rmdup', type=str, default='true')
    parser.add_argument('--elip', type=str, default='false')
    parser.add_argument('--gname', type=str, default='graph')
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()

    main(args)


