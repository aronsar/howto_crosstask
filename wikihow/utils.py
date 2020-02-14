import os
import sys
import pickle
import re
import scipy
import numpy as np
import json
from allennlp.commands.elmo import ElmoEmbedder
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer


# MISSING_STEPS = defaultdict(dict)
# elmo = ElmoEmbedder()

def load_task_steps():
    task_primary_path = "/data/aronsar/crosstask/crosstask_release/elmo_tasks_primary.txt"
    num_tasks = 18 # number of primary tasks
    with open(task_primary_path, "rb") as f:
        # lines 0, 6, 12, 18, ... have the task id
        # lines 4, 10, 16, 22, ... have the task_steps
        task_steps = {}
        task_step_nouns = {}
        split_task_steps = {}
        all_lines = f.readlines()
        for i in range(num_tasks):
            task_id = all_lines[i*7].decode('UTF-8').rstrip()
            task_steps[task_id] = all_lines[i*7+4].decode('UTF-8').rstrip().split(",")
            task_step_nouns[task_id] = all_lines[i*7+5].decode('UTF-8').rstrip().split(",")
            split_task_steps[task_id] = [step.split(" ") for step in task_steps[task_id]]

    return task_steps, task_step_nouns, split_task_steps

def check_node_for(json_node, unwanted_sections):
    for section_name in unwanted_sections:
        if "children" in json_node \
            and len(json_node["children"]) > 0 \
            and "text" in json_node["children"][0] \
            and json_node["children"][0]["text"] == section_name:

            return True

class Error(Exception):
    pass

class ReachedCommunityQA(Error):
    pass

def extract_article_sentences(json_node, article_sentences):
    if check_node_for(json_node, ["Community Q&A"]):
        raise ReachedCommunityQA # nothing interesting after community Q&A
    
    unwanted_sections = ["Community Q&A", "Related wikiHows", "References", "Warnings", "Ingredients", "Tips", "Advice", "Caution"]
    if check_node_for(json_node, unwanted_sections):
        return

    elif "children" in json_node:
        for subnode in json_node["children"]:
            extract_article_sentences(subnode, article_sentences)

    elif "text" in json_node:
        for sentence in re.findall(r'([A-Z][^\.!?]*[\.!?])', json_node["text"]):
            stripped_sentence = re.sub(r'[.,?!@#$%^&*]','',sentence)
            split_sentence = str.split(stripped_sentence)
            article_sentences.append(split_sentence)

def sentence_steps(sentence, nouns, task_steps):
    # returns a set of task steps that occur in sentence
    steps_found = set()
    for noun, task_step in zip(nouns, task_steps):
        if noun in sentence:
            # sentence_vec = elmo.embed_sentence(sentence)
            # split_task_step = task_step.split()
            # task_step_vec = elmo.embed_sentence(split_task_step)
            # n_i = split_task_step.index(noun)
            # s_i = sentence.index(noun)
            # dist = scipy.spatial.distance.cosine(task_step_vec[2][n_i], sentence_vec[2][s_i])            
            # if dist < .9:
            #     sentence[s_i] += ">>>>>>>>"# + str(dist)[0:5]
            steps_found.add(task_step)

    # print(sentence)
    return steps_found


def sentence_steps(sentence, syns_list, task_steps):
    # returns a set of task steps that occur in sentence
    # Case folding
    porter = PorterStemmer()
    for i, word in enumerate(sentence):
        sentence[i] = porter.stem(word.casefold())
    
    steps_found = set()
    for syns, task_step in zip(syns_list, task_steps):
        for syn in syns:
            if syn in sentence:
                steps_found.add(task_step)
                break

    return steps_found

def get_synonyms(word):
    porter = PorterStemmer()
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            name = l.name()
            name = porter.stem(name.casefold())
            synonyms.append(name)

    return set(synonyms)

def normalize(A):
    lengths = (A**2).sum(axis=1, keepdims=True)**.5
    return A/lengths