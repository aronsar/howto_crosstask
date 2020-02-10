import os
import sys
import pickle
import re
import scipy
import numpy as np
import json
from allennlp.commands.elmo import ElmoEmbedder
from collections import defaultdict

TASKS_PRIMARY_PATH = "/data/aronsar/crosstask/crosstask_release/elmo_tasks_primary.txt"
NUM_TASKS = 18 # number of primary tasks
TASK_STEPS = {}
TASK_STEP_NOUNS = {}
SPLIT_TASK_STEPS = {}
MISSING_STEPS = defaultdict(dict)
elmo = ElmoEmbedder()


def load_task_steps():
    with open(TASKS_PRIMARY_PATH, "rb") as f:
        # lines 0, 6, 12, 18, ... have the task id
        # lines 4, 10, 16, 22, ... have the task_steps
        all_lines = f.readlines()
        for i in range(NUM_TASKS):
            task_id = all_lines[i*7].decode('UTF-8').rstrip()
            TASK_STEPS[task_id] = all_lines[i*7+4].decode('UTF-8').rstrip().split(",")
            TASK_STEP_NOUNS[task_id] = all_lines[i*7+5].decode('UTF-8').rstrip().split(",")
            SPLIT_TASK_STEPS[task_id] = [step.split(" ") for step in TASK_STEPS[task_id]]

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

def sentence_steps(sentence, task):
    # returns a set of task steps that occur in sentence
    steps_found = set()
    for noun, task_step in zip(TASK_STEP_NOUNS[task], TASK_STEPS[task]):
        if noun in sentence:
            sentence_vec = elmo.embed_sentence(sentence)
            split_task_step = task_step.split()
            task_step_vec = elmo.embed_sentence(split_task_step)
            n_i = split_task_step.index(noun)
            s_i = sentence.index(noun)
            dist = scipy.spatial.distance.cosine(task_step_vec[2][n_i], sentence_vec[2][s_i])            
            if dist < .9:
                sentence[s_i] += ">>>>>>>>" + str(dist)[0:5]
                steps_found.add(task_step)

    print(sentence)
    return steps_found


def normalize(A):
    lengths = (A**2).sum(axis=1, keepdims=True)**.5
    return A/lengths

if __name__ == '__main__':
    with open("YT_embeds.pkl", "rb") as f:
        YT_embeds = pickle.load(f)

    with open("yttitles", "r") as f:
        yttitles = [line.rstrip() for line in f]

    with open("whtitles", "r") as f:
        whtitles = [line.rstrip() for line in f]

    yt_embed = normalize(np.load("yt_embed.npy"))
    wh_embed = normalize(np.load("wh_embed.npy"))
    dists = yt_embed.dot(wh_embed.T)
    bestfits = np.argmax(dists, axis=1)

    load_task_steps()
    
    i = 0
    for task in TASK_STEPS.keys():
        import pdb; pdb.set_trace()
        for vid in YT_embeds[task]:
            yt_title, _ = YT_embeds[task][vid]
            if yt_title.rstrip() != yttitles[i]:
                import pdb; pdb.set_trace()
            bestfit_title = whtitles[bestfits[i]]
            bestfit_article = str(bestfits[i]) + ".json"
            article_path = os.path.join("content", bestfit_article)

            with open(article_path) as f:
                article_json = json.load(f)

            article_sentences = []
            try:
                extract_article_sentences(article_json, article_sentences)
            except:
                pass
            
            article_steps = set()
            for sentence in article_sentences:
                article_steps.update(sentence_steps(sentence, task))
            MISSING_STEPS[task][vid] = [(i,step) for i,step in enumerate(TASK_STEPS[task]) if step not in article_steps]
            i += 1

    import pdb; pdb.set_trace()
    pickle.dump(MISSING_STEPS, open("missing_steps.pkl", "wb"))
