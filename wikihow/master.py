import os
import sys
import pickle
import re
import scipy
import numpy as np
import json
from allennlp.commands.elmo import ElmoEmbedder
from collections import defaultdict
import yt
import create_title_embedding as cte 
import utils


def main():
    # YT Dict [task][video_id]
    yt_meta_f = "data/YT_meta.pkl"
    if not os.path.isfile(yt_meta_f):
        yt.create_yt_meta()
    with open(yt_meta_f, "rb") as f:
        YT_meta = pickle.load(f)
    print("Loaded YT Meta ...")
    
    # Wiki Title
    with open("data/whtitles", "r") as f:
        whtitles = [line.rstrip() for line in f]
    print("Loaded Wikihow Titles ...")

    # YT pure embedding
    yt_embed_f = "data/yt_embed.npy"
    if not os.path.isfile(yt_embed_f):
        cte.create_embeddings("../InferSent", "data/", "yt")
    yt_embed = utils.normalize(np.load(yt_embed_f))
    print("Loaded YT Embeddings ...")

    # Wiki pure embedding
    wh_embed_f = "data/wh_embed.npy"
    if not os.path.isfile(wh_embed_f):
        cte.create_embeddings("../InferSent", "data/", "wh")
    wh_embed = utils.normalize(np.load(wh_embed_f))
    print("Loaded Wikihow Embeddings ...")

    task_steps, task_step_nouns, split_task_steps = utils.load_task_steps()
    primary_task_ids = task_steps.keys()
    print("Split Task Steps ...")

    bestfits = np.argmax(yt_embed.dot(wh_embed.T), axis=1)
    print("Bestfits ...")
    
    i = 0
    total_steps = 0
    missing_steps = defaultdict(dict)
    # for task in task_steps.keys():
    for task, vids in YT_meta.items():
        if task in primary_task_ids:
            task_step_syns = []
            for task_step_vocabs in split_task_steps[task]:
                task_step_syn = []
                for vocab in task_step_vocabs:
                    task_step_syn.extend(utils.get_synonyms(vocab))
                task_step_syns.append(task_step_syn)

            for vid in YT_meta[task]:
                yt_title = YT_meta[task][vid]

                bestfit_title = whtitles[bestfits[i]]
                bestfit_article = str(bestfits[i]) + ".json"
                article_path = os.path.join("content", bestfit_article)
                with open(article_path) as f:
                    article_json = json.load(f)

                article_sentences = []
                try:
                    utils.extract_article_sentences(article_json, article_sentences)
                except:
                    pass
                
                article_steps = set()
                for sentence in article_sentences:
                    article_steps.update(utils.sentence_steps(sentence, task_step_syns, task_steps[task]))


                missing_steps[task][vid] = [(j,step) for j,step in enumerate(task_steps[task]) if step not in article_steps]
                i += 1
                total_steps += len(task_steps[task])

    missed_total = sum([sum([len(missing_steps[task][vid]) for vid in missing_steps[task]]) for task in missing_steps])
    print("Total Count: {}, Missed Count: {}".format(total_steps, missed_total))
    pickle.dump(missing_steps, open("data/missing_steps.pkl", "wb"))


if __name__ == '__main__':
    main()
