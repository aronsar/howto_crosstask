import sys, os
sys.path.append("../InferSent")
import nltk
import torch
from models import InferSent
import yt
import numpy as np

def create_embeddings(infer_path, data_path, em_type):
    yt_titles = yt.get_yt_titles()
    with open("data/whtitles", "r") as f:
        wh_titles = [line.rstrip('\n') for line in f]
    
    if em_type == "yt": # Youtube
        save_f = os.path.join(data_path, "yt_embed")
        titles = yt_titles
    elif em_type == "wh": # Wikihow
        save_f = os.path.join(data_path, "wh_embed")
        titles = wh_titles
    else:
        raise "Unknown embedding type: {}".format(em_type)

    nltk.download('punkt')
    V = 1
    MODEL_PATH = os.path.join(infer_path, 'encoder/infersent%s.pkl' % V)
    params_model = {
            'bsize': 256,
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'pool_type': 'max',
            'dpout_model': 0.0,
            'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    infersent = infersent.cuda()

    W2V_PATH = os.path.join(infer_path, 'GloVe/glove.840B.300d.txt')
    infersent.set_w2v_path(W2V_PATH)

    infersent.build_vocab(yt_titles + wh_titles, tokenize=True)    
    embed = infersent.encode(titles, tokenize=True)
    np.save(save_f, embed)
    