from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.nn.functional import adaptive_max_pool1d
from torch.utils.data import Dataset
import math
import pandas as pd
import os
import numpy as np
import re
import random
import torch.nn.functional as F


def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path,'r') as f:
        idx = f.readline()
        while idx is not '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


class CrossTask(Dataset):
    def __init__(
            self,
            data_path,
            features_path,
            features_path_3D,
            we, # word embedding
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            we_dim=300, # dimension of word vectors
            max_words=20
    ):
        self.we_dim = we_dim
        self.we = we
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = {'2d': features_path}
        if features_path_3D:
            self.feature_path['3d'] = features_path_3D
        self.annotation_path = os.path.join(data_path, 'annotations')
        self.video_ids = [fname[:-4] for fname in os.listdir("./primary_vids")]
        #self.video_ids = [fname[:-4] for fname in os.listdir(self.annotation_path)]
        task_info_path = os.path.join(data_path, 'tasks_primary.txt')
        self.task_info = read_task_info(task_info_path)
        self.max_words = max_words
        self.step_vectors = {}
        for task, steps in self.task_info['steps'].items():
            self.step_vectors[task] = th.stack([self._words_to_we(self._tokenize_text(step)) for step in steps], dim=0)

    def __len__(self):
        return len(self.video_ids)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def _get_video(self, feature_path):
        video = np.load(feature_path)
        return th.from_numpy(video).float()

    def _get_labels(self, task, video_id, T):
        labels = np.zeros([T, self.task_info['n_steps'][task]])
        with open(os.path.join(self.annotation_path, task+'_'+video_id+'.csv'), 'r') as f:
            for line in f:
                k, s, e = line.strip().split(',')
                s = int(float(s))
                e = int(math.ceil(float(e)))
                labels[s:e+1, int(k)-1] = 1
        return th.from_numpy(labels)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx][-11:]
        task = self.video_ids[idx][:-12]
        vid_path_2d = os.path.join(self.feature_path['2d'], video_id+'.npy')
        vid_path_3d = os.path.join(self.feature_path['3d'], video_id+'.npy')
        video_2d = self._get_video(vid_path_2d)
        video_3d = self._get_video(vid_path_3d)
        T = len(video_2d)
        video_3d = adaptive_max_pool1d(video_3d.transpose(1,0)[None,:,:],T).view(-1,T).transpose(1,0)
        video = th.cat((F.normalize(video_2d, dim=1), F.normalize(video_3d, dim=1)), dim=1)
        labels = self._get_labels(task, video_id, T)

        steps = self.step_vectors[task]
        
        return {'video': video, 'steps': steps, 'video_id': video_id, 'task': task, 'labels': labels}
