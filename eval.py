from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import torch as th
from torch.utils.data import DataLoader
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from dp.dp import dp
from crosstask import CrossTask
from model import Net
import csv
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_mode', type=int, default=1)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_pair', type=int, default=32)
parser.add_argument('--feature_dim', type=int, default=4096)
parser.add_argument('--we_dim', type=int, default=300)
parser.add_argument('--max_words', type=int, default=20)
parser.add_argument('--feature_framerate', type=float, default=1)
parser.add_argument('--feature_framerate_3D', type=float, default=1.5)
parser.add_argument('--model_path', required=1)
parser.add_argument('--sentence_dim', type=int, default=-1,
    help='sentence dimension')
parser.add_argument('--we_path', 
    default='./data/GoogleNews-vectors-negative300.bin',
    help='word embedding path')
parser.add_argument('--num_thread_reader', type=int, default=1,
    help='')
parser.add_argument('--embd_dim', type=int, default=6144,
    help='embedding dim')
parser.add_argument(
    '--features_path',
    default='features',
    help='feature path')
parser.add_argument(
    '--features_path_3D',
    default='features_3D',
    help='3D feature path')
parser.add_argument(
    '--data_path',
    default='crosstask_release',
    help='')
parser.add_argument(
    '--use_gt_order',
    action='store_true',
    help='during evaluation, use the ground truth ordering of the action steps')
parser.add_argument(
    '--use_greedy_order',
    action='store_true',
    help='instead of using the canonical order and dp, simply assign the highest scoring video segment the action step label')

args = parser.parse_args()

def score(video, steps, model):
    sim_matrix = model.forward(video,steps).transpose(1,0)
    return sim_matrix.detach().cpu().numpy()

def get_recall(y_true, y):
    return ((y*y_true).sum(axis=0)>0).sum() / ((y_true.sum(axis=0)>0).sum() + 1e-5)

def find_gt_order(annotation_filepath):
    with open(annotation_filepath, "r") as annot_file:
        reader = csv.reader(annot_file)
        gt_order = []
        for line in reader:
            # actions organized in line by start time
            action_num, _, _ = line
            if action_num not in gt_order:
                gt_order.append(int(action_num)-1)
        return gt_order

def eval(model, dataloader, save_argmax=False, argmaxes={}):
    import pdb; pdb.set_trace()
    model.eval()
    recalls = {}
    counts = {}
    for batch in dataloader:
        for sample in batch:
            video = sample['video'].cuda() if args.gpu_mode else sample['video']
            video_id = sample['video_id']
            text = sample['steps'].cuda() if args.gpu_mode else sample['steps']
            task = sample['task']
            scores = score(video, text, model)
            y = np.empty(scores.shape, dtype=np.float32)
    
            if args.use_greedy_order:
                y = np.zeros(scores.shape, dtype=np.int)
                amax = np.argmax(scores, axis=0)
                if save_argmax:
                    y[amax, np.arange(scores.shape[1])] = 1
                    if task not in argmaxes:
                        # HERP DERP YOU WANT MAX NOT ARGMAX
                        argmaxes[task] = amax
                    else:
                        argmaxes[task] = np.vstack([argmaxes[task], amax])

                else: # use the argmaxes info to identify missing action steps
                    mod_amax = []
                    step_idx = []
                    argmax_threshold = np.average(argmaxes[task], axis=0)
                    for i, (m,a) in enumerate(zip(argmax_threshold, amax)):
                        if m < a: # only predict label for colum if pass threshold
                            mod_amax.append(a)
                            step_idx.append(i)
                    mod_amax = np.array(mod_amax)
                    step_idx = np.array(step_idx)
                    y[mod_amax, step_idx] = 1

            elif args.use_gt_order:
                annotation_filename = task + "_" + video_id + ".csv"
                annotation_filepath = os.path.join(
                    "/data/aronsar/CrossTask/crosstask_release/annotations/", annotation_filename)
                gt_order = find_gt_order(annotation_filepath)
                gt_order_scores = scores[:, gt_order]
                gt_order_y = np.zeros(np.shape(gt_order_scores), dtype=np.float32)
                dp(gt_order_y, -gt_order_scores, exactly_one=True)
                y[:, gt_order] = gt_order_y
            else:
                dp(y, -scores, exactly_one=True)


            y_true = sample['labels'].numpy()
            if task not in recalls:
                recalls[task] = 0.
            recalls[task] += get_recall(y_true, y)
            if task not in counts:
                counts[task] = 0
            counts[task] += 1
    recalls = {task: recall / counts[task] for task,recall in recalls.items()}
    if save_argmax:
        return argmaxes
    else:
        return recalls

print ('Loading word vectors...')
we = KeyedVectors.load_word2vec_format(args.we_path, binary=1)
testset = CrossTask(
    data_path=args.data_path,
    features_path=args.features_path,
    features_path_3D=args.features_path_3D,
    we=we,
    feature_framerate=args.feature_framerate,
    feature_framerate_3D=args.feature_framerate_3D,
    we_dim=args.we_dim,
    max_words=args.max_words,
)
testloader = DataLoader(
    testset,
    batch_size=1,
    num_workers=args.num_thread_reader,
    shuffle=False,
    drop_last=False,
    collate_fn=lambda batch: batch,
)

model = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    n_pair=args.n_pair,
    max_words=args.max_words,
    sentence_dim=args.sentence_dim,
)
if args.gpu_mode:
    model = model.cuda()
print ('Loading the model...')
model.load_checkpoint(args.model_path)

print ('Evaluating...')
if args.use_greedy_order:
    argmaxes = eval(model, testloader, save_argmax=True)
    recalls = eval(model, testloader, argmaxes=argmaxes)
else:
    recalls = eval(model, testloader)

print ('Results:')
for task, r in recalls.items():
    print ('{0}. Recall = {1:0.2f}'.format(testset.task_info['title'][task], r*100))

print ('Average recall: {0:0.2f}'.format(100*np.mean(list(recalls.values()))))


