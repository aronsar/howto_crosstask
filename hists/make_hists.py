import dill
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def normed(input_list):
    max_element = max(input_list)
    output_list = [e/max_element for e in input_list]
    return output_list

gt_hist, bk_hist = dill.load(open("hists.dill", "rb"))

for task in gt_hist.keys():
    for step in gt_hist[task].keys():
        labels = sorted([label for label in gt_hist[task][step].keys()])
        gt_freqs = normed([gt_hist[task][step][interval] for interval in labels])
        bk_freqs = normed([bk_hist[task][step][interval] for interval in labels])

        x = np.arange(len(labels)) # label locations
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, gt_freqs, width, label='ground_truth')
        rects2 = ax.bar(x + width/2, bk_freqs, width, label='background')

        ax.set_ylabel('Frequency')
        ax.set_title('Histogram for ' + task)
        ax.set_xticks([l for i, l in enumerate(x) if i % 4 == 0])
        ax.set_xticklabels([l for i, l in enumerate(labels) if i % 4 == 0])
        ax.legend()

        fig.savefig(task + '---' + str(step) + '.png')
        plt.close(fig)
