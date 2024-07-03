# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def count_ones(element_list):
    return len([x for x in element_list if x == 1])


def extract_info(file_1):
    with open(file_1, 'r') as f:
        data1 = json.load(f)

    average_recall_k1 = []
    p1_list = []
    for f_index in range(0, len(data1)):

        f1 = data1[f_index][0]
        pf1 = data1[f_index][1]
        recall_p1 = []
        p1 = []
        # we start from 1 to remove ourselves
        for k in range(1, 200):
            cut1 = f1[0:k]
            p1k = float(count_ones(cut1))
            recall_p1.append(p1k / pf1)
            if k==199:
                p1.append(p1k)
        average_recall_k1.append(recall_p1)
        p1_list.append(p1)
    print(np.max(p1_list,axis=0))
    avg_p10 = np.average(average_recall_k1, axis=0)

    return avg_p10


def print_graph(info1, info2, info3, file_name, label_y, title_1, title_2, title_3, p):
    fig= plt.figure(figsize=(5, 6))
    plt.plot(range(0, len(info1)), info1, color='r', label=title_1)
    plt.plot(range(0, len(info1)), info2, color='b', label=title_2)
    plt.plot(range(0, len(info1)), info3, color='g', label=title_3)
    plt.grid(b=True, which='major', linestyle="--")
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.4)
    plt.legend(loc=p, shadow=True, fontsize='x-large')
    plt.xlabel("Number of Nearest Results")
    plt.ylabel(label_y)
    fig.savefig(file_name,dpi =300, bbox_inches='tight')
    plt.close(fig)


def compare_and_print(file):
    filename = file.split('_')[0] + '_' + file.split('_')[1]
    t_short = filename
    #label_1 = t_short + '_' + file.split('_')[3]

    recall_p1 = extract_info(file)

    #fname = filename + '_recall.tiff'
    #print_graph(recall_p1, fname, 'Recall', label_1, 'lower right')


    return recall_p1

HBinSim_rootdir = r"/home/wangyan/experiment/HBinSim/function_search/output_HBinSim"
s2v_atte_rootdir = r"/home/wangyan/experiment/HBinSim/function_search/output_atte"
gemini_rootdir = r"/home/wangyan/experiment/HBinSim/function_search/output_gemini"
tif_name = r"/home/wangyan/experiment/HBinSim/function_search/recall.pdf"

e1 = 'HBinSim_embeddings'
e2 = "s2v_attention_embeddings"
e3 = "s2v_gemini_embeddings"

opt = ['O2']
projects = [r"Clobberin'Time","ffmpeg",'heartbeat', "Shellshock", "Shellshock#2", "Venom", "wget", "ws-snmp"]
values1 = []
values2 = []
values3 = []

for o in opt:
    for c in projects:
        f1 = '' + c + '_' + o + '_' + e1 + '_top200.json'
        filedir1 = os.path.join(HBinSim_rootdir, f1)
        values1.append(filedir1)

        f2 = '' + c + '_' + o + '_' + e2 + '_top200.json'
        filedir2 = os.path.join(s2v_atte_rootdir, f2)
        values2.append(filedir2)

        f3 = '' + c + '_' + o + '_' + e3 + '_top200.json'
        filedir3 = os.path.join(gemini_rootdir, f3)
        values3.append(filedir3)

recal_p1 = []
recal_p2 = []
recal_p3 = []

p1 = Pool(8)
result1 = p1.map(compare_and_print, values1)

for t in result1:
    recal_p1.append(t)
recal_p1 = np.average(recal_p1, axis=0)
print("recal_p1: ", recal_p1)

p2 = Pool(8)
result2 = p2.map(compare_and_print, values2)

for t in result2:
    recal_p2.append(t)
recal_p2 = np.average(recal_p2, axis=0)
print("recal_p2: ", recal_p2)

p3 = Pool(8)
result3 = p3.map(compare_and_print, values3)

for t in result3:
    recal_p3.append(t)
recal_p3 = np.average(recal_p3, axis=0)
print("recal_p3: ", recal_p3)

print_graph(recal_p1, recal_p2, recal_p3, tif_name, 'Recall', 'HBinSim', "i2v_attention", "Gemini",'lower right')
