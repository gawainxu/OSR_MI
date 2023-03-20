import os
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pearson(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum([i*i for i in x])
    sum_y2 = sum([i*i for i in y])
    n = len(x)
    sum_xy = sum([i*j for i, j in zip(x, y)])
    r = (n*sum_xy - sum_x*sum_y) / math.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))

    return r



file_root = "D:\\projects\\open_cross_entropy\\info_analysis\\save"
methods = ["vanillia", "msp_opt", "arpl", "arpl_cs"]
model = "vgg32"


# vgg32, cifar10 -------------------------------------------------------------------------------
dataset = "cifar10"
#loss_files = [["cifar-10-10_vanillia_classifier32_0_module.avgpool_mi_losses", "cifar-10-10_vanillia_classifier32_1_module.avgpool_mi_losses",
#              "cifar-10-10_vanillia_classifier32_2_module.avgpool_mi_losses", "cifar-10-10_vanillia_classifier32_3_module.avgpool_mi_losses",
#              "cifar-10-10_vanillia_classifier32_4_module.avgpool_mi_losses"], ["cifar-10-10_msp_optimal_classifier32_0_module.avgpool_mi_losses",
#              "cifar-10-10_msp_optimal_classifier32_1_module.avgpool_mi_losses", "cifar-10-10_msp_optimal_classifier32_2_module.avgpool_mi_losses",
#              "cifar-10-10_msp_optimal_classifier32_3_module.avgpool_mi_losses", "cifar-10-10_msp_optimal_classifier32_4_module.avgpool_mi_losses"],
#              ["cifar-10-10_arpl_classifier32_0_module.avgpool_mi_losses", "cifar-10-10_arpl_classifier32_1_module.avgpool_mi_losses",
#               "cifar-10-10_arpl_classifier32_2_module.avgpool_mi_losses", "cifar-10-10_arpl_classifier32_3_module.avgpool_mi_losses",
#               "cifar-10-10_arpl_classifier32_4_module.avgpool_mi_losses"], ["cifar-10-10_arpl_cs_classifier32_0_module.avgpool_mi_losses",
#               "cifar-10-10_arpl_cs_classifier32_1_module.avgpool_mi_losses", "cifar-10-10_arpl_cs_classifier32_2_module.avgpool_mi_losses",
#               "cifar-10-10_arpl_cs_classifier32_3_module.avgpool_mi_losses","cifar-10-10_arpl_cs_classifier32_4_module.avgpool_mi_losses"]]

layer = "fc"
loss_files = [["cifar-10-10_vanillia_classifier32_0_module.fc_mi_losses", "cifar-10-10_vanillia_classifier32_1_module.fc_mi_losses",
              "cifar-10-10_vanillia_classifier32_2_module.fc_mi_losses", "cifar-10-10_vanillia_classifier32_3_module.fc_mi_losses",
              "cifar-10-10_vanillia_classifier32_4_module.fc_mi_losses"], ["cifar-10-10_msp_optimal_classifier32_0_module.fc_mi_losses",
              "cifar-10-10_msp_optimal_classifier32_1_module.fc_mi_losses", "cifar-10-10_msp_optimal_classifier32_2_module.fc_mi_losses",
              "cifar-10-10_msp_optimal_classifier32_3_module.fc_mi_losses", "cifar-10-10_msp_optimal_classifier32_4_module.fc_mi_losses"],
              ["cifar-10-10_arpl_classifier32_0_module.fc_mi_losses", "cifar-10-10_arpl_classifier32_1_module.fc_mi_losses",
               "cifar-10-10_arpl_classifier32_2_module.fc_mi_losses", "cifar-10-10_arpl_classifier32_3_module.fc_mi_losses",
               "cifar-10-10_arpl_classifier32_4_module.fc_mi_losses"], ["cifar-10-10_arpl_cs_classifier32_0_module.fc_mi_losses",
               "cifar-10-10_arpl_cs_classifier32_1_module.fc_mi_losses", "cifar-10-10_arpl_cs_classifier32_2_module.fc_mi_losses",
               "cifar-10-10_arpl_cs_classifier32_3_module.fc_mi_losses","cifar-10-10_arpl_cs_classifier32_4_module.fc_mi_losses"]]

accs = [[95.17, 95.3, 91.27, 95.88, 94.27], 
        [96.92, 96.93, 93.38, 97.17, 96.41], 
        [95.03, 95.28, 90.13, 95.67, 93.85],
        [96.55, 96.12, 92.43, 96.43, 95.67]]

oscrs = [[86.06, 86.45, 83.70, 83.98, 87.75], 
         [91.29, 91.44, 88.35, 90.45, 91.47], 
         [87.09, 86.79, 83.50, 86.10, 88.12],
         [89.54, 89.11, 86.51, 87.22, 90.42]]
# ------------------------------------------------------------------------------------------------


"""
# vgg32, svhn -------------------------------------------------------------------------------------
dataset = "svhn"
loss_files = [["svhn_vanillia_classifier32_0_module.avgpool_mi_losses", "svhn_vanillia_classifier32_1_module.avgpool_mi_losses",
              "svhn_vanillia_classifier32_2_module.avgpool_mi_losses", "svhn_vanillia_classifier32_3_module.avgpool_mi_losses",
              "svhn_vanillia_classifier32_4_module.avgpool_mi_losses"], ["svhn_msp_optimal_classifier32_0_module.avgpool_mi_losses",
              "svhn_msp_optimal_classifier32_1_module.avgpool_mi_losses", "svhn_msp_optimal_classifier32_2_module.avgpool_mi_losses",
              "svhn_msp_optimal_classifier32_3_module.avgpool_mi_losses", "svhn_msp_optimal_classifier32_4_module.avgpool_mi_losses"],
              ["svhn_arpl_classifier32_0_module.avgpool_mi_losses", "svhn_arpl_classifier32_1_module.avgpool_mi_losses",
               "svhn_arpl_classifier32_2_module.avgpool_mi_losses", "svhn_arpl_classifier32_3_module.avgpool_mi_losses",
               "svhn_arpl_classifier32_4_module.avgpool_mi_losses"], ["svhn_arpl_cs_classifier32_0_module.avgpool_mi_losses",
               "svhn_arpl_cs_classifier32_1_module.avgpool_mi_losses", "svhn_arpl_cs_classifier32_2_module.avgpool_mi_losses",
               "svhn_arpl_cs_classifier32_3_module.avgpool_mi_losses","svhn_arpl_cs_classifier32_4_module.avgpool_mi_losses"]]
accs = [[91.14, 90.74, 91.63, 91.37, 92.18],
        [96.13, 96.24, 97.00, 96.51, 97.28],
        [91.12, 92.38, 91.19, 90.97, 93.59],
        [91.93, 92.30, 93.23, 90.09, 93.60]]

oscrs = [[84.92, 84.25, 83.35, 87.59, 86.74],
         [93.47, 93.49, 94.17, 94.53, 94.55],
         [84.15, 86.77, 78.06, 86.59, 86.53],
         [87.60, 87.35, 85.45, 84.67, 86.83]]
# ------------------------------------------------------------------------------------------------
"""

"""
# vgg32, tinyimagenet------------------------------------------------------------------------------
dataset = "TinyImageNet"
loss_files = [["tinyimagenet_vanillia_classifier32_0_module.avgpool_mi_losses", "tinyimagenet_vanillia_classifier32_1_module.avgpool_mi_losses",
                 "tinyimagenet_vanillia_classifier32_2_module.avgpool_mi_losses", "tinyimagenet_vanillia_classifier32_3_module.avgpool_mi_losses"],       # "tinyimagenet_vanillia_classifier32_4_module.avgpool_mi_losses"
                ["tinyimagenet_msp_optimal_classifier32_0_module.avgpool_mi_losses", "tinyimagenet_msp_optimal_classifier32_1_module.avgpool_mi_losses",
                 "tinyimagenet_msp_optimal_classifier32_2_module.avgpool_mi_losses", "tinyimagenet_msp_optimal_classifier32_3_module.avgpool_mi_losses"],   # "tinyimagenet_msp_optimal_classifier32_4_module.avgpool_mi_losses"
                ["tinyimagenet_arpl_classifier32_0_module.avgpool_mi_losses", "tinyimagenet_arpl_classifier32_1_module.avgpool_mi_losses",
                 "tinyimagenet_arpl_classifier32_2_module.avgpool_mi_losses", "tinyimagenet_arpl_classifier32_3_module.avgpool_mi_losses"],                # "tinyimagenet_arpl_classifier32_3_module.avgpool_mi_losses"
                ["tinyimagenet_arpl_cs_classifier32_0_module.avgpool_mi_losses", "tinyimagenet_arpl_cs_classifier32_1_module.avgpool_mi_losses",
                "tinyimagenet_arpl_cs_classifier32_2_module.avgpool_mi_losses", "tinyimagenet_arpl_cs_classifier32_3_module.avgpool_mi_losses"]]           # "tinyimagenet_arpl_cs_classifier32_4_module.avgpool_mi_losses"

accs = [[38.9, 45.85, 37.2, 43.15],             # 55.0
        [71.1, 64.3, 63.5, 63.85],              # 74.0
        [67.3, 60.35, 59.5, 62.15],             # 72.8
        [62.6, 60.9, 59.65, 56.35]]             # 70.9

oscrs = [[26.91, 33.47, 25.54, 30.55],          # 39.37
         [58.51, 52.69, 52.44, 52.23],          # 62.15
         [52.86, 47.08, 47.72, 50.23],          # 60.03
         [49.94, 49.54, 47.09, 44.93]]          # 57.94
"""

batch_size = 64

mins = []
for losses_group in loss_files:
    group_mins = []
    for losses_path in losses_group:
        with open(os.path.join(file_root, losses_path), "rb") as f:
            losses = pickle.load(f)
            group_mins.append(min(losses))
    mins.append(group_mins)

#print(mins)

fig, ax = plt.subplots()
group_means = []
for group_mins in mins:
    group_mins = [-i/batch_size for i in group_mins]
    group_means.append(sum(group_mins) / len(group_mins))

acc_means = [sum(acc)/len(acc) for acc in accs]
oscr_means = [sum(oscr)/len(oscr) for oscr in oscrs]

#fig = plt.figure()
#ax = fig.add_subplot(111)
fig, ax = plt.subplots(figsize=(8, 6))
ax2 = ax.twinx()
width = 0.1
#ax.bar(X + 0.00, group_means, color="b", width=0.25)
#ax2.bar(X+0.25, acc_means, color = 'g', width = 0.25)
#ax2.bar(X+0.5, oscr_means, color = 'r', width = 0.25)

df_plot = pd.DataFrame({"MI": group_means, "ACC": acc_means, "OSCR": oscr_means}, 
                        index=methods)
p1 = df_plot.MI.plot(kind="bar", color="orange", ax=ax, width=width, position=0)
p2 = df_plot.ACC.plot(kind="bar", color="blue", ax=ax2, width=width, position=1)
p3 = df_plot.OSCR.plot(kind="bar", color="green", ax=ax2, width=width, position=2)

font_title = {'family': 'Zapfino',
               'color':  'k',
               'weight': 'bold',
               'size': 12,}

ax.set_ylabel("Mutual Information (Nats)")
ax2.set_ylabel("Accuracy (%)")
ax.set_xlabel("Methods")
plt.title('Mutual Information, Inlier Accuracy and OSR Aaccuracy' + " (" + dataset + ")", fontdict=font_title)

ax.legend(loc=2)
ax2.legend(loc=1)
plt.savefig(file_root + "\\mi_" + dataset + "_" + model + "_" + layer + ".png")

print("mi & acc", pearson(group_means, acc_means))
print("mi & oscr", pearson(group_means, oscr_means))