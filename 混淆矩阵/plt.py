# 绘制混淆矩阵
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

mutix = [
    ['ghsalt000539.png', '0.927', '0.073', '0.943', '0.057', '0.949', '0.051', '0.989', '0.004', '0.003', '0.004'],
    ['kd10.png', '0.926', '0.074', '0.917', '0.083', '0.959', '0.041', '0.991', '0.004', '0.001', '0.004'],
    ['normal105.png', '0.906', '0.094', '0.912', '0.088', '0.924', '0.076', '0.987', '0.005', '0.004', '0.004']
]

savepath='混淆矩阵'
for num in mutix:
    # 计算混淆矩阵
    confusion_matrix = np.zeros((4, 4), dtype=np.float32)
    confusion_matrix[0][0] = num[1]
    confusion_matrix[0][3] = num[2]
    confusion_matrix[1][1] = num[3]
    confusion_matrix[1][3] = num[4]
    confusion_matrix[2][2] = num[5]
    confusion_matrix[2][3] = num[6]
    confusion_matrix[3][3] = num[7]
    confusion_matrix[3][0] = num[8]
    confusion_matrix[3][1] = num[10]
    confusion_matrix[3][2] = num[9]

    # 绘制混淆矩阵图
    sns.set()
    sns.heatmap(confusion_matrix, annot=True, cmap="YlOrBr", fmt='.1%', xticklabels=['特征线', '电阻丝', '回波', '背景'], yticklabels=['特征线', '电阻丝', '回波', '背景'], linewidths=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(savepath,num[0]))
    plt.close()