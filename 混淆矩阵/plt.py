# 绘制混淆矩阵
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 定义真实标签和预测标签
true_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
pred_labels = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2])

# 计算混淆矩阵
confusion_matrix = np.zeros((4, 4), dtype=np.float32)
confusion_matrix[0][0] = 0.926
confusion_matrix[0][3] = 0.074
confusion_matrix[1][1] = 0.917
confusion_matrix[1][3] = 0.083
confusion_matrix[2][2] = 0.959
confusion_matrix[2][3] = 0.041
confusion_matrix[3][3] = 0.991
confusion_matrix[3][0] = 0.004
confusion_matrix[3][1] = 0.001
confusion_matrix[3][2] = 0.004

# 绘制混淆矩阵图
sns.set()
sns.heatmap(confusion_matrix, annot=True, cmap="YlOrBr", fmt='.1%', xticklabels=['特征线', '电阻丝', '回波', '背景'], yticklabels=['特征线', '电阻丝', '回波', '背景'], linewidths=1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fig1.png')
