from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
import os


# 计算混淆矩阵 数据，像素匹配率
def rightrate(photoname):
    # 读取图片文件
    y_true = Image.open(os.path.join("data/val/labels", photoname))
    y_pred = Image.open(os.path.join("predict_out", photoname))

    # 将图片转为NumPy数组
    y_true = np.asarray(y_true, np.uint8)
    y_pred = np.asarray(y_pred, np.uint8)

    featrueline_count = 0  # 特征线总数
    featrueline_right = 0  # 特征线预测正确个数
    featruelinetoback = 0  # 特征线预测为背景个数
    featruelinetoress = 0  # 特征线预测为电阻丝个数
    featruelinetobottom = 0  # 特征线预测为回波个数

    ress_count = 0  # 电阻丝总数
    ress_right = 0  # 电阻丝预测正确个数
    resstoback = 0  # 电阻丝预测为背景个数
    resstofeat = 0  # 电阻丝预测为特征线个数
    resstobottom = 0  # 电阻丝预测为回波个数

    bottom_count = 0  # 回波总数
    bottom_right = 0  # 回波预测正确个数
    bottomtoback = 0  # 回波预测为背景个数
    bottomtoress = 0  # 回波预测为电阻丝个数
    bottomtofeat = 0  # 回波预测为特征线个数

    back_count = 0  # 背景总数
    back_right = 0  # 背景预测正确个数
    backtobottom = 0  # 背景预测为回波个数
    backtoress = 0  # 背景预测为电阻丝个数
    backtofeat = 0  # 背景预测为特征线个数
    for i in range(240):
        for j in range(400):
            if y_true[i][j] == 1:
                featrueline_count += 1
                if y_pred[i][j][0] == 128 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    featrueline_right += 1
                elif y_pred[i][j][0] == 0 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    featruelinetoback += 1
                elif y_pred[i][j][0] == 128 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    featruelinetobottom += 1
                elif y_pred[i][j][0] == 0 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    featruelinetoress += 1
            if y_true[i][j] == 2:
                ress_count += 1
                if y_pred[i][j][0] == 0 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    ress_right += 1
                elif y_pred[i][j][0] == 0 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    resstoback += 1
                elif y_pred[i][j][0] == 128 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    resstobottom += 1
                elif y_pred[i][j][0] == 128 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    resstofeat += 1
            if y_true[i][j] == 3:
                bottom_count += 1
                if y_pred[i][j][0] == 128 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    bottom_right += 1
                elif y_pred[i][j][0] == 0 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    bottomtoback += 1
                elif y_pred[i][j][0] == 0 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    bottomtoress += 1
                elif y_pred[i][j][0] == 128 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    bottomtofeat += 1
            if y_true[i][j] == 0:
                back_count += 1
                if y_pred[i][j][0] == 0 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    back_right += 1
                elif y_pred[i][j][0] == 0 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    backtoress += 1
                elif y_pred[i][j][0] == 128 and y_pred[i][j][1] == 128 and y_pred[i][j][2] == 0:
                    backtobottom += 1
                elif y_pred[i][j][0] == 128 and y_pred[i][j][1] == 0 and y_pred[i][j][2] == 0:
                    backtofeat += 1
    word = photoname+'特征线正确率：'+str(featrueline_right/featrueline_count)+'\n电阻丝正确率：'+str(ress_right/ress_count)+'\n回波正确率：'+str(bottom_right/bottom_count)+'\n背景正确率'+str(back_right/back_count)+'\n特征线2背景正确率：'+str(featruelinetoback/featrueline_count)+'\n特征线2电阻丝正确率：'+str(featruelinetoress/featrueline_count)+'\n特征线2回波正确率：'+str(featruelinetobottom/featrueline_count)+'\n电阻丝2背景正确率：'+str(
        resstoback/ress_count)+'\n电阻丝2特征线正确率：'+str(resstofeat/ress_count)+'\n电阻丝2回波正确率：'+str(resstobottom/ress_count)+'\n回波2背景正确率：'+str(bottomtoback/bottom_count)+'\n回波2特征线正确率：'+str(bottomtofeat/bottom_count)+'\n回波2电阻丝正确率：'+str(bottomtoress/bottom_count)+'\n背景2特征线正确率：'+str(backtofeat/back_count)+'\n背景2回波正确率：'+str(backtobottom/back_count)+'\n背景2电阻丝正确率：'+str(backtoress/back_count)
    word += '\n\n'
    return word