import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.font_manager as fm

# 读取文件
with open('logs/loss_2023_02_25_18_51_45/epoch_loss.txt', 'r') as f:
    lines = f.readlines()



font_path = 'C:\Windows\Fonts\STKAITI.TTF'
font_prop = fm.FontProperties(fname=font_path)
# 提取数据
x_data = []
train_data = []
val_data = []
i = 1
for line in lines:
    x = line.strip()
    train_data.append(float(x))
    x_data.append(float(i))
    i += 1
# 读取文件
with open('logs/loss_2023_02_25_18_51_45/epoch_val_loss.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    x = line.strip()
    val_data.append(float(x))

# 绘制折线图
train_data_smooth = savgol_filter(train_data, window_length=11, polyorder=3)
val_data_smooth = savgol_filter(val_data, window_length=11, polyorder=3)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(x_data, train_data_smooth, linewidth=2.5, label='train')
plt.plot(x_data, val_data_smooth, linewidth=2.5, label='test')
plt.legend(loc='best')
plt.savefig('fig.png')