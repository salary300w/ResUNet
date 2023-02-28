#复制图片到论文文件夹
import os
import shutil


photoname=['normalsalt001105','normalsalt00199','kd10','ghgs0000139','kdsalt00110','normal99','normalgs0000199','normalfz105','normalsalt000599','kdgs0000110','normal105','ghsalt000539','ghsalt000541','normalfz99','kdfz10','kdsalt000510']

srcdirp_pre='predict_out'
srcdirp_rea='data/val/labels'
tardir='/mnt/d/workspace/Paper/论文图片'
for filename in photoname:
    shutil.copy(src=os.path.join(srcdirp_rea,filename+'.png'),dst=os.path.join(tardir,filename+'rea.png'))
    shutil.copy(src=os.path.join(srcdirp_pre,filename+'.png'),dst=os.path.join(tardir,filename+'pre.png'))
print('*****copy done*****')