#复制图片到论文文件夹
import os
import shutil


photoname=['ghsalt000539','kd10','normal105']

srcdirp_pre='predict_out'
srcdirp_rea='data/val/labels'
tardir='/mnt/d/workspace/Paper/论文图片'
for filename in photoname:
    shutil.copy(src=os.path.join(srcdirp_rea,filename+'.png'),dst=os.path.join(tardir,filename+'rea.png'))
    shutil.copy(src=os.path.join(srcdirp_pre,filename+'.png'),dst=os.path.join(tardir,filename+'pre.png'))
print('*****copy done*****')