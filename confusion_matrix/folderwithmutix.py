import os
from mutix import *


path='predict_out'
file=open('混淆矩阵/All.txt','w')
for filename in os.listdir(path):
    word=rightrate(filename)
    file.write(word)
file.close()