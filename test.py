from tqdm import tqdm
import os
file=[]
f = open("val1.txt","r")
lines = f.readlines()      #读取全部内容 ，并以列表方式返回
for line in lines:
    file.append(line)
for i in tqdm(file):
    f = open('val.txt', 'a')
    f.write("/home/fandong/val2017/"+os.path.split(i)[1])
    f.close()
