import os

file_path='/home/fandong/Code/Data/data6/detect_voc/VisDrone2019-DET-val'

xmlfilepath = os.path.join(file_path,"xml/")
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
total_xml.sort()

ftrainval = open(file_path + '/trainval.txt', 'w')

count1= 0
for i in range(num):
    name = total_xml[i][:-4] + '\n'
    if i in range(num):
        ftrainval.write(name)
        count1 += 1

print(count1)
