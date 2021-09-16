import os
from os import getcwd

wd = getcwd()
mulu = ['/' + 'annotations_train_xml', '/' + 'annotations_val_xml']
count = 0
for i in mulu:
    count += 1
    dir = wd + i
    print(dir)
    filenames = os.listdir(dir)
    if count == 1:
        f = open('train.txt', 'w')
        count_1 = 0
        for filename in filenames:
            count_1 += 1
            out_path = dir + '/' + filename.replace('xml', 'jpg')
            out_path = out_path.replace('annotations_train_xml', 'JPEGImages/train2017')
            f.write(out_path + '\n')
        f.close()
        print('done!,total:%s' % count_1)
    elif count == 2:
        f = open('val.txt', 'w')
        count_1 = 0
        for filename in filenames:
            count_1 += 1
            out_path = dir + '/' + filename.replace('xml', 'jpg')
            out_path = out_path.replace('annotations_val_xml', 'JPEGImages/val2017')
            f.write(out_path + '\n')
        f.close()
        print('done!,total:%s' % count_1)

