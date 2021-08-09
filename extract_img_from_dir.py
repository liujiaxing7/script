import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        rate=0.08    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = "/home/fandong/Code/Data/data5/remap/data_20210803/record_1/imsee_data.bag.imgs.L/"    #源图片文件夹路径
	tarDir = '/home/fandong/Code/Data/data5/remap_/data_20210803/1/'    #移动到新的文件夹路径
	moveFile(fileDir)