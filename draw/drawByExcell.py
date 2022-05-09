import numpy as np
from matplotlib import pyplot as plt
import chinese
import xlrd
import xlwt

# chinese.set_ch()
filename='./0821_esca.xls'
book_wind=xlrd.open_workbook(filename=filename)
wind_sheet1=book_wind.sheets()[0]					#这个[0]我没看懂
#读取第1行标题
title=wind_sheet1.row_values(0)

#读取第一、二、三列标题以下的数据 col_values(colx,start_row=0,end_row=none)
x=['GPU','NPU']
y0=wind_sheet1.col_values(0)
y1=wind_sheet1.col_values(1)
y2=wind_sheet1.col_values(2)
y3=wind_sheet1.col_values(3)
y4=wind_sheet1.col_values(4)
y5=wind_sheet1.col_values(5)

plt.ylim(0,1)
#绘制曲线图
line0,=plt.plot(x,y0,label='V1.0.3')
line1,=plt.plot(x,y1,label='V1.0.12')
# line1.set_dashes([2,2,10,2])			#将曲线设置为点划线，set_dashes([line_space,space_space,line_space,space_space])
line2,=plt.plot(x,y2,label='V1.0.17')
# line2.set_dashes([2,2,2,2])
line3,=plt.plot(x,y3,label='V1.0.15')
# line3.set_dashes([2,2,2,2])
line4,=plt.plot(x,y4,label='V1.0.20')
# line4.set_dashes([2,2,2,2])
line5,=plt.plot(x,y5,label='V1.0.21')
# line5.set_dashes([2,2,2,2])
plt.title('EVT3',fontsize=16)
plt.legend(loc=4)						#设置图例位置，4表示右下角
plt.show()