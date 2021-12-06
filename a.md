# 坐标系转换



## 资料汇总：

https://blog.csdn.net/xueluowutong/article/details/80950915

https://zhuanlan.zhihu.com/p/179933911

https://zhuanlan.zhihu.com/p/55132750



## 张氏标定法：

https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247484873&idx=1&sn=e2affdaaf226d83bda7532963e917f62&chksm=97d7e05ea0a06948c4d31348913ffe2a70ab67b333f6445000a3b0541a0093849b600438a2d5&scene=21#wechat_redirect

https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247484931&idx=1&sn=c3b8ca92772b196fd4107328df0522cb&chksm=97d7e394a0a06a825682a492218a276d7d9e0f9737920f2c1789cd271c85fae638aa069e9662&scene=21#wechat_redirect

https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247484969&idx=1&sn=abda897d596adedfe13c3495b0b02d50&chksm=97d7e3bea0a06aa8e3fcb2b68c8863488e66410166d7bf159d6d3ff2c103f515345ada4c1191&scene=21#wechat_redirect

https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247485041&idx=1&sn=9268b9a0aa90b4f64f216e08ef64f63d&chksm=97d7e3e6a0a06af0f33720bc8d9ca4e4a07ca0a8e7c8a6b0b4d7261e03fb5e1ee564ae367b44&scene=21#wechat_redirect

## 个人理解相机标定：

相机标定的输入：世界坐标系下的坐标和像素坐标系下的坐标

相机标定的输出：相机的内外参数

相机的内参：指相机的焦距等内部的一些参数

相机的外参：是描述相机放置的位置和高度

当标定完成后，如果改变相机的位置，是改变相机的外参。

## 个人理解坐标转换：

###  1.世界坐标系到相机坐标系到图像坐标系到像素坐标系。

#### 由世界坐标系到相机坐标系：

经过旋转和平移，这里产生的矩阵是外参矩阵。在相机标定时，是把世界坐标系放在了标定板上，也就是已知世界坐标系。

![img](https://img-blog.csdn.net/20180707141348947?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h1ZWx1b3d1dG9uZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 由相机坐标系到图像坐标系：

![img](https://img-blog.csdn.net/20180707142747216?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h1ZWx1b3d1dG9uZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

通过三角形相似，得到一个带有相机焦距的矩阵（1）。

#### 由图像坐标系到像素坐标系：

![img](https://img-blog.csdn.net/20180707144133312?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h1ZWx1b3d1dG9uZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

得到一个带1mm内的像素个数的矩阵（2）（随着相机改变而改变）

将矩阵（2）右乘矩阵（1）可以得到相机的内参矩阵。又电脑上的像素坐标系的坐标是已知的，

![img](https://img-blog.csdn.net/20180707144409406?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h1ZWx1b3d1dG9uZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

所以根据这个公式，通过多组点列写线性方程即可求出内外参数。
