代码：
本代码仓基于yolov8增加分块特征融合卷积BFFConv、多维卷积反池化注意力模块MCDA、辅助训练头AuxHead
模块代码分别存储于：
1.分块特征融合卷积BFFConv位于bmayolo-main\ultralytics\nn\extra_modules\block.py下的class EMSConv
2.多维卷积反池化注意力模块MCDA位于bmayolo-main\ultralytics\nn\extra_modules\attention.py下的class MCDA
3.辅助训练头Auxhead位于bmayolo-main\ultralytics\nn\extra_modules\head.py下的class DetectAux
4.yaml文件存储于bmayolo-main\ultralytics\cfg\models
训练时将train.py中model更改为ultralytics/cfg/models/BMA-YOLO.yaml

数据集：
本实验所使用的数据集为开源数据集，均按照训练集、验证集、测试集7：2：1进行划分
可从百度网盘链接中获取：https://pan.baidu.com/s/1ZHoPjYaQRxxjnrvuoCxlOA?pwd=duue 提取码: duue 

如有疑问或需要（如：实验结果权重等）可与我联系：邮箱jh77cf77@163.com