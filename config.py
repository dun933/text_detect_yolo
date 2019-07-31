import os
########################文字检测########################
##文字检测引擎
pwd = os.getcwd()
opencvFlag = 'keras' ##keras,opencv,darknet，模型性能 keras>darknet>opencv
IMGSIZE = (608,608)## yolo3 输入图像尺寸
## keras 版本anchors
#keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
keras_anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
class_names = ['none','text',]
kerasTextModel=os.path.join(pwd,"model.hd5")##keras版本模型权重文件
ocrModelKeras = os.path.join(pwd,"models","ocr-dense-keras.h5")##keras版本OCR，暂时支持dense