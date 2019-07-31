"""
YOLO_v3 Model Defined in Keras.
Reference: https://github.com/qqwweee/keras-yolo3.git
"""
import cv2
from config import kerasTextModel,IMGSIZE,keras_anchors,class_names
from yolo3.keras_yolo3 import yolo_text,box_layer,K

from PIL import Image
import numpy as np
import tensorflow as tf


def resize_im(w, h, scale=416, max_scale=608):
    f = float(scale) / min(h, w)
    if max_scale is not None:
        if f * max(h, w) > max_scale:
            f = float(max_scale) / max(h, w)
    newW, newH = int(w * f), int(h * f)

    return newW - (newW % 32), newH - (newH % 32)


anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)

num_classes = len(class_names)
textModel = yolo_text(num_classes,anchors)
textModel.load_weights(kerasTextModel)


sess = K.get_session()
image_shape = K.placeholder(shape=(2, ))##图像原尺寸:h,w
input_shape = K.placeholder(shape=(2, ))##图像resize尺寸:h,w
box_score = box_layer([*textModel.output,image_shape,input_shape],anchors, num_classes)



def text_detect(img,prob = 0.05):
    im    = Image.fromarray(img)
    scale = IMGSIZE[0]
    w,h   = im.size
    w_,h_ = resize_im(w,h, scale=scale, max_scale=2048)##短边固定为608,长边max_scale<4000
    #boxed_image,f = letterbox_image(im, (w_,h_))
    boxed_image = im.resize((w_,h_), Image.BICUBIC)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    imgShape   = np.array([[h,w]])
    inputShape = np.array([[h_,w_]])
    
    

    """
    pred = textModel.predict_on_batch([image_data,imgShape,inputShape])
    box,scores = pred[:,:4],pred[:,-1]
    
    """
    box,scores = sess.run(
    [box_score],
    feed_dict={
        textModel.input: image_data,
        input_shape: [h_, w_],
        image_shape: [h, w],
        K.learning_phase(): 0
    })[0]


    keep = np.where(scores>prob)
    box[:, 0:4][box[:, 0:4]<0] = 0
    box[:, 0][box[:, 0]>=w] = w-1
    box[:, 1][box[:, 1]>=h] = h-1
    box[:, 2][box[:, 2]>=w] = w-1
    box[:, 3][box[:, 3]>=h] = h-1
    box = box[keep[0]]
    scores = scores[keep[0]]
    return box,scores

if __name__ == '__main__':
    box , scores = text_detect(np.array(Image.open('1.jpg')))
    img = cv2.imread('1.jpg')
    for x1,y1,x2,y2 in box:
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite('test_1.jpg',img)
    print (box,scores)
