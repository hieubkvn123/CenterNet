import cv2
import time
import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from gluoncv import model_zoo, data, utils

center_net_model = 'center_net_resnet101_v1b_dcnv2_coco'
yolo_model = 'yolo3_darknet53_coco'

net = model_zoo.get_model(yolo_model, pretrained=True)

img = cv2.imread('/home/hieu/Pictures/traffic/1.jpg')
H, W, C = img.shape

start = time.time()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = mx.nd.array(img.reshape(H, W, C))
x, img = data.transforms.presets.center_net.transform_test(x, short=512)
class_IDS, scores, bounding_boxes = net(x)
end = time.time()
print(end - start)

ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDS[0], class_names=net.classes)
plt.show()
