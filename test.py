import sys
CENTERNET_PATH = 'src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = './models/ctdet_coco_hg.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

img = '/home/hieu/Pictures/traffic/1.jpg' 
ret = detector.run(img)['results']
