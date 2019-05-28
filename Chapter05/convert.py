from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from models import YoloV3
from utils import load_darknet_weights

flags.DEFINE_string('weights', './yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './yolov3.tf', 'path to output')


def main(_argv):
    yolo = YoloV3()
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

