#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Modified: Keigo Takahashi (Forked from demo.py)

"""
Detects Cars in an image using KittiBox.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiBox weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import sys
import collections
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'incl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'incl', 'utils'))
from incl.utils import train_utils as kittibox_utils

try:
    # Check whether setup was done correctly
    import incl.tensorvision.utils as tv_utils
    import incl.tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


class annotate:
    def __init__(self):
        # configure logging
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            level=logging.INFO,
                            stream=sys.stdout)

        # https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070

        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS

        self.flags.DEFINE_string('logdir', None,
                                 'Path to logdir.')
        self.flags.DEFINE_string('input_image', None,
                                 'Image to apply KittiBox.')
        self.flags.DEFINE_string('output_image', None,
                                 'Image to apply KittiBox.')

        self.default_run = 'KittiBox_pretrained'
        self.weights_url = ("ftp://mi.eng.cam.ac.uk/"
                            "pub/mttt2/models/KittiBox_pretrained.zip")
        # TODO: Make this as a comment for error of argument conflict of "gpu".
        # tv_utils.set_gpus_to_use()

        if self.FLAGS.logdir is None:
            # Download and use weights from the MultiNet Paper
            if 'TV_DIR_RUNS' in os.environ:
                runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                        'KittiBox')
            else:
                runs_dir = os.path.join(os.path.dirname(__file__), 'RUNS')
            self.maybe_download_and_extract(runs_dir)
            logdir = os.path.join(runs_dir, self.default_run)
        else:
            logging.info("Using weights found in {}".format(self.FLAGS.logdir))
            logdir = self.FLAGS.logdir

        # Loading hyperparameters from logdir
        self.hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')
        self.hypes["dirs"]["data_dir"] = os.path.join(os.path.dirname(__file__), self.hypes["dirs"]["data_dir"])

        logging.info("Hypes loaded successfully.")

        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        modules = tv_utils.load_modules_from_logdir(logdir)
        logging.info("Modules loaded successfully. Starting to build tf graph.")

        # Create tf graph and build module.
        with tf.Graph().as_default():
            # Create placeholder for input
            self.image_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(self.image_pl, 0)

            # build Tensorflow graph using the model from logdir
            self.prediction = core.build_inference_graph(self.hypes, modules,
                                                        image=image)

            logging.info("Graph build successfully.")

            # Create a session for running Ops on the Graph.
            self.sess = tf.Session()
            saver = tf.train.Saver()

            # Load weights from logdir
            core.load_weights(logdir, self.sess, saver)

            logging.info("Weights loaded successfully.")

        # list of image size
        self.original_image_size = []
        self.resized_image_size = []

    def maybe_download_and_extract(self, runs_dir):
        logdir = os.path.join(runs_dir, self.default_run)

        if os.path.exists(logdir):
            # weights are downloaded. Nothing to do
            return

        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        import zipfile
        download_name = tv_utils.download(weights_url, runs_dir)

        logging.info("Extracting KittiBox_pretrained.zip")

        zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

        return

    def make_annotate(self, img, threshold):
        # Load and resize input image
        # TODO: This image value is the input of MultiNet, type:numpy.ndarray shape(375, 1242, 3)
        self.original_image_size = [img.shape[1], img.shape[0]]
        image = img.copy()
        # TODO: resized image shape(384,1248,3)
        image = scp.misc.imresize(image, (self.hypes["image_height"],
                                          self.hypes["image_width"]),
                                  interp='cubic')
        feed = {self.image_pl: image}
        self.resized_image_size = [image.shape[1], image.shape[0]]

        # Run KittiBox model on image
        pred_boxes = self.prediction['pred_boxes_new']
        pred_confidences = self.prediction['pred_confidences']

        (np_pred_boxes, np_pred_confidences) = self.sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)

        # Apply non-maximal suppression
        # and draw predictions on the image
        # TODO: use_stitching option turns off to False, because when I import stitch.so, error has come.
        # TODO: the error is "undefined symbol: _Py_ZeroStruct"
        output_image, rectangles = kittibox_utils.add_rectangles(
            self.hypes, [image], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=False, rnn_len=1,
            min_conf=0.50, tau=self.hypes['tau'], color_acc=(0, 255, 0))

        accepted_predictions = []
        accepted_predictions_int = []
        # removing predictions <= threshold
        for rect in rectangles:
            if rect.score >= threshold:
                # transform image size to inverse
                rect.x1 = rect.x1 * self.original_image_size[0] / self.resized_image_size[0]
                rect.x2 = rect.x2 * self.original_image_size[0] / self.resized_image_size[0]
                rect.y1 = rect.y1 * self.original_image_size[1] / self.resized_image_size[1]
                rect.y2 = rect.y2 * self.original_image_size[1] / self.resized_image_size[1]
                accepted_predictions.append(rect)
                accepted_predictions_int.append(((int(round(rect.x1)), int(round(rect.y1))),
                                                 (int(round(rect.x2)), int(round(rect.y2)))))

        print('')
        logging.info("{} Cars detected".format(len(accepted_predictions)))

        # Printing coordinates of predicted rects.
        for i, rect in enumerate(accepted_predictions):
            logging.info("")
            logging.info("Coordinates of Box {}".format(i))
            logging.info("    x1: {}".format(rect.x1))
            logging.info("    x2: {}".format(rect.x2))
            logging.info("    y1: {}".format(rect.y1))
            logging.info("    y2: {}".format(rect.y2))
            logging.info("    Confidence: {}".format(rect.score))

        # save output_image
        #self.save_output_image(output_image)
        return output_image, accepted_predictions_int

    def make_input_image(self):
        if self.FLAGS.input_image is None:
            logging.error("No input_image was given.")
            logging.info(
                        "Usage: python demo.py --input_image data/test.png "
                        "[--output_image output_image] [--logdir /path/to/weights] "
                        "[--gpus GPUs_to_use] ")
            exit(1)
        input_image = self.FLAGS.input_image
        logging.info("Starting inference using {} as input".format(input_image))
        return scp.misc.imread(input_image)

    def save_output_image(self, output_image):
        # save Image
        if self.FLAGS.output_image is None:
            output_name = self.FLAGS.input_image.split('.')[0] + '_rects.png'
        else:
            output_name = self.FLAGS.output_image

        scp.misc.imsave(output_name, output_image)
        logging.info("")
        logging.info("Output image saved to {}".format(output_name))

if __name__ == '__main__':
    #tf.app.run()
    a = annotate()
    a.make_annotate()
