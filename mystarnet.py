import tensorflow as tf
from PIL import Image as img
import logging
tf.get_logger().setLevel(logging.ERROR)
from starnet_v1_TF2 import StarNet
import tifffile as tiff
import sys

if len(sys.argv) > 1:
    if sys.argv[1] == '--version':
        resume = False
        print("")
        print("starnet1  version: 1.0.0")
        print("")
        sys.stdout.flush()
    else:
        starnet = StarNet(mode = 'RGB', window_size = 512, stride = 128)
        starnet.load_model('./weights', './history')
        print("Weights Loaded!")
        in_name = sys.argv[2]
        out_name = sys.argv[3]
        starnet.transform(in_name, out_name)
