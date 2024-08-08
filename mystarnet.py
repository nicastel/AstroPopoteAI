import tensorflow as tf
from PIL import Image as img
import logging
tf.get_logger().setLevel(logging.ERROR)
from starnet_v1_TF2 import StarNet
import tifffile as tiff
import sys
from numba import cuda

if len(sys.argv) > 1:
    if sys.argv[1] == '--version':
        resume = False
        print("")
        print("starnet1  version: 1.0.0")
        print("")
        sys.stdout.flush()
    else:
        # -i input.tif -o starless_imput.tif
        print("Starnet TensorFlow 2 - Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        starnet = StarNet(mode = 'RGB', window_size = 512, stride = 128)
        starnet.load_model('./weights', './history')
        print("Weights Loaded!")
        in_name = sys.argv[2]
        out_name = sys.argv[4]
        starnet.transform(in_name, out_name)
        if cuda.is_available:
            device = cuda.get_current_device()
            device.reset()
        print("100% finished")
