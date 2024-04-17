from PIL import Image as img

from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import copy
import pickle
import tifffile as tiff

from matplotlib import pyplot as plt

class StarNet():
    def __init__(self, mode:str, window_size:int = 512, stride:int = 256, lr:float = 1e-4, train_folder:str = './train/', batch_size:int = 1):
        assert mode in ['RGB', 'Greyscale'], "Mode should be either RGB or Greyscale"
        self.mode = mode
        if self.mode == 'RGB': self.input_channels = 3
        else: self.input_channels = 1
        self.window_size = window_size
        self.stride = stride
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.history = {}
        self._ema = 0.9999
        self.weights = []
        self.lr = lr

        self.original = []
        self.starless = []

    def __str__(self):
        return "Starnet instance"

    def load_model(self, weights = None, history = None):
        self.G = self._generator(m = 64)
        self.D = self._discriminator()

        self.gen_optimizer = tf.optimizers.Adam(self.lr)
        self.dis_optimizer = tf.optimizers.Adam(self.lr / 4)

        self.D.build(input_shape = (None, self.window_size, self.window_size, self.input_channels))
        self.G.build(input_shape = (None, self.window_size, self.window_size, self.input_channels))

        #if weights: self.G.load_weights(weights + '_' + self.mode + '.h5')
        if weights:
            self.G.load_weights(weights + '_G_' + self.mode + '.h5')
            self.D.load_weights(weights + '_D_' + self.mode + '.h5')
        if history:
            with open(history + '_' + self.mode + '.pkl', "rb") as h:
                self.history = pickle.load(h)

    def initialize_model(self):
        self.load_model()

    def _ramp(self, x):
        return tf.clip_by_value(x, 0, 1)

    def _augmentator(self, o, s):
        # flip horizontally
        if np.random.rand() < 0.50:
            o = np.flip(o, axis = 1)
            s = np.flip(s, axis = 1)

        # flip vertically
        if np.random.rand() < 0.50:
            o = np.flip(o, axis = 0)
            s = np.flip(s, axis = 0)

        # rotate 90, 180 or 270
        if np.random.rand() < 0.50:
            k = int(np.random.rand() * 3 + 1)
            o = np.rot90(o, k, axes = (1, 0))
            s = np.rot90(s, k, axes = (1, 0))

        if self.mode == 'RGB':
            # tweak colors
            if np.random.rand() < 0.70:
                ch = int(np.random.rand() * 3)
                m = np.min((o, s))
                offset = np.random.rand() * 0.25 - np.random.rand() * m
                o[:, :, ch] = o[:, :, ch] + offset * (1.0 - o[:, :, ch])
                s[:, :, ch] = s[:, :, ch] + offset * (1.0 - s[:, :, ch])

            # flip channels
            if np.random.rand() < 0.70:
                seq = np.arange(3)
                np.random.shuffle(seq)
                Xtmp = np.copy(o)
                Ytmp = np.copy(s)
                for i in range(3):
                    o[:, :, i] = Xtmp[:, :, seq[i]]
                    s[:, :, i] = Ytmp[:, :, seq[i]]
        else:
            # tweak brightness
            if np.random.rand() < 0.70:
                m = np.min((o, s))
                offset = np.random.rand() * 0.25 - np.random.rand() * m
                o[:, :] = o[:, :] + offset * (1.0 - o[:, :])
                s[:, :] = s[:, :] + offset * (1.0 - s[:, :])

        o = np.clip(o, 0.0, 1.0)
        s = np.clip(s, 0.0, 1.0)

        if self.mode == 'RGB': return o, s
        else:
            c = np.random.randint(3)
            return o[:, :, c, None], s[:, :, c, None]

    def _get_sample(self, r, h, w, type:str):
        assert type in ['original', 'starless']
        if type == 'original':
            return self.original[r][h:h+self.window_size, w:w+self.window_size] / 255
        else:
            return self.starless[r][h:h+self.window_size, w:w+self.window_size] / 255

    def transform(self, in_name, out_name):
        data = tiff.imread(in_name)
        if len(data.shape) > 3:
            layer = input("Tiff has %d layers, please enter layer to process: "%data.shape[0])
            layer = int(layer)
            data=data[layer]

        input_dtype = data.dtype
        if input_dtype == 'uint16':
            image = (data / 255.0 / 255.0).astype('float32')
        elif input_dtype == 'uint8':
            image = (data / 255.0).astype('float32')
        else:
            raise ValueError('Unknown image dtype:', data.dtype)

        if self.mode == 'Greyscale' and len(image.shape) == 3:
            raise ValueError('You loaded Greyscale model, but the image is RGB!')

        if self.mode == 'Greyscale':
            image = image[:, :, None]

        if self.mode == 'RGB' and len(image.shape) == 2:
            raise ValueError('You loaded RGB model, but the image is Greyscale!')

        if self.mode == 'RGB' and image.shape[2] == 4:
            print("Input image has 4 channels. Removing Alpha-Channel")
            image=image[:,:,[0,1,2]]

        offset = int((self.window_size - self.stride) / 2)

        h, w, _ = image.shape

        ith = int(h / self.stride) + 1
        itw = int(w / self.stride) + 1

        dh = ith * self.stride - h
        dw = itw * self.stride - w

        image = np.concatenate((image, image[(h - dh) :, :, :]), axis = 0)
        image = np.concatenate((image, image[:, (w - dw) :, :]), axis = 1)

        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset) :, :, :]), axis = 0)
        image = np.concatenate((image[: offset, :, :], image), axis = 0)
        image = np.concatenate((image, image[:, (w - offset) :, :]), axis = 1)
        image = np.concatenate((image[:, : offset, :], image), axis = 1)

        image = image * 2 - 1

        output = copy.deepcopy(image)

        for i in range(ith):
            for j in range(itw):
                x = self.stride * i
                y = self.stride * j

                tile = np.expand_dims(image[x:x+self.window_size, y:y+self.window_size, :], axis = 0)
                tile = (self.G(tile)[0] + 1) / 2
                tile = tile[offset:offset+self.stride, offset:offset+self.stride, :]
                output[x+offset:self.stride*(i+1)+offset, y+offset:self.stride*(j+1)+offset, :] = tile

        output = np.clip(output, 0, 1)

        if self.mode == 'Greyscale':
            output = output[offset:-(offset+dh), offset:-(offset+dw), 0]
        else:
            output = output[offset:-(offset+dh), offset:-(offset+dw), :]

        if input_dtype == 'uint8':
            tiff.imsave(out_name, (output * 255).astype('uint8'))
        else:
            tiff.imsave(out_name, (output * 255 * 255).astype('uint16'))

    def _generator(self, m):
        layers = []

        filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64]

        input = L.Input(shape=(self.window_size, self.window_size, self.input_channels), name = "gen_input_image")

        # layer 0
        convolved = L.Conv2D(filters[0], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(input)
        layers.append(convolved)

        # layer 1
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[1], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 2
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[2], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 3
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[3], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 4
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[4], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 5
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[5], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 6
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[6], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 7
        rectified = L.LeakyReLU(alpha = 0.2)(layers[-1])
        convolved = L.Conv2D(filters[7], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(convolved, training = True)
        layers.append(normalized)

        # layer 8
        rectified = L.ReLU()(layers[-1])
        deconvolved = L.Conv2DTranspose(filters[8], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 9
        concatenated = tf.concat([layers[-1], layers[6]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[9], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 10
        concatenated = tf.concat([layers[-1], layers[5]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[10], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 11
        concatenated = tf.concat([layers[-1], layers[4]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[11], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 12
        concatenated = tf.concat([layers[-1], layers[3]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[12], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 13
        concatenated = tf.concat([layers[-1], layers[2]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[13], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 14
        concatenated = tf.concat([layers[-1], layers[1]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(filters[14], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        normalized = L.BatchNormalization()(deconvolved, training = True)
        layers.append(normalized)

        # layer 15
        concatenated = tf.concat([layers[-1], layers[0]], axis = 3)
        rectified = L.ReLU()(concatenated)
        deconvolved = L.Conv2DTranspose(self.input_channels, kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
        rectified = L.ReLU()(deconvolved)
        output = tf.math.subtract(input, rectified)

        return K.Model(inputs = input, outputs = output, name = "generator")

    def _discriminator(self):
        layers = []
        filters = [32, 64, 64, 128, 128, 256, 256, 256, 8]

        input = L.Input(shape=(self.window_size, self.window_size, self.input_channels), name = "dis_input_image")

        # layer 1
        convolved = L.Conv2D(filters[0], kernel_size = 3, strides = (1, 1), padding="same")(input)
        rectified = L.LeakyReLU(alpha = 0.2)(convolved)
        layers.append(rectified)

        # layer 2
        convolved = L.Conv2D(filters[1], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 3
        convolved = L.Conv2D(filters[2], kernel_size = 3, strides = (1, 1), padding="same")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 4
        convolved = L.Conv2D(filters[3], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 5
        convolved = L.Conv2D(filters[4], kernel_size = 3, strides = (1, 1), padding="same")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 6
        convolved = L.Conv2D(filters[5], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 7
        convolved = L.Conv2D(filters[6], kernel_size = 3, strides = (1, 1), padding="same")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 8
        convolved = L.Conv2D(filters[7], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 9
        convolved = L.Conv2D(filters[8], kernel_size = 3, strides = (2, 2), padding="valid")(layers[-1])
        normalized = L.BatchNormalization()(convolved, training = True)
        rectified = L.LeakyReLU(alpha = 0.2)(normalized)
        layers.append(rectified)

        # layer 10
        dense = L.Dense(1)(layers[-1])
        sigmoid = tf.nn.sigmoid(dense)
        layers.append(sigmoid)

        output = [layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[-1]]

        return K.Model(inputs = input, outputs = output, name = "discriminator")
