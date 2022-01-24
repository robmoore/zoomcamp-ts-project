import tensorflow as tf
from nbeats_keras.model import NBeatsNet
from tcn import TCN

from constants import FEATURES_LENGTH, INPUT_LENGTH, LABELS_LENGTH


class RepeatBaseline(tf.keras.Model):
    def __init__(self, labels_length: int):
        super(RepeatBaseline, self).__init__()
        self.labels_length = labels_length

    def call(self, inputs):
        return inputs[:, -self.labels_length :, :]


class LastBaseline(tf.keras.Model):
    def __init__(self, labels_length: int):
        super(LastBaseline, self).__init__()
        self.labels_length = labels_length

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.labels_length, 1])


def repeat_baseline() -> RepeatBaseline:
    return RepeatBaseline(LABELS_LENGTH)


def last_baseline() -> LastBaseline:
    return LastBaseline(LABELS_LENGTH)


def tcn() -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(
        TCN(
            nb_filters=100,
            kernel_size=2,
            dilations=[1, 2, 4, 8],
            dropout_rate=0.1,
            input_shape=(INPUT_LENGTH, FEATURES_LENGTH),
        )
    )
    model.add(tf.keras.layers.Dense(LABELS_LENGTH))

    return model


def tuned_tcn() -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(
        TCN(
            nb_filters=224,
            kernel_size=7,
            dropout_rate=0.1,
            input_shape=(INPUT_LENGTH, FEATURES_LENGTH),
        )
    )
    model.add(tf.keras.layers.Dense(LABELS_LENGTH))

    return model


def nbeats() -> NBeatsNet:
    return NBeatsNet(
        backcast_length=INPUT_LENGTH,
        forecast_length=LABELS_LENGTH,
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        nb_blocks_per_stack=2,
        hidden_layer_units=100,
    )


def lstm():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(INPUT_LENGTH, FEATURES_LENGTH)))
    model.add(tf.keras.layers.Dense(LABELS_LENGTH))

    return model


def tcn_for_tuning(hp):
    model = tf.keras.Sequential()
    model.add(
        TCN(
            nb_filters=hp.Int("nb_filters", min_value=64, max_value=256, step=32),
            kernel_size=hp.Int("kernel_size", min_value=2, max_value=8),
            dropout_rate=hp.Choice("dropout_rate", values=[0.1, 0.05, 0.0]),
            input_shape=(INPUT_LENGTH, FEATURES_LENGTH),
        )
    )
    model.add(tf.keras.layers.Dense(LABELS_LENGTH))

    return model
