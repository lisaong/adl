from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, \
    TimeDistributed, GRU, Dense


def create_model(height, width, channels, rnn_units, num_classes):
    # load pre-trained model, set to non-trainable
    base_model = MobileNetV2(input_shape=(height, width, channels), include_top=False)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)  # compute batch average
    cnn_model = Model(base_model.input, x)

    # sequential model that wraps the cnn submodel
    # input shape == (sequence, height, width, channels)
    model_input = Input(shape=(None, height, width, channels))
    x = TimeDistributed(cnn_model)(model_input)

    # pass the output to our RNN
    x = GRU(rnn_units, activation='tanh')(x)
    x = Dense(rnn_units, activation='relu')(x)

    if num_classes == 2:
        x = Dense(1, activation='sigmoid')(x)  # binary
    else:
        x = Dense(num_classes, activation='softmax')(x)  # multi-class

    rnn_model = Model(model_input, x)
    rnn_model.summary()
    return rnn_model