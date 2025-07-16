from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, \
    BatchNormalization, Activation
from tensorflow.keras.models import Model


class UNetModel7030:
    def __init__(self, input_shape=(256, 256, 1), weights_path="Model/zoo/models/unet-1024.h5"):
        self.input_shape = input_shape
        self.weights_path = weights_path
        self.model = self._build_model()

    def _conv_block(self, input, num_filters, dropout_rate=0.1):
        conv = Conv2D(num_filters, (3, 3), padding="same")(input)
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        conv = Dropout(dropout_rate)(conv)

        conv = Conv2D(num_filters, (3, 3), padding="same")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        conv = Dropout(dropout_rate)(conv)
        return conv

    def _encoder_block(self, input, num_filters):
        conv = self._conv_block(input, num_filters)
        pool = MaxPooling2D((2, 2))(conv)
        return conv, pool

    def _decoder_block(self, input, skip_features, num_filters):
        uconv = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        con = concatenate([uconv, skip_features])
        conv = self._conv_block(con, num_filters)
        return conv

    def _build_model(self):
        input_layer = Input(self.input_shape)

        # Encoder
        s1, p1 = self._encoder_block(input_layer, 64)
        s2, p2 = self._encoder_block(p1, 128)
        s3, p3 = self._encoder_block(p2, 256)
        s4, p4 = self._encoder_block(p3, 512)

        # Bridge
        b1 = self._conv_block(p4, 1024)

        # Decoder
        d1 = self._decoder_block(b1, s4, 512)
        d2 = self._decoder_block(d1, s3, 256)
        d3 = self._decoder_block(d2, s2, 128)
        d4 = self._decoder_block(d3, s1, 64)

        # Output
        output_layer = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(input_layer, output_layer, name="U-Net")

        # Load weights if specified
        if self.weights_path:
            model.load_weights(self.weights_path)

        return model

    def load_model(self):
        return self.model