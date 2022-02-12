from tensorflow import keras
import numpy


class Assembler(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, input_state):
        h = self.encoder(input_state)
        h = self.decoder(h)
        return h

    def save_model(self, path):
        encoder_weights = self.encoder.get_weights()
        decoder_weights = self.decoder.get_weights()
        numpy.save(f"{path}/encoder",encoder_weights)
        numpy.save(f"{path}/decoder",decoder_weights)

    def load_model(self, path):
        encoder_weights = numpy.load(f"{path}/encoder.npy", allow_pickle=True)
        self.encoder.set_weights(encoder_weights)
        decoder_weights = numpy.load(f"{path}/decoder.npy", allow_pickle=True)
        self.decoder.set_weights(decoder_weights)
        
