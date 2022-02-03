from tensorflow import keras


class Assembler(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def call(self,input_state):
        h = self.encoder(input_state)
        h = self.decoder(h)
        return h
