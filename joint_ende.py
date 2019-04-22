from keras.layers import Input, Dense
from keras.models import Model,Sequential
import Data_generate as data
import numpy as np
import tensorflow as tf
from keras import backend as K

def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))
# this is the size of our encoded representations
encoding_dim = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
k = 8
train_rate =1
x_bits = data.info_generate(k,train_rate)
# this is our input placeholder
input_img = Input(shape=(k,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(k, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

autoencoder.fit(x_bits,x_bits,epochs=100,batch_size=256,shuffle=True,metrics=[ber])

test_batch = 10
d_test = np.random.randint(0, 2, size=(test_batch, k))
for x in d_test:
    #print(x)
    encode = encoder.predict(x.reshape(1,-1))
    #print(encode)
    decode = K.round(decoder.predict(encode))
    tf.Print(decode,[decode])