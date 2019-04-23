from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
import Data_generate as data
import numpy as np
import tensorflow as tf
from keras import backend as K
import functions as F
import matplotlib.pyplot as plt


def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))


def testber(y_true, y_pred):
    # y_pred = tf.cast(K.round(y_pred),dtype=tf.int32)
    y_pred = K.cast(K.round(y_pred), dtype='int64')
    a = K.not_equal(y_true, y_pred)
    return K.mean(a)


# this is the size of our encoded representations
def return_output_shape(input_shape):
    return input_shape


def normalize(x):
    x_mean, x_var = tf.nn.moments(x, [0])
    x = (x - x_mean) * 1.0 / tf.sqrt(x_var)
    return x

def errors(y_true, y_pred):
    #y_pred = K.cast(K.round(y_pred), dtype='int32')
    #print(y_true.dtype)
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)),dtype='float16'))

def count_errors(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), dtype='int64')
    #print(y_true.dtype)
    return K.sum(K.cast(K.not_equal(y_true,y_pred),dtype='float16'))

N = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

k = 8
train_rate = 1
x_bits = data.info_generate(k, train_rate)
# this is our input placeholder
input_img = Input(shape=(k,))
train_SNR_Es = 1  # training-Eb/No
As = 1
train_sigma = 10 ** (-train_SNR_Es * 1.0 / 20) * As
# "encoded" is the encoded representation of the input
encoded = Dense(64, activation='relu')(input_img)
encoded0 = Dense(128, activation='relu')(encoded)
encoded1 = Dense(256, activation='relu')(encoded0)
encoded2 = Dense(N, activation='sigmoid')(encoded1)
norm_encoded = Lambda(normalize)(encoded2)
noise_layers = Lambda(F.addNoise, arguments={'sigma': train_sigma}, input_shape=(N,), output_shape=return_output_shape,
                      name="noise")
noised = noise_layers(norm_encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(256, activation='relu')(noised)
decoded0 = Dense(128, activation='relu')(decoded)
decoded1 = Dense(64, activation='relu')(decoded0)
decoded2 = Dense(k, activation='sigmoid')(decoded1)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded2)
encoder = Model(input_img, norm_encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(N,))
x = encoded_input
# retrieve the last layer of the autoencoder model
decoder_layers = autoencoder.layers[-4:]
for layer in decoder_layers:
    x = layer(x)
y = x
# create the decoder model
decoder = Model(encoded_input, y)
decoder.compile(optimizer='adam', loss='mse', metrics=[ber])
autoencoder.compile(optimizer='adam', loss='mse', metrics=[ber])
autoencoder.summary()
history = autoencoder.fit(x_bits, x_bits, epochs=2**N, batch_size=256, shuffle=True, verbose=0)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

test_batch = 10000
SNR_dB_start_Es = 0
SNR_dB_stop_Es = 5
SNR_points = 10

sigma_start = 10 ** (-SNR_dB_start_Es * 1.0 / 20)
sigma_stop = 10 ** (-SNR_dB_stop_Es * 1.0 / 20)
SNRs = np.linspace(SNR_dB_start_Es, SNR_dB_stop_Es, SNR_points)
sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)
bers = np.zeros((len(sigmas),1))
for z in range(len(sigmas)):
    d_test = np.random.randint(0, 2, size=(test_batch, k))
    encodes = encoder.predict(d_test)
    #print('----')
    noise_test = encodes + sigmas[z] * np.random.standard_normal(encodes.shape)
    # noise_test = F.addNoise(encodes,sigmas[z])
    y_pred = decoder.predict(noise_test)
    #print('----')
    result = testber(d_test, y_pred)
    with tf.Session():
        result = result.eval()
    print(result)
    bers[z] = result
    # results = autoencoder.evaluate(d_test,d_test)[1]
    '''
    results = decoder.evaluate(noise_test,d_test)
    print(results)
    errors[z] = decoder.evaluate(noise_test,d_test)
    bers[z] = errors[z]/d_test.size
    '''
legend = []
plt.plot(SNRs, bers)
legend.append('NN')
plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_s/N_0$')
plt.ylabel('BER')
plt.grid(True)
plt.show()
#print(ber)
