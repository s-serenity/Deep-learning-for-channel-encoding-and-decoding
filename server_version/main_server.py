import sys
import configuration as config
import numpy as np
import Encoder as E
import functions as F
import Decoder as D
import Data_generate as data
from numpy.random import RandomState
import matplotlib.pyplot as plt
import pyldpc as ldpc
import time
import tensorflow as tf
'''
top = config.TopConfig()
polar = config.Polarconfig()
LDPC = config.LDPCconfig()
MLP = config.MLPconfig()
RNN = config.RNNconfig()
TCN = config.TCNconfig()
Train = config.Trainconfig()
Test = config.Testconfig()
'''
# paras = [top,Polar,LDPC,Train,Test,MLP,RNN,TCN]
top,Polar,LDPC,Train,Test,MLP,RNN,TCN = config.get_paras(sys.argv)
# Data generate
print('---------information generating begin-----------')
before = time.time()
x_bits = data.info_generate(top.K_code,Train.train_rate)
# 编码
print('---------encoding begin-----------')
code = top.code
k = top.K_code
N = top.N_code
if top.code == 'LDPC':
    matrix_H = ldpc.RegularH(16, 1, 2)
    in_code = E.LDPC(N,k,matrix_H)
    symbols = in_code.encode(x_bits)
elif top.code == 'polar':
    in_code = E.Polar(N,k,Polar.design_snr_db)
    symbols = in_code.encode(x_bits)
after = time.time()
print('cost:',after-before)
# 训练
print('---------training begin-----------')
before = time.time()
decode_choose = top.decode
print(symbols.shape)
# Modulator (BPSK)
# signals = F.BPSK_mod(symbols)
# Channel (AWGN)
#r = RandomState(1234567890)
#train_sigma = Train.train_sigma
# noise_symbols = F.addNoise(signals,train_sigma)
#with tf.Session():
    #noise_symbols = noise_symbols.eval()
decoder = D.train_network(top,Train,MLP,RNN,TCN,symbols,x_bits)
after = time.time()
print('cost:',after-before)
print('---------testing begin-----------')
before = time.time()
sigmas = Test.sigmas
nb_errors = np.zeros(len(sigmas), dtype=int)
nb_bits = np.zeros(len(sigmas), dtype=int)
nb_bers = np.zeros(len(sigmas), dtype=float)
decode_choose = top.decode
# Source
s = RandomState(0)
for i in range(0, len(sigmas)):
    for ii in range(0, np.round(Test.test_samples / Test.test_batch).astype(int)):
        d_test = np.random.randint(0, 2, size=(Test.test_batch, k))
        # Encoder
        x_test = in_code.encode(d_test)
        # Modulator (BPSK)
        test_signals = F.BPSK_mod(x_test)
        # Channel (AWGN)
        y_test = test_signals + sigmas[i] * np.random.standard_normal(test_signals.shape)
        # Decoder
        if decode_choose == 'RNN' or decode_choose == 'LSTM' or decode_choose == 'TCN':
            y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[1])
            d_test = d_test.reshape(d_test.shape[0],1,d_test.shape[1])
        #nb_bers[i] = decoder.evaluate(y_test, d_test, batch_size=Test.test_batch, verbose=0)[1]
        nb_errors[i] += decoder.evaluate(y_test, d_test, batch_size=Test.test_batch, verbose=0)[1]
        nb_bits[i] += d_test.size
after = time.time()
print('cost:',after-before)
legend = []
plt.plot(10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N), nb_errors/nb_bits)
#plt.plot(10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N), nb_bers)
legend.append('NN')
plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')
plt.grid(True)
plt.savefig('./results/'+str(k) + '_' + str(N) + top.code+'_'+decode_choose+'_ber.png')
plt.show()