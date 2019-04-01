import numpy as np
import Encoder as E
import functions as F
import Decoder as D
import Data_generate as data
import matplotlib.pyplot as plt
import pyldpc as ldpc
import time
# 编码部分的设置
k = 8                     # number of information bits
N = 16                     # code length
code = 'LDPC'              # type of code ('LDPC' or 'polar')
design_snr_db = 0           # design snr of polar code in dB unit
file_H = format('./LDPC_matrix/LDPC_chk_mat_%d_%d.alist' % (N, k))                 # LDPC校验矩阵的文件地址
# 解码部分的设置
decode_choose = 'RNN'
# 神经网络训练参数的设置
nb_epoch = 2**16            # number of learning epochs
design = [512, 256, 128]      # each list entry defines the number of nodes in a layer
train_batch_size = 256            # size of batches for calculation the gradient
train_SNR_Eb = 1            # training-Eb/No
train_SNR_Es = train_SNR_Eb + 10*np.log10(k/N)
train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))
train_rate = 1              # define how many of all codewords used to train the network
LLR = False                 # 'True' enables the log-likelihood-ratio layer
optimizer = 'adam'
loss = 'mse'                # or 'binary_crossentropy'

'''
test_SNR_Eb = 1             # testing-Eb/No
test_SNR_Es = test_SNR_Eb + 10*np.log10(k/N)
test_sigma = np.sqrt(1/(2*10**(test_SNR_Es/10)))
'''
test_samples = 100000
test_batch = 1000
SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 5
SNR_points = 20

SNR_dB_start_Es = SNR_dB_start_Eb + 10 * np.log10(k / N)
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10 * np.log10(k / N)

sigma_start = np.sqrt(1 / (2 * 10 ** (SNR_dB_start_Es / 10)))
sigma_stop = np.sqrt(1 / (2 * 10 ** (SNR_dB_stop_Es / 10)))

# 决定产生样本的随机种子
# 这是用来产生随机样本进行训练的
print('---------encoding begin-----------')
before = time.time()
x_bits = data.info_generate(k,train_rate)
# 编码
if code == 'LDPC':
    matrix_H = ldpc.RegularH(16, 1, 2)
    in_code = E.LDPC(N,k,matrix_H)
    symbols = in_code.encode(x_bits)
elif code == 'polar':
    in_code = E.Polar(N,k,design_snr_db)
    symbols = in_code.encode(x_bits)
after = time.time()
print('cost:',after-before)
# 训练网络
print('---------training begin-----------')
before = time.time()
if decode_choose == 'MLP':
    decoder = D.train_network_MLP(code,symbols,x_bits,N,k,train_sigma,LLR,design,train_batch_size,nb_epoch,optimizer,loss)
elif decode_choose == 'RNN':
    decoder = D.train_network_RNN(code,symbols,x_bits, N, k, train_sigma, 'gru', 2,200,train_batch_size, nb_epoch,optimizer,
                                  loss)
elif decode_choose == 'LSTM':
    decoder = D.train_network_RNN(code,symbols, x_bits, N, k, train_sigma, 'lstm',2,200,train_batch_size, nb_epoch,optimizer,
                                  loss)
after = time.time()
print('cost:',after-before)
# 测试网络
sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

nb_errors = np.zeros(len(sigmas), dtype=int)
nb_bits = np.zeros(len(sigmas), dtype=int)
print('---------testing begin-----------')
before = time.time()
for i in range(0, len(sigmas)):

    for ii in range(0, np.round(test_samples / test_batch).astype(int)):
        # Source
        np.random.seed(0)
        d_test = np.random.randint(0, 2, size=(test_batch, k))
        # Encoder
        x_test = in_code.encode(d_test)
        # Modulator (BPSK)
        signals = F.BPSK_mod(x_test)
        # Channel (AWGN)
        y_test = signals + sigmas[i] * np.random.standard_normal(signals.shape)
        if LLR:
            y_test = 2 * y_test / (sigmas[i] ** 2)
        # Decoder
        if decode_choose == 'RNN' or decode_choose == 'LSTM':
            y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[1])
            d_test = d_test.reshape(d_test.shape[0],1,d_test.shape[1])
        nb_errors[i] += decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=0)[1]
        nb_bits[i] += d_test.size
after = time.time()
print('cost:',after-before)
print(10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N))
print(nb_errors/nb_bits)
legend = []
plt.plot(10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/N), nb_errors/nb_bits)
legend.append('NN')
plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')
plt.grid(True)
plt.show()