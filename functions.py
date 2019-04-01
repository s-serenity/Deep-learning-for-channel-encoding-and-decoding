import numpy as np
from keras import backend as K
# 二进制逆转函数
def bitrevorder(x):
    m = np.amax(x)
    #print(m)
    n = np.ceil(np.log2(m)).astype(int)
    #print(n)
    for i in range(0, len(x)):
        #print('{:0{n}b}'.format(x[i], n=n))
        x[i] = int('{:0{n}b}'.format(x[i], n=n)[::-1], 2)
    return x

def polar_channel_choose(N, k, design_snr_dB):
    S = 10 ** (design_snr_dB / 10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel
    #print(z0)
    # sort into increasing order
    idx = np.argsort(z0)
    #print(idx)
    # select k best channels
    #print(bitrevorder(idx[0:k]))
    # 巴氏参数最小的传输信息比特
    idx = np.sort(bitrevorder(idx[0:k]))
    #print(idx)
    A = np.zeros(N, dtype=bool)
    # 构建表示信道选择的列表
    A[idx] = True
    return A

def polar_generate(u):
    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):
        i = 0
        while i < N:
            for j in range(0, n):
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]
            i = i + 2 * n
        n = 2 * n
    return x

# 校验矩阵的读取
def alistToNumpy(lines):
    """Converts a parity-check matrix in AList format to a 0/1 numpy array. The argument is a
    list-of-lists corresponding to the lines of the AList format, already parsed to integers
    if read from a text file.
    The AList format is introduced on http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html.
    This method supports a "reduced" AList format where lines 3 and 4 (containing column and row
    weights, respectively) and the row-based information (last part of the Alist file) are omitted.
    Example:
        alistToNumpy([[3,2], [2, 2], [1,1,2], [2,2], [1], [2], [1,2], [1,2,3,4]])
        array([[1, 0, 1],
               [0, 1, 1]])
    """
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=np.int)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix
# 调制和噪声
def BPSK_mod(x):
    return -2*x +1

def addNoise(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w

def log_likelihood_ratio(x, sigma):
    return 2*x/np.float32(sigma**2)
