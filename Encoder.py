import numpy as np
import functions
import pyldpc as ldpc

class LDPC:
    def __init__(self, N, K,matrix_H):
        self.N = N
        self.K = K
        self.H_matrix = matrix_H
        _ ,self.G_matrix = ldpc.CodingMatrix_systematic(self.H_matrix)

    # 将文件内容转换为矩阵
    '''
        def init_LDPC_H(self,file_H):
        with open(file_H) as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                new_lines.append(list(map(int, line.split())))
            H_matrix = functions.alistToNumpy(new_lines)
        return H_matrix
    '''

    #使用生成矩阵进行编码
    def encode(self,x_bits):
        # coding
        matrix_G = self.G_matrix.transpose()
        u_coded_bits = np.mod(np.matmul(x_bits,matrix_G), 2)  # G_matrix
        return u_coded_bits

class Polar:
    def __init__(self,N,K,design_snr_dB):
        self.N = N
        self.K = K
        self.design_snr_dB = design_snr_dB

    def encode(self,x_bits):
        codesize = x_bits.shape[0]
        infos = functions.polar_channel_choose(self.N, self.K, self.design_snr_dB)
        # 建立指定大小的编码集合
        u = np.zeros((codesize, self.N),dtype = bool)
        u_coded_bits = np.zeros((codesize, self.N),dtype = bool)
        u[:,infos] = x_bits
        for i in range(0,codesize):
            u_coded_bits[i] = functions.polar_generate(u[i])
        return u_coded_bits

