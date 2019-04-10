import numpy as np

# top layer setting, including which code to encode and which network to decode, is to train or test
class TopConfig:
    def __init__(self):
        # select functions to be executed, including training and testing
        self.function = 'Train'
        # choose code and set code rate
        self.code = 'polar'
        self.N_code = 16
        self.K_code = 8
        # choose network to decode
        self.decode = 'TCN'

class Polarconfig:
    def __init__(self,TopConfig):
        self.N_code = TopConfig.N_code
        self.K_code = TopConfig.K_code
        self.design_snr_db = 0

class LDPCconfig:
    def __init__(self,TopConfig):
        self.N_code = TopConfig.N_code
        self.K_code = TopConfig.K_code
        self.H_file = format('./LDPC_matrix/LDPC_chk_mat_%d_%d.alist' % (self.N_code, self.K_code))

class MLPconfig:
    def __init__(self):
        self.layernum = 3
        self.dims = [128,64,32]
        self.LLR = False

class RNNconfig:
    def __init__(self):
        # GRU or LSTM
        self.layers = 'GRU'
        self.num_layer = 2
        self.num_unit = 200

class TCNconfig:
    def __init__(self):
        #self.nb_filters =
        self.num_channels = [70,80]
        self.kernel_size = 2
        self.dropout_rate = 0

class Trainconfig:
    def __init__(self,TopConfig):
        self.train_SNR_Eb = 1  # training-Eb/No
        self.train_rate = 1.0
        self.train_SNR_Es = self.train_SNR_Eb + 10 * np.log10(TopConfig.K_code / TopConfig.N_code)
        self.train_sigma = np.sqrt(1 / (2 * 10 ** (self.train_SNR_Es / 10)))
        self.batch_size = 256
        self.nb_epoch = 2**TopConfig.N_code
        self.optimizer = 'adam'
        self.loss = 'mse'

class Testconfig:
    def __init__(self,TopConfig):
        self.test_samples = 100000
        self.test_batch = 1000
        self.SNR_dB_start_Eb = 0
        self.SNR_dB_stop_Eb = 5
        self.SNR_points = 20
        self.SNR_dB_start_Es = self.SNR_dB_start_Eb + 10 * np.log10(TopConfig.K_code / TopConfig.N_code)
        self.SNR_dB_stop_Es = self.SNR_dB_stop_Eb + 10 * np.log10(TopConfig.K_code / TopConfig.N_code)
        self.sigma_start = np.sqrt(1 / (2 * 10 ** (self.SNR_dB_start_Es / 10)))
        self.sigma_stop = np.sqrt(1 / (2 * 10 ** (self.SNR_dB_stop_Es / 10)))
        self.sigmas = np.linspace(self.sigma_start, self.sigma_stop, self.SNR_points)

def get_paras(argv):
    top = TopConfig()
    Polar = Polarconfig(top)
    LDPC = LDPCconfig(top)
    Train = Trainconfig(top)
    Test = Testconfig(top)
    MLP = MLPconfig()
    RNN = RNNconfig()
    TCN = TCNconfig()
    i = 1
    while i < len(argv):
        if argv[i] == '-Func':
            top.function = argv[i + 1]
            print('Function is set to %s' % argv[i + 1])
        # choose code
        elif argv[i] == '-code':
            top.code = argv[i + 1]
            print('Code is set to %s' % argv[i + 1])
        # set code rate
        elif argv[i] == '-decode':
            top.decode = argv[i + 1]
            print('Decoder is set to %s' % argv[i + 1])
        elif argv[i] == '-k':
            top.K_code = int(argv[i + 1])
            print('information bit k is set to %d' % argv[i + 1])
        elif argv[i] == '-N':
            top.N_code = int(argv[i + 1])
            print('symbol bit N is set to %d' % argv[i + 1])
        # polar code and LDPC configuration
        elif argv[i] == '-design_snr_db':
            Polar.design_snr_db = float(argv[i + 1])
            print('design_snr_db of polar code is set to %f' % float(argv[i + 1]))
        elif argv[i] == '-H_file':
            LDPC.H_file = format('./LDPC_matrix/LDPC_chk_mat_%d_%d.alist' % (argv[i + 1], argv[i + 2]))
            print('H_file is set to %s' % str(LDPC.H_file))
        # train setting
        elif argv[i] == '-trainrate':
            Train.train_rate = float(argv[i + 1])
            print('Trainning batchsize is set to: %.1f' % argv[i + 1])
        elif argv[i] == '-batchsize':
            Train.batch_size = int(argv[i + 1])
            print('Trainning batchsize is set to: %d' % argv[i + 1])
        elif argv[i] == '-loss':
            Train.loss = argv[i + 1]
            print('Trainning loss is set to: %s' % argv[i + 1])
        elif argv[i] == '-epoch':
            Train.nb_epoch = int(argv[i + 1])
            print('Trainning epoch is set to: %d' % argv[i + 1])
        elif argv[i] == '-train_SNR_Eb':
            Train.train_SNR_Eb = float(argv[i + 1])
            print('Trainning SNR Eb is set to: %.1f' % argv[i + 1])
        # test config
        elif argv[i] == '-test_SNR':
            Test.SNR_dB_start_Eb = float(argv[i + 1])
            Test.SNR_dB_stop_Eb = float(argv[i + 2])
            Test.SNR_points = int(argv[i + 3])
            print('Test SNR is set to:begin from %.1f to %.1f for %d points'%(argv[i + 1],argv[i + 2],argv[i + 3]))
        elif argv[i] == '-testSamples':
            Test.test_samples = int(argv[i+1])
            print('Test samples is set to: %d' % argv[i + 1])
        elif argv[i] == '-testbatch':
            Test.test_batch = int(argv[i+1])
            print('Test batch is set to: %d' % argv[i + 1])
        # MLP config
        elif argv[i] == '-LLR':
            MLP.LLR = argv[i + 1]
            print('LLR is set to: %d' % argv[i + 1])
        elif argv[i] == '-MLPlayernum':
            MLP.layernum = int(argv[i + 1])
            print('MLP layer number is set to %d' % MLP.layernum)
        elif argv[i] == '-MLPdims':
            MLP.dims = np.fromstring(argv[i + 1], np.float32, sep=' ')
            print('MLP dims are set to %s' % np.array2string(MLP.dims))
        # RNN config
        elif argv[i] == '-G_L':
            RNN.layers = argv[i+1]
            print('RNN layer is set to %s' % argv[i+1])
        elif argv[i] == '-RNNlayernum':
            RNN.num_layer = int(argv[i+1])
            print('RNN layer number is set to %d'%argv[i+1])
        elif argv[i] == '-RNNunit':
            RNN.num_unit = int(argv[i+1])
            print('RNN unit number is set to %d'%argv[i+1])
        # TCN config
        elif argv[i] == '-TCNfilter':
            TCN.nb_filters = int(argv[i + 1])
            print('TCN filter is set to %d' % TCN.nb_filters)
        elif argv[i] == '-TCNkernel':
            TCN.kernel_size = int(argv[i + 1])
            print('TCN kernel size is set to: %sd' % TCN.kernel_size)
        elif argv[i] == '-dprate':
            TCN.dropout_rate = float(argv[i + 1])
            print('TCN dropout rate is set to %f' % TCN.dropout_rate)
        else:
            print('Invali parameter %s!' % argv[i])
            exit(0)
        i = i + 2
    return top,Polar,LDPC,Train,Test,MLP,RNN,TCN


