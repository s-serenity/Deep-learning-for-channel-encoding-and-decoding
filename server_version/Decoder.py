import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras import backend as K
import functions as F
from keras.layers import Input,BatchNormalization,LSTM,GRU, Dense, TimeDistributed, Lambda,SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tcn import TCN
#from tftcn import TemporalBlock
from keras.models import load_model

def compose_model(layers):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model

def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))

def errors(y_true, y_pred):
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)),dtype='float16'))

def return_output_shape(input_shape):
    return input_shape

# 搭建训练网络
# 参数：N：码长，k:信息长，train_sigma：训练时的SNR，LLR：是否使用LLR层，design:全连接层的维数列表
# 生于参数为训练网络的超参数
def train_network(top,train,MLP,RNN,TCNconfig,x,d):
    decode_choose = top.decode
    k = top.K_code
    N = top.N_code
    modulator_layers = [Lambda(F.BPSK_mod,
                               input_shape=(N,), output_shape=return_output_shape, name="modulator")]
    # Define noise
    noise_layers = [Lambda(F.addNoise, arguments={'sigma': train.train_sigma},
                           input_shape=(N,), output_shape=return_output_shape, name="noise")]
    if decode_choose == 'MLP':
        design = MLP.dims
        decoder_layers = [Dense(design[0], activation='relu', input_shape=(N,))]
        for i in range(1, len(design)):
            decoder_layers.append(Dense(design[i], activation='relu'))
        decoder_layers.append(Dense(k, activation='sigmoid'))
        decoder = compose_model(decoder_layers)
    elif decode_choose == 'RNN' or decode_choose == 'LSTM':
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        d = d.reshape((d.shape[0], 1, d.shape[1]))
        ont_pretrain_trainable = True
        dropout_rate = 1
        for layer in range(RNN.num_layer):
            if RNN.layers == 'gru':
                if layer == 0:
                    decoder_layers = [Bidirectional(GRU(units=RNN.num_unit, activation='tanh', dropout=dropout_rate,
                                                        return_sequences=True, input_shape=(N, 1),
                                                        trainable=ont_pretrain_trainable),
                                                    name='Dec_' + RNN.layers + '_' + str(layer))]
                else:
                    decoder_layers.append(Bidirectional(GRU(units=RNN.num_unit, activation='tanh', dropout=dropout_rate,
                                                            return_sequences=True, input_shape=(N, 1),
                                                            trainable=ont_pretrain_trainable),
                                                        name='Dec_' + RNN.layers + '_' + str(layer)))
            elif RNN.layers == 'lstm':
                if layer == 0:
                    decoder_layers = [Bidirectional(LSTM(units=RNN.num_unit, activation='tanh', dropout=dropout_rate,
                                                        return_sequences=True, input_shape=(N, 1),
                                                        trainable=ont_pretrain_trainable),
                                                    name='Dec_' + RNN.layers + '_' + str(layer))]
                else:
                    decoder_layers.append(Bidirectional(LSTM(units=RNN.num_unit, activation='tanh', dropout=dropout_rate,
                                                            return_sequences=True, input_shape=(N, 1),
                                                            trainable=ont_pretrain_trainable),
                                                        name='Dec_' + RNN.layers + '_' + str(layer)))
            decoder_layers.append(BatchNormalization(name='Dec_bn' + '_' + str(layer), trainable=ont_pretrain_trainable))
        # decoder_layers = [SimpleRNN(units=num_Dec_unit, activation='tanh', dropout=dropout_rate,return_sequences=True,input_shape=(1,N), trainable=ont_pretrain_trainable)]
        decoder_layers.append(Dense(k, activation='sigmoid'))
        decoder = compose_model(decoder_layers)
    elif decode_choose == 'TCN':
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        # d = d.reshape((d.shape[0], 1, d.shape[1]))
        '''
        code_input = Input(shape=(1, N), dtype='float32', name='main_input')
        out = TCN(nb_filters=TCNconfig.nb_filters, kernel_size=TCNconfig.kernel_size, dropout_rate=TCNconfig.dropout_rate,return_sequences=False)(code_input)
        final_out = Dense(k, activation='sigmoid')(out)
        decoder = Model(inputs=[code_input], outputs=[final_out])
        '''
        decoder_layers = []
        decoder = tf.keras.Sequential()
        num_levels = len(TCNconfig.num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i  # exponential growth
            decoder.add(TemporalBlock(dilation_rate, TCNconfig.num_channels[i], TCNconfig.kernel_size,padding='causal',dropout_rate=TCNconfig.dropout_rate))
        decoder.add(Dense(k, activation='sigmoid'))
        #decoder = compose_model(decoder_layers)
    else:
        print('There is no other network available')
    decoder.compile(optimizer=train.optimizer, loss=train.loss, metrics=[errors])
    model_layers = modulator_layers + noise_layers + decoder_layers
    model = compose_model(model_layers)
    model.compile(optimizer=train.optimizer, loss=train.loss, metrics=[ber])
    model.summary()
    plot_model(model, to_file='./model_save/' + str(k) + '_' + str(N) + top.code+'_'+decode_choose + '_model.png', show_shapes=True,show_layer_names=True)
    checkpointer = ModelCheckpoint(filepath='./model_save/'+ str(k)+'_'+ str(N) +'_'+ top.code+'_'+ decode_choose+'checkpoint-{epoch:02d}.hdf5',monitor='val_loss', save_best_only=True, verbose=1, period=50)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10000, verbose=0)
    history = model.fit(x, d, batch_size=train.batch_size, epochs=train.nb_epoch, verbose=0, shuffle=True, validation_split=0.01,callbacks=[checkpointer,early_stopping ])
    #history = decoder.fit(x, d, batch_size=train.batch_size, epochs=train.nb_epoch, verbose=0, shuffle=True,validation_split=0.1, callbacks=[checkpointer])
    decoder.save('./model_save/' + str(k) + str(N) + top.code +'_'+ decode_choose+ '.h5')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('./results/'+str(k) + '_' + str(N) + top.code+'_'+decode_choose+'_loss.png')
    plt.show()
    return decoder

def train_network_MLP(code,x,d,N,k,train_sigma,LLR,design,batch_size,nb_epoch,optimizer,loss):
    # Define modulator
    modulator_layers = [Lambda(F.BPSK_mod,
                              input_shape=(N,), output_shape=return_output_shape, name="modulator")]
    modulator = compose_model(modulator_layers)
    modulator.compile(optimizer=optimizer, loss=loss)

    # Define noise
    noise_layers = [Lambda(F.addNoise, arguments={'sigma':train_sigma},
                           input_shape=(N,), output_shape=return_output_shape, name="noise")]
    noise = compose_model(noise_layers)
    noise.compile(optimizer=optimizer, loss=loss)
    # Define LLR
    llr_layers = [Lambda(F.log_likelihood_ratio, arguments={'sigma':train_sigma},
                         input_shape=(N,), output_shape=return_output_shape, name="LLR")]
    llr = compose_model(llr_layers)
    llr.compile(optimizer=optimizer, loss=loss)
    # define Decoder
    decoder_layers = [Dense(design[0], activation='relu', input_shape=(N,))]
    for i in range(1,len(design)):
        decoder_layers.append(Dense(design[i], activation='relu'))
    decoder_layers.append(Dense(k, activation='sigmoid'))
    decoder = compose_model(decoder_layers)
    decoder.compile(optimizer=optimizer, loss=loss, metrics=[errors])
    # Define model
    if LLR:
        model_layers = modulator_layers + noise_layers + llr_layers + decoder_layers
    else:
        model_layers = modulator_layers + noise_layers + decoder_layers
    model = compose_model(model_layers)
    model.compile(optimizer=optimizer, loss=loss, metrics=[errors])
    model.summary()
    plot_model(model, to_file='./model_save/'+str(k)+'_'+str(N)+code+'MLP_'+'model.png',show_shapes=True,show_layer_names=True)
    checkpointer = ModelCheckpoint(filepath='./model_save/checkpoint-{epoch:02d}.hdf5',monitor='val_loss',save_best_only=True, verbose=0, period=50)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10000, verbose=0)
    history = model.fit(x, d, batch_size=batch_size, epochs=nb_epoch, verbose=0,validation_split=0.1, shuffle=True,callbacks=[checkpointer,early_stopping])
    model.save('./model_save/'+str(k)+str(N)+code+'MLP_model_' + '_'.join(map(lambda num:str(num), design))+ '.h5')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    return decoder

def train_network_RNN(code,x,d,N,k,train_sigma,rnn_setup,num_Dec_layer,num_Dec_unit,batch_size,nb_epoch,optimizer,loss):
    # Define modulator
    x = x.reshape((x.shape[0],1,x.shape[1]))
    d = d.reshape((d.shape[0],1,d.shape[1]))
    modulator_layers = [Lambda(F.BPSK_mod,
                              input_shape=(1,N), output_shape=return_output_shape, name="modulator")]
    modulator = compose_model(modulator_layers)
    modulator.compile(optimizer=optimizer, loss=loss)

    # Define noise
    noise_layers = [Lambda(F.addNoise, arguments={'sigma':train_sigma},
                           input_shape=(1,N), output_shape=return_output_shape, name="noise")]
    noise = compose_model(noise_layers)
    noise.compile(optimizer=optimizer, loss=loss)
    # define Decoder
    ont_pretrain_trainable = True
    dropout_rate = 1.
    for layer in range(num_Dec_layer):
        if rnn_setup == 'gru':
            if layer==0:
                decoder_layers = [Bidirectional(GRU(units=num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                               return_sequences=True, input_shape=(N,1),trainable=ont_pretrain_trainable),
                                           name = 'Dec_'+rnn_setup+'_'+str(layer))]
            else:
                decoder_layers.append(Bidirectional(GRU(units=num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                               return_sequences=True, input_shape=(N,1),trainable=ont_pretrain_trainable),
                                           name = 'Dec_'+rnn_setup+'_'+str(layer)))
        elif rnn_setup == 'lstm':
            if layer == 0:
                decoder_layers = [Bidirectional(LSTM(units=num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                                return_sequences=True,input_shape=(N,1), trainable=ont_pretrain_trainable),
                                           name = 'Dec_'+rnn_setup+'_'+str(layer))]
            else:
                decoder_layers.append(Bidirectional(LSTM(units=num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                                        return_sequences=True, input_shape=(N, 1),
                                                        trainable=ont_pretrain_trainable),
                                                    name='Dec_' + rnn_setup + '_' + str(layer)))
        decoder_layers.append(BatchNormalization(name = 'Dec_bn'+'_'+str(layer), trainable=ont_pretrain_trainable))
    #decoder_layers = [SimpleRNN(units=num_Dec_unit, activation='tanh', dropout=dropout_rate,return_sequences=True,input_shape=(1,N), trainable=ont_pretrain_trainable)]
    decoder_layers.append(Dense(k, activation='sigmoid'))
    decoder = compose_model(decoder_layers)
    decoder.compile(optimizer=optimizer, loss=loss, metrics=[errors])
    # Define model
    model_layers = modulator_layers + noise_layers + decoder_layers
    model = compose_model(model_layers)
    model.compile(optimizer=optimizer, loss=loss, metrics=[ber])
    model.summary()
    plot_model(model,to_file='./model_save/'+str(k)+'_'+str(N)+code+'RNN_'+'model.png', show_shapes=True, show_layer_names=True)
    #print(x.shape)
    #print(d.shape)
    checkpointer = ModelCheckpoint(filepath='./model_save/checkpoint-{epoch:02d}.hdf5',monitor='val_loss',save_best_only = True, verbose = 1, period = 50)
    history = model.fit(x, d, batch_size=batch_size, epochs=nb_epoch, verbose=0, shuffle=True,validation_split=0.1,callbacks=[checkpointer])
    model.save('./model_save/'+str(k)+str(N)+code+'RNN_model_'+str(num_Dec_layer)+'_'+str(num_Dec_unit)+'.h5')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    return decoder

# 使用TCN
def train_network_TCN(code,x,d,N,k,train_sigma,nb_filters,kernel_size,dropout_rate,batch_size,nb_epoch,optimizer,loss):
    x = x.reshape((x.shape[0], 1, x.shape[1]))
    #d = d.reshape((d.shape[0], 1, d.shape[1]))
    code_input = Input(shape=(1,N), dtype='float32', name='input')
    modulator_layers = Lambda(F.BPSK_mod,input_shape=(1, N), output_shape=return_output_shape, name="modulator")
    modu = modulator_layers(code_input)
    noise_layers = Lambda(F.addNoise, arguments={'sigma': train_sigma},input_shape=(1, N), output_shape=return_output_shape, name="noise")
    train_data = noise_layers(modu)
    out = TCN(nb_filters=nb_filters, kernel_size=kernel_size,dropout_rate=dropout_rate,return_sequences=False)(train_data)
    final_out = Dense(k, activation='sigmoid')(out)
    model = Model(inputs=[code_input], outputs=[final_out])
    #decoder = Model(inputs=[train_data],outputs=[final_out])
    model.compile(optimizer=optimizer, loss=loss,metrics=[errors])
    model.summary()
    plot_model(model, to_file='./model_save/' + str(k) + '_' + str(N) + code + 'TCN_' + 'model.png', show_shapes=True,show_layer_names=True)
    checkpointer = ModelCheckpoint(filepath='./model_save/TCN_'+code+'checkpoint-{epoch:02d}.hdf5', monitor='val_loss',save_best_only=True, verbose=1, period=50)
    history = model.fit(x, d, batch_size=batch_size, epochs=nb_epoch, verbose=0, shuffle=True, validation_split=0.1,callbacks=[checkpointer])
    model.save('./model_save/' + str(k) + str(N) + code + 'TCN_model_' + str(nb_filters) + '_' + str(kernel_size) + '.h5')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    return decoder
