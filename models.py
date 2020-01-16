from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Reshape, Conv1D, Flatten, MaxPool1D, Dropout, MaxPooling1D, ConvLSTM2D
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from keras.metrics import mean_absolute_error
from enum import Enum


class Nets(Enum):
    neuron = 1
    conv = 2
    lstm = 3
    lstm2 = 4
    seq_mlp = 5

def simple_neuron(fdim, lrate):
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation="relu"))
    model.compile(optimizer=RMSprop(lr=lrate, decay=0.0),
                  loss=mean_squared_error, metrics=[mean_absolute_error])
    return model, "weights/simple_neuron.hdf5"

def lstm(seqlength, fdim, nn=20, lr=0.001, dr=0.3):
    model = Sequential()    
    model.add(LSTM(nn, input_shape=(seqlength, fdim))) 
    model.add(Dropout(dr))
    model.add(Dense(10, activation="relu"))
    model.compile(optimizer=RMSprop(lr=lr, decay=0.0),
                  loss=mean_squared_error, metrics=[mean_absolute_error])
    return model, "weights/lstm.hdf5"

def lstm2(seqlength, fdim, nn=20, nn2=10, lr=0.001, dr=0.3):
    model = Sequential()
    model.add(LSTM(nn, input_shape=(seqlength, fdim)))  
    model.add(Dropout(dr))
    model.add(Dense(nn2, activation='relu'))
    model.add(Dense(10, activation="relu"))
    model.compile(optimizer=RMSprop(lr=lr, decay=0.0),
                  loss=mean_squared_error, metrics=[mean_absolute_error])
    return model, "weights/lstm.hdf5"

def conv(seqlength=5, fdim=4, lrate=0.001):
#    model = Sequential()
#    model.add(Conv1D(4, 2, input_shape=(seqlength, 4)))
#    model.add(MaxPool1D())
#    model.add(Flatten())
#    model.add(Dense(100, activation="tanh"))
#    model.add(Dense(8, activation="relu"))
    
    #model = Sequential()
    #model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(seqlength,4)))
    #model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(8, activation="relu"))
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,4), activation='relu', input_shape=(fdim, 1, 5, 4)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=lrate, decay=0.0),
                  loss=mean_squared_error, metrics=[mean_absolute_error])
    return model, "weights/conv1d.hdf5"


def get_network(network, fdim, seqlength=5, nn=50, nn2=25, lr=0.001,  do=0.03):
    
    if (network == Nets.neuron):
        return simple_neuron(fdim, lr)
    if (network == Nets.conv):
        return conv(seqlength, fdim, lr)
    if (network == Nets.lstm):
        return lstm(seqlength, fdim, nn=nn, lr=lr, dr=do)
    if (network == Nets.lstm2):
        return lstm2(seqlength, fdim, nn=nn, nn2= nn2, lr=lr, dr=do)
    raise NotImplementedError(
        "The requested network is not yet implemented", network)
