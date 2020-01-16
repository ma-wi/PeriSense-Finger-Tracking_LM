import models as ms
from sklearn.utils import shuffle as do_shuffle
from numpy import hstack
import pandas as pd
from keras.callbacks import EarlyStopping
import os

def _label_names():
    return ["index","middle","ring","pinky","thumb","indexbase","middlebase","ringbase","pinkybase","thumbbase"] # "index":"thumbbase"

def _sliding_window(a, stepsize=1, windowsize=5):
    return hstack(a[i:1 + i - windowsize or None:stepsize] for i in range(0, windowsize))

def train(network, data_file, train_users, validation_users, tid, run_no=1, epochs=25, seqlength=20, nn=100, nn2=25, lr=0.001, do=0.3):
    """Trains a network using data of the specified users.
    
    """
    
    panda_frame = pd.read_csv(data_file)
    
    # setup network
    model, weight_path = ms.get_network(network, 4, seqlength, nn, nn2, lr, do)
    
    # prepare result file holding the trained weights
    parts = weight_path.split(".")
    if not os.path.isdir("weights"):
        os.mkdir("weights")
    if not os.path.isdir("weights/" + str(run_no)):
        os.mkdir("weights/" + str(run_no))
    wp = "weights/" + str(run_no) + "/" + network.name + "_user" + str(tid) + "_data" + data_file[data_file.find("/")+1:] + "_model" + str(network) + "_sl" + str(seqlength) + "_nn" + str(nn) + "_nn2" + str(nn2) + "_lr" + str(lr) + "_do" + str(do) + "." + parts[1]
    
    # split dataset
    testset = panda_frame[panda_frame["User"].isin(validation_users)]  # everything for current user aka fold
    data_test = testset.loc[:, "e1":"e4"].values
    labels_test = testset.loc[:, _label_names()].values
    trainset = panda_frame[panda_frame["User"].isin(train_users)]  # everything else
    data_train = trainset.loc[:, "e1":"e4"].values
    labels_train = trainset.loc[:, _label_names()].values
    if (network == ms.Nets.seq_mlp or network == ms.Nets.conv or network == ms.Nets.lstm or network == ms.Nets.lstm2):
        # sliding windows for networks that work on sequences
        points, features = data_test.shape
        data_test = _sliding_window(data_test, windowsize=seqlength)
        data_train = _sliding_window(data_train, windowsize=seqlength)
        labels_test = labels_test[seqlength - 1:]
        labels_train = labels_train[seqlength - 1:]
        if (network == ms.Nets.conv or network == ms.Nets.lstm or network == ms.Nets.lstm2):
            # reshape for lstm/conv input format
            data_test = data_test.reshape((-1, seqlength, features))
            data_train = data_train.reshape((-1, seqlength, features))
            
    print(data_train.shape)
#    # shuffle
#    data_train, labels_train = do_shuffle(data_train, labels_train)
#    
#    # train
#    model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=epochs, callbacks=[EarlyStopping(patience=2, verbose=1)], verbose=2)
#    
#    # store weights        
#    print("saving weights to " + wp)
#    model.save(wp)

