# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: <authorname>
"""

#system
import multiprocessing
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import models as ms
import argparse

useGPU = False
lock = multiprocessing.Lock()


def train_conv(net, path, tu, vu, tid, rn, ep, seq, n, n2, l, d):
    lock.acquire()
    
    if useGPU:
        import tensorflow as tf
        #restrict the gpu-resources used by this program (multi-user-friendly):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0) #choose one of the two gpu's
        # enable dynamic gpu-memory allocation (otherwise the whole memory will be allocated):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
    else:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""    

    import Trainer
    
    lock.release()
    
    Trainer.train(network=net, data_file=path, train_users=tu, validation_users=vu, tid=tid, run_no=rn, epochs=ep, seqlength=seq, nn=n, nn2=n2, lr=l, do=d)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Skript to train and evalute finger tracking for PeriSense.')
    parser.add_argument("-t", "--train", dest="train", action='store_true', help="run training")
    parser.add_argument("-e", "--eval", dest="eval", action='store_true', help="run eval")
    
    args = parser.parse_args()
    
    if not (args.train or args.eval):
        parser.print_help()
        exit(-1)

    if args.train:
            
        params = []        
        validation_users = [130, 132, 133]
        users = np.arange(111,128,1)
        
        for u in range(111,128):
            train_users = np.delete(users, np.where(r == u)[0][0])
            for r in range(1,3):
                net = ms.Nets.lstm
                p = "data/user_number_1_to_6"
                if r == 2:
                    net = ms.Nets.neuron # baseline
                
                params.append((net, p, train_users, validation_users, u, r, ep=25, seq=5, n=50, n2=25, l=0.001, d=0.3))
                                    
        with multiprocessing.Pool(processes=88) as pool:
            pool.starmap(train_conv, params)

    if args.eval:
        pass
    
    exit()

