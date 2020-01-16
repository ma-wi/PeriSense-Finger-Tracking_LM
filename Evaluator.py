import models as ms
import models2 as ms2
from importlib import reload
from Trainer import load_input, RESULTS, USER_NUMBER_NORMALIZED, sliding_window, CROSSEVAL
from Datacollector import NORMALIZED
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.losses import mean_squared_error

# this file contains functions for plotting the results of previously learned networks


def plot_network(model, weight_path):
    parts = weight_path.split(".")
    name = parts[0]
    plot_model(model, to_file=name + '.png')


def plot_estimate(net, n, start=0, end=5000):
    model, weight_path = ms.get_network(net)
    # import data set
    print("loading data set")
    data, labels = load_input(net, "./data/all_normalized", False)
    # load weights
    parts = weight_path.split(".")
    weight_path = parts[0] + str(n) + "." + parts[1]
    model.load_weights(weight_path)
    # predict
    prediction = model.predict(data[start:end])
    # plot
    plt.plot(range(end - start), prediction, "--", label="prediction")
    plt.plot(range(end - start), labels[start:end, n], label="label")
    plt.plot(range(end - start), data[start:end], ":", label="sensors")
    plt.legend()
    plt.show()


def write_to_file(mean_squared_errors):
    with open(CROSSEVAL(), "a") as f:
        f.write("FINAL EVALUATION RESULTS\n")
        f.write("(networks, users, angles)\n")
        f.write(str(mean_squared_errors) + "\n")
        f.write("(networks, users)\n")
        f.write(str(mean_squared_errors.mean(axis=2)) + "\n")
        f.write("(networks, angles)\n")
        f.write(str(mean_squared_errors.mean(axis=1)) + "\n")
        f.write("(networks)\n")
        f.write(str(mean_squared_errors.mean(axis=2).mean(axis=1)) + "\n")


def eval2(net, seqlength=5, n1=100, n2=0, l2_regularizer=False, reg_rate=0, use_gyro=False, lr=0.001):
    # load data
    panda_frame = pd.read_csv(USER_NUMBER_NORMALIZED())
    
    # split dataset
    testset = panda_frame[panda_frame["User"] > 12]  # everything for current user aka fold
    data_test = testset.loc[:, "e1":"e4"].values
    labels_test = testset.loc[:, "index":"thumbbase"].values
    
    if (net == ms.Nets.neuron or net == ms.Nets.mlp):
        data = data_test
        labels = labels_test
    if (net == ms.Nets.seq_mlp or net == ms.Nets.conv or net == ms.Nets.lstm):
        # sliding windows for networks that work on sequences
        points, features = data_test.shape
        data = sliding_window(data_test, windowsize=seqlength)
        labels = labels_test[seqlength - 1:]
        if (net == ms.Nets.conv or net == ms.Nets.lstm):
            # reshape for lstm/conv input format
            data = data.reshape((-1, seqlength, features))

    model, weight_path = ms2.get_network(network=net, seqlength=seqlength, use_gyro=use_gyro, lr=lr, neurons_layer1=n1, neurons_layer2=n2, reg_rate=reg_rate, l2_regularizer=l2_regularizer)
    # load weights
    # parts = weight_path.split(".")
    # wp = parts[0] + "_user" + str(user_index) + "." + parts[1]
    print(weight_path)
    model.load_weights(weight_path)
    # predict
    prediction = model.predict(data)
    # calculate errors
    absolute_errors = np.abs(np.subtract(labels, prediction))  # (n,10)
    squared_errors = absolute_errors * absolute_errors
    mean_squared_errors = squared_errors.mean(axis=0)
    mean_squared_errors = np.append(mean_squared_errors, mean_squared_errors.sum() / len(mean_squared_errors))
    mean_squared_errors = np.append(mean_squared_errors, [seqlength, n1, n2, reg_rate, l2_regularizer])

    return mean_squared_errors


def eval(net, seqlength=5, lrate=0.001, neurons_layer1=100, neurons_layer2=0, reg_rate=0, reg_rate2=0):
    
    # load data
    panda_frame = pd.read_csv(USER_NUMBER_NORMALIZED())
    
    # split dataset
    testset = panda_frame[panda_frame["User"] > 12]  # everything for current user aka fold
    data_test = testset.loc[:, "e1":"e4"].values
    labels_test = testset.loc[:, "index":"thumbbase"].values
    
    if (net == ms.Nets.neuron or net == ms.Nets.mlp):
        data = data_test
        labels = labels_test
    if (net == ms.Nets.seq_mlp or net == ms.Nets.conv or net == ms.Nets.lstm):
        # sliding windows for networks that work on sequences
        points, features = data_test.shape
        data = sliding_window(data_test, windowsize=seqlength)
        labels = labels_test[seqlength - 1:]
        if (net == ms.Nets.conv or net == ms.Nets.lstm):
            # reshape for lstm/conv input format
            data = data.reshape((-1, seqlength, features))

    model, weight_path = ms.get_network(net, seqlength, False, lrate, neurons_layer1, neurons_layer2, reg_rate, reg_rate2, False)
    # load weights
    # parts = weight_path.split(".")
    # wp = parts[0] + "_user" + str(user_index) + "." + parts[1]
    print(weight_path)
    model.load_weights(weight_path)
    # predict
    prediction = model.predict(data)
    # calculate errors
    absolute_errors = np.abs(np.subtract(labels, prediction))  # (n,10)
    squared_errors = absolute_errors * absolute_errors
    mean_squared_errors = squared_errors.mean(axis=0)
    mean_squared_errors = np.append(mean_squared_errors, mean_squared_errors.sum() / len(mean_squared_errors))
    mean_squared_errors = np.append(mean_squared_errors, [seqlength, neurons_layer1, neurons_layer2, reg_rate, reg_rate2])

    return mean_squared_errors


def plot_result_charts(seqlength=5, columnchart=False, boxplot=False, write_results=False, subangles=False, use_gyro=False):
    # networks = [ms.Nets.neuron, ms.Nets.mlp, ms.Nets.seq_mlp, ms.Nets.conv, ms.Nets.lstm]
    networks = [ms.Nets.mlp]
    # load data
    panda_frame = pd.read_csv(USER_NUMBER_NORMALIZED())
    # container for absolute errors (for boxplots)
    all_absolute_errors = []  # (users, networks)
    # container for labels (for subangles)
    all_labels = []  # (users, networks)
    # container for mean squared errors
    mean_squared_errors = np.zeros((5, 9, 10))  # (Networks, Users, Angles)
    userlist = [(0, 2), (1, 3), (2, 5), (3, 7), (4, 9), (5, 11), (6, 14), (7, 18), (8, 21)]
    for user_index, user_number in userlist:
        # split dataset
        testset = panda_frame[panda_frame["User"] == user_number]  # everything for current user aka fold
        if use_gyro:
            data_test = testset.loc[:, "e1":"gz"].values
        else:
            data_test = testset.loc[:, "e1":"e4"].values
        labels_test = testset.loc[:, "index":"thumbbase"].values
        # expand list
        all_absolute_errors.append([])
        all_labels.append([])

        for net in networks:
            if (net == ms.Nets.neuron or net == ms.Nets.mlp):
                data = data_test
                labels = labels_test
            if (net == ms.Nets.seq_mlp or net == ms.Nets.conv or net == ms.Nets.lstm):
                # sliding windows for networks that work on sequences
                points, features = data_test.shape
                data = sliding_window(data_test, windowsize=seqlength)
                labels = labels_test[seqlength - 1:]
                if (net == ms.Nets.conv or net == ms.Nets.lstm):
                    # reshape for lstm/conv input format
                    data = data.reshape((-1, seqlength, features))

            model, weight_path = ms.get_network(net, use_gyro=use_gyro)
            # load weights
            parts = weight_path.split(".")
            wp = parts[0] + "_user" + str(user_index) + "." + parts[1]
            print(wp)
            model.load_weights(wp)
            # predict
            print(net.name + " predicting user " + str(user_index))
            prediction = model.predict(data)
            # calculate errors
            absolute_errors = np.abs(np.subtract(labels, prediction))  # (n,10)
            if boxplot or subangles:
                # save absolute errors for boxplot
                all_absolute_errors[user_index].append(absolute_errors * 90)
            if columnchart:
                # save mean squared errors for column charts
                squared_errors = absolute_errors * absolute_errors
                mean_squared_errors[net.value - 1, user_index] = squared_errors.mean(axis=0)
            if subangles:
                # save labels for subangle analysis
                all_labels[user_index].append(labels * 90)
                
    if write_results:
        write_to_file(mean_squared_errors)
    if columnchart:
        plot_column_chart(mean_squared_errors, True)
        plot_column_chart(mean_squared_errors, False)
    if boxplot:
        plot_boxplot(all_absolute_errors, True)
        plot_boxplot(all_absolute_errors, False)
    if subangles:
        plot_boxplot(all_absolute_errors, subangles=True, all_labels=all_labels)


def plot_boxplot(all_absolute_errors, users=True, subangles=False, all_labels=[]):  # input: list (users, networks) holds np.arrays (setlength, angles)
    # reshape to target
    target_absolute_errors = []
    if subangles:
        for s in range(9):  # subangles
            for n in range(len(all_absolute_errors[0])):  # networks
                subangle_errors = []
                for u in range(len(all_absolute_errors)):  # users
                    condition = np.abs(all_labels[u][n] - (s * 10 + 5)) <= 5
                    subangle_errors.extend(all_absolute_errors[u][n][condition])
                target_absolute_errors.append(subangle_errors)
    elif users:
        for u in range(len(all_absolute_errors)):  # users
            for n in range(len(all_absolute_errors[u])):  # networks
                target_absolute_errors.append(all_absolute_errors[u][n].flatten())
    else:
        for a in range(10):  # angles
            for n in range(len(all_absolute_errors[0])):  # networks
                network_angle_errors = []
                for u in range(len(all_absolute_errors)):  # users
                    network_angle_errors.extend(all_absolute_errors[u][n][:, a].flatten())
                target_absolute_errors.append(network_angle_errors)

    # plot
    groups = len(target_absolute_errors) // 5
    group_pos = np.arange(groups)  # x position for bar groups
    width = 0.15
    offsets = np.ones((groups, 1)).dot(((np.arange(5) - 2) * width).reshape((1, 5))).T
    pos = (group_pos + offsets).flatten(order="F")  # x positions for each bar
    plt.figure(figsize=(10, 7), dpi=400)
    if subangles:
        plt.title("Absolute error distribution of each network for each label range")
    elif users:
        plt.title("Absolute error distribution of each network for each user fold")
    else:
        plt.title("Absolute error distribution of each network for each finger angle across all folds")    
    bp = plt.boxplot(target_absolute_errors, widths=width, positions=pos, showfliers=False, whis=[len(all_absolute_errors[0]), 95])
    # axis labels
    plt.ylabel("absolute error in degrees")
    if subangles:
        plt.xlabel("label range in degrees")
        plt.xticks(group_pos, ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"])
    elif users:
        plt.xlabel("user IDs")
        plt.xticks(group_pos, np.arange(len(all_absolute_errors)) + 1)
    else:
        plt.xlabel("outer finger angle                                                    inner finger angle")
        plt.xticks(group_pos, ["index", "middle", "ring", "little", "thumb", "index", "middle", "ring", "little", "thumb"])
    # create legend
    if len(all_absolute_errors[0]) is 5:
        handles = [bp['boxes'][0], bp['boxes'][1], bp['boxes'][2], bp['boxes'][3], bp['boxes'][4]]
        plt.legend(handles, ["Neuron", "MLP", "Seq. MLP", "1D Conv.", "LSTM"], loc="upper right")
    # save plot
    if subangles:
        plt.savefig("results/subangles.png")
    elif users:
        plt.savefig("results/user_absolute_results.png")
    else:
        plt.savefig("results/angle_absolute_results.png")
    plt.clf()
    

def plot_column_chart(mean_squared_errors, users=True):  # input: (networks, users, angles)
    # calculate target means
    if users:
        target_mean_squared_errors = mean_squared_errors.mean(axis=2)  # (networks, users)
    else:
        target_mean_squared_errors = mean_squared_errors.mean(axis=1)  # (networks, angles)

    # plot
    pos = np.arange(len(target_mean_squared_errors[0]))  # x position for bar groups
    width = 0.15
    plt.figure(figsize=(10, 7), dpi=400)
    if users:
        plt.title("Average squared error of each network for each user fold")
    else:
        plt.title("Average squared error of each network for each finger angle across all folds")        
    plt.bar(pos - 2 * width, target_mean_squared_errors[0], width, label="Neuron")
    plt.bar(pos - 1 * width, target_mean_squared_errors[1], width, label="MLP")
    plt.bar(pos          , target_mean_squared_errors[2], width, label="Seq. MLP")
    plt.bar(pos + 1 * width, target_mean_squared_errors[3], width, label="1D Conv.")
    plt.bar(pos + 2 * width, target_mean_squared_errors[4], width, label="LSTM")
    plt.ylabel("average squared error")
    if users:
        plt.xlabel("user IDs")
        plt.xticks(pos, np.arange(9) + 1)
    else:
        plt.xlabel("outer finger angle                                                    inner finger angle")
        plt.xticks(pos, ["index", "middle", "ring", "little", "thumb", "index", "middle", "ring", "little", "thumb"])
    plt.legend(loc="upper right")
    # save plot
    if users:
        plt.savefig("results/user_square_results.png")
    else:
        plt.savefig("results/angle_square_results.png")
    plt.clf()


def plot_data_histogram(path=NORMALIZED()):
    panda_frame = pd.read_csv(path)
    panda_frame = panda_frame.rename(columns={"e1": "Capacitive sensor #1", "e2": "Capacitive sensor #2", "e3": "Capacitive sensor #3", "e4": "Capacitive sensor #4"})
    hist = panda_frame.loc[:, "Capacitive sensor #1":"Capacitive sensor #4"].hist(figsize=(6, 6), sharey=True, sharex=True, grid=False)
    plt.savefig("results/sensor_histograms.png")
    plt.clf()
    
    panda_frame = panda_frame.rename(columns={"index": "Outer index finger angle", "indexbase": "Inner index finger angle", "middle": "Outer middle finger angle", "middlebase": "Inner middle finger angle", "ring": "Outer ring finger angle", "ringbase": "Inner ring finger angle", "pinky": "Outer little finger angle", "pinkybase": "Inner little finger angle", "thumb": "Outer thumb angle", "thumbbase": "Inner thumb angle"})
    hist = panda_frame.loc[:, "Outer index finger angle":"Inner thumb angle"].hist(figsize=(15, 6), sharey=True, sharex=True, grid=False, layout=(2, 5))
    plt.savefig("results/label_histograms.png")
    plt.clf()
