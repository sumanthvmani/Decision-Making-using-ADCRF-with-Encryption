import numpy as np
import os
import pandas as pd
from numpy import matlib
from AOA import AOA
from BlockChain import BlockChain
from DES import DES
from ECC import ECC
from FSA import FSA
from MRA import MRA
from Model_ADCRF import Model_ADCRF
from Model_DNN import Model_DNN
from Model_LSTM import Model_LSTM
from Model_RF import Model_RF
from PFOA import PFOA
from PROPOSED import PROPOSED
from Plot_results import *
from RSA import RSA
from objfun import *

# Read the Dataset
an = 0
if an == 1:
    Dataset_Path = './Datasets/diabetes.csv'
    Datas = pd.read_csv(Dataset_Path)
    Datas.drop('Outcome', inplace=True, axis=1)
    Datas = np.asarray(Datas)
    tar = pd.read_csv(Dataset_Path, usecols=['Outcome'])
    tar = np.asarray(tar)
    index = np.arange(len(tar))
    np.random.shuffle(index)
    Shuffled_Datas = Datas[index]
    Shuffled_Target = tar[index]
    np.save('Index.npy', index)
    np.save('Datas.npy', Shuffled_Datas)
    np.save('Targets.npy', Shuffled_Target)


# Initialization Attributes
an = 0
if an == 1:
    # Number of IoT nodes / samples
    no_of_nodes = 6000

    # Communication Range (meters) – typical IoT range 10–500 m
    Communication_Range = np.random.randint(10, 500, size=no_of_nodes)

    # Data Rate (kbps) – typical sensor transmission speed
    Data_Rate = np.random.randint(50, 1000, size=no_of_nodes)

    # Power Consumption (mW)
    Power_Consumption = np.random.uniform(10, 100, size=no_of_nodes)

    # Latency (ms)
    Latency = np.random.uniform(1, 100, size=no_of_nodes)

    # Data Transmission Frequency (packets per second)
    Data_Transmission_Frequency = np.random.randint(1, 60, size=no_of_nodes)

    # Network Topology
    # 0 = Star, 1 = Mesh, 2 = Tree, 3 = Hybrid
    Network_Topology = np.random.randint(4, size=no_of_nodes)

    topology_names = ["Star", "Mesh", "Tree", "Hybrid"]
    topology_description = []

    for i in range(no_of_nodes):
        topology_description.append(topology_names[Network_Topology[i]])

    # Sensor Type
    # 0 = Temperature, 1 = Humidity, 2 = Pressure, 3 = Motion, 4 = Gas
    Sensor_Type = np.random.randint(5, size=no_of_nodes)

    # sensor_names = ["Temperature", "Humidity", "Pressure", "Motion", "Gas"]
    sensor_names = [0, 1, 2, 3, 4]
    sensor_description = []

    for i in range(no_of_nodes):
        sensor_description.append(sensor_names[Sensor_Type[i]])

    # Bandwidth (MHz)
    Bandwidth = np.random.randint(1, 100, size=no_of_nodes)

    # Combine dataset
    data = np.column_stack((Communication_Range,
                            Data_Rate,
                            Power_Consumption,
                            Latency,
                            Data_Transmission_Frequency,
                            Network_Topology,
                            Sensor_Type,
                            Bandwidth))

    # Total number of samples
    n_samples = data.shape[0]
    shuffle_idx = np.random.permutation(n_samples)
    Datas = data[shuffle_idx]
    # Split into two halves
    half = n_samples // 2

    verified_data = Datas[:half].copy()
    non_verified_data = Datas[half:].copy()

    # Targets
    y_verified = np.zeros(half)
    y_non_verified = np.ones(n_samples - half)

    # Shuffle the ID column (column 0) only for non-verified data
    shuffled_ids = np.random.permutation(non_verified_data[:, 0])
    non_verified_data[:, 0] = shuffled_ids

    # Combine back
    X = np.vstack((verified_data, non_verified_data))
    y = np.concatenate((y_verified, y_non_verified))

    final_shuffle = np.random.permutation(len(X))
    X = X[final_shuffle]
    y = y[final_shuffle]
    y = np.asarray(y).reshape(-1, 1).astype('int')

    np.save('IoT_Network_Data.npy', X)
    np.save('IoT_Target.npy', y)


# BlockChain
an = 0
if an == 1:
    Data = np.load('IoT_Network_Data.npy', allow_pickle=True)
    Target = np.load('IoT_Target.npy', allow_pickle=True)
    SecuredData, SecuredTarget = BlockChain(Data, Target)
    np.save('SecuredData.npy', SecuredData)
    np.save('SecuredTarget.npy', SecuredTarget)


# Optimization for Classification
an = 0
if an == 1:
    Data = np.load('SecuredData.npy', allow_pickle=True)
    Target = np.load('SecuredTarget.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron, Epoch, Learning rate in DCRF
    xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("MRA...")
    [bestfit1, fitness1, bestsol1, time1] = MRA(initsol, fname, xmin, xmax, Max_iter)  # MRA

    print("FSA...")
    [bestfit2, fitness2, bestsol2, time2] = FSA(initsol, fname, xmin, xmax, Max_iter)  # FSA

    print("AOA...")
    [bestfit3, fitness3, bestsol3, time3] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("PFOA...")
    [bestfit4, fitness4, bestsol4, time4] = PFOA(initsol, fname, xmin, xmax, Max_iter)  # PFOA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),  bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))  # Save the Fitness
    np.save('BestSol.npy', np.asarray(BestSol))  # Save the Best sol

# Classification
an = 0
if an == 1:
    Feat = np.load('SecuredData.npy', allow_pickle=True)
    Target = np.load('SecuredTarget.npy', allow_pickle=True)
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # Load the Best Solution
    k_fold = 5
    Per = 1 / k_fold
    EVAL = []
    Perc = round(Feat.shape[0] * Per)
    for i in range(k_fold):
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(j)
            sol = BestSol[j, :]
            Eval[j, :], pred0 = Model_ADCRF(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
        Eval[5, :], pred1 = Model_RF(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_ADCRF(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_ALL_Fold.npy', np.asarray(EVAL))


# Optimization for Cryptography
an = 0
if an == 1:
    NoOfBlocks = [5, 10, 15, 20, 25]
    Data_S = np.load('SecuredData.npy', allow_pickle=True)
    Data_NS = np.load('SecuredTarget.npy', allow_pickle=True)
    Best_Crypto = []
    Fit_Crypto = []
    for n in range(len(NoOfBlocks)):
        Global_Vars.sensitive = Data_S
        Global_Vars.Blocks = NoOfBlocks[n]
        Npop = 10
        Chlen = 16
        xmin = matlib.repmat([0], Npop, Chlen)
        xmax = matlib.repmat([1], Npop, Chlen)
        fname = objfun_Crypt
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = (np.random.uniform(xmin[p1, p2], xmax[p1, p2]))
        Max_iter = 50

        print("MRA...")
        [bestfit1, fitness1, bestsol1, time1] = MRA(initsol, fname, xmin, xmax, Max_iter)  # MRA

        print("FSA...")
        [bestfit2, fitness2, bestsol2, time2] = FSA(initsol, fname, xmin, xmax, Max_iter)  # FSA

        print("AOA...")
        [bestfit3, fitness3, bestsol3, time3] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("PFOA...")
        [bestfit4, fitness4, bestsol4, time4] = PFOA(initsol, fname, xmin, xmax, Max_iter)  # PFOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        Best_Crypto.append(BestSol)
        Fit_Crypto.append(fitness)
    np.save('Fitness_Crypto.npy', np.asarray(Fit_Crypto))
    np.save('BestSol_crypto.npy', np.asarray(Best_Crypto))

# Cryptography
an = 0
if an == 1:
    Data = np.load('SecuredData.npy', allow_pickle=True)
    BestSol = np.load('BestSol_crypto.npy', allow_pickle=True)
    EVAL = []
    NoOfBlocks = [5, 10, 15, 20, 25]
    for i in range(len(NoOfBlocks)):
        Eval = np.zeros((10, 4))
        for j in range(BestSol.shape[0]):
            print(i, len(NoOfBlocks), j, BestSol.shape[0])
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], Encrypt_data, Decrypt_data = OMA_BE(Data, NoOfBlocks[i], sol=sol)
        Eval[5, :], Encrypt_data_1, Decrypt_data_1 = ECC(Data, NoOfBlocks[i])
        Eval[6, :], Encrypt_data_2, Decrypt_data_2 = RSA(Data, NoOfBlocks[i])
        Eval[7, :], Encrypt_data_3, Decrypt_data_3 = DES(Data, NoOfBlocks[i])
        Eval[8, :], Encrypt_data_4, Decrypt_data_4 = OMA_BE(Data, NoOfBlocks[i])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    Encryption_Time = EVAL[:, 0]
    Decryption_Time = EVAL[:, 1]
    Memory_size = EVAL[:, 2]
    Computational_time = EVAL[:, 3]
    np.save('Time_encryp.npy', Encryption_Time)
    np.save('Time_decryp.npy', Decryption_Time)
    np.save('Memory_size.npy', Memory_size)
    np.save('Total_comput_time.npy', Computational_time)


plot_convergence()
plot_Crypto_convergence()
ROC_curve()
Plot_Confusion()
Plot_Activation()
Plot_KFold()
Plot_encryption()
plot_Block_Chain_Security()
