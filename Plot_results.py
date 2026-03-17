from itertools import cycle
import numpy as np
from prettytable import PrettyTable
from matplotlib import pylab
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import warnings
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


def Statastical(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_convergence():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'MRA-ADCRF', 'FSA-ADCRF', 'AOA-ADCRF', 'PFOA-ADCRF', 'MES-PFOA-ADCRF']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv = np.zeros((Fitness.shape[-2], 5))
    for j in range(len(Algorithm) - 1):
        Conv[j, :] = Statastical(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv[j, :])
    print('-------------------------------------------------- Statistical Report ',
          '  --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#0cff0c', linewidth=3, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#aa23ff', linewidth=3, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.savefig("./Results/Convergence.png")
    plt.show()


def plot_Crypto_convergence():
    NoOfBlocks = [5, 10, 15, 20, 25]
    Fitness = np.load('Fitness_Crypto.npy', allow_pickle=True)

    ENC_Algorithm = ['TERMS', 'MRA-O-MA-ABE', 'FSA-O-MA-ABE', 'AOA-O-MA-ABE', 'PFOA-O-MA-ABE', 'MES-PFOA-O-MA-ABE']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv = np.zeros((Fitness.shape[-2], 5))
    for n in range(Fitness.shape[0]):
        for j in range(len(ENC_Algorithm) - 1):
            Conv[j, :] = Statastical(Fitness[n, j, :])
        Table = PrettyTable()
        Table.add_column(ENC_Algorithm[0], Terms)
        for j in range(len(ENC_Algorithm) - 1):
            Table.add_column(ENC_Algorithm[j + 1], Conv[j, :])
        print('-------------------------------------------------- Statistical Report ', str(NoOfBlocks[n]),
              'Block size',
              '  --------------------------------------------------')
        print(Table)
        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[n]

        plt.plot(length, Conv_Graph[0, :], color='#f97306', linewidth=3, label=ENC_Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='#9a0eea', linewidth=3, label=ENC_Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='#01ff07', linewidth=3, label=ENC_Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='#fe01b1', linewidth=3, label=ENC_Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label=ENC_Algorithm[5])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve')
        plt.savefig("./Results/No of Block size %s Convergence.png" % NoOfBlocks[n])
        plt.show()


def ROC_curve():
    lw = 2
    cls = ['RF', 'DNN', 'LSTM', 'DCRF', 'MES-PFOA-ADCRF']
    Actual = np.load('Targets.npy', allow_pickle=True).astype('int')
    colors = cycle(
        ["#fe2f4a", "#0165fc", "#fcb001", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
    Classes = ['Normal', 'Diabetes']
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(Classes)
    ax.yaxis.set_ticklabels(Classes)
    path = "./Results/Confusion_matrix.png"
    plt.title('Confusion matrix')
    plt.savefig(path)
    plt.show()


def Plot_Activation():
    eval = np.load('Eval_ALL_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Terms = [0, 1, 3, 4, 5, 7, 9, 12]
    Activation = ['Linear', 'Relu', 'Tanh', 'Softmax', 'Sigmoid', 'Leaky Relu']
    Algorithm = ['MRA-ADCRF', 'FSA-ADCRF', 'AOA-ADCRF', 'PFOA-ADCRF', 'MES-PFOA-ADCRF']
    Classifier = ['RF', 'DNN', 'LSTM', 'DCRF', 'MES-PFOA-ADCRF']
    for j in range(len(Graph_Terms)):
        Graph = eval[:, :, Graph_Terms[j] + 4]
        # Algorithm data
        Algorithm_data = Graph[:, :5]
        x = np.arange(Algorithm_data.shape[0])
        width = 0.15
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#bf77f6']  # Colors for the regions
        for i, region in enumerate(Algorithm):
            ax.bar(x + i * width, Algorithm_data[:, i], width, label=region, color=colors[i])
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(Activation, fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold', color='#35530a')
        circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in
                          range(len(Algorithm))]
        ax.legend(circle_markers, Algorithm, title="", fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  frameon=False, ncol=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        path = "./Results/Activation_%s_ALG.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Activation vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()

        # Method data
        Method_data = Graph[:, 5:]
        x = np.arange(Method_data.shape[0])
        width = 0.15
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#bf77f6']  # Colors for the regions
        for i, region in enumerate(Classifier):
            ax.bar(x + i * width, Method_data[:, i], width, label=region, color=colors[i])
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(Activation, fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold', color='#35530a')
        circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in
                          range(len(Classifier))]
        ax.legend(circle_markers, Classifier, title="", fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  frameon=False, ncol=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        path = "./Results/Activation_%s_MTD.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Activation vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Plot_KFold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Table_Term = [0, 2, 4, 5, 7, 8, 9, 12]
    k_fold = ['1', '2', '3', '4', '5']

    Algorithm = ['TERMS', 'MRA-ADCRF', 'FSA-ADCRF', 'AOA-ADCRF', 'PFOA-ADCRF', 'MES-PFOA-ADCRF']
    Classifier = ['TERMS', 'RF', 'DNN', 'LSTM', 'DCRF', 'MES-PFOA-ADCRF']
    for k in range(eval.shape[0]):
        value = eval[k, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Term])
        print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of',
              'Algorithm Comparison --------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
        print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of',
              'Classifier Comparison --------------------------------------------------')
        print(Table)


def plot_Attacks():
    Terms = ['CPA', 'KPA']
    Graph_Term = [0, 1]
    Attacks = np.load('Attacks.npy', allow_pickle=True)
    ENC_Algorithm = ['MRA-O-MA-ABE', 'FSA-O-MA-ABE', 'AOA-O-MA-ABE', 'PFOA-O-MA-ABE', 'MES-PFOA-O-MA-ABE']
    ENC_Methods = ['ECC', 'RSA', 'DES', 'MA-ABE', 'MES-PFOA-O-MA-ABE']
    Cases = ['1', '2', '3', '4', '5']
    for n in range(len(Graph_Term)):

        Graph = Attacks[Graph_Term[n], :5, :]
        Graphs = np.transpose(Graph)
        Graph_Time = Attacks[Graph_Term[n], 5:, :]
        Graphs_Time = np.transpose(Graph_Time)

        Algorithm_data = Graphs[:, :]
        width = 0.15
        spacing_within_group = 0.05
        group_spacing = 0.5
        x_base = np.arange(len(Cases)) * (5 * width + group_spacing)
        x_alg_1 = x_base
        x_alg_2 = x_base + (width + spacing_within_group)
        x_alg_3 = x_base + 2 * (width + spacing_within_group)
        x_alg_4 = x_base + 3 * (width + spacing_within_group)
        x_alg_5 = x_base + 4 * (width + spacing_within_group)
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x_alg_1, Algorithm_data[:, 0], width, label=ENC_Algorithm[0], color='#fe01b1')
        rects2 = ax.bar(x_alg_2, Algorithm_data[:, 1], width, label=ENC_Algorithm[1], color='#c65102')
        rects3 = ax.bar(x_alg_3, Algorithm_data[:, 2], width, label=ENC_Algorithm[2], color='#a9f971')
        rects4 = ax.bar(x_alg_4, Algorithm_data[:, 3], width, label=ENC_Algorithm[3], color='#0485d1')
        rects5 = ax.bar(x_alg_5, Algorithm_data[:, 4], width, label=ENC_Algorithm[4], color='#d2bd0a')
        ax.set_xticks(x_base + 2 * (width + spacing_within_group))
        ax.set_xticklabels(Cases)
        ax.set_xlabel('Cases', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
                  handleheight=1, handlelength=1, fontsize=10)
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        ax.spines['left'].set_color('lightgray')
        ax.spines['left'].set_linewidth(0.0)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Block Size vs ' + Terms[Graph_Term[n]])
        plt.tight_layout()
        path = "./Results/%s_Blocksize_bar_enc_ALG.png" % (Terms[Graph_Term[n]])
        plt.savefig(path)
        plt.show()

        Method_data = Graphs_Time[:, :]
        width = 0.15
        spacing_within_group = 0.05
        group_spacing = 0.5
        x_base = np.arange(len(Cases)) * (5 * width + group_spacing)
        x_alg_1 = x_base
        x_alg_2 = x_base + (width + spacing_within_group)
        x_alg_3 = x_base + 2 * (width + spacing_within_group)
        x_alg_4 = x_base + 3 * (width + spacing_within_group)
        x_alg_5 = x_base + 4 * (width + spacing_within_group)
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x_alg_1, Method_data[:, 0], width, label=ENC_Methods[0], color='#fe01b1')
        rects2 = ax.bar(x_alg_2, Method_data[:, 1], width, label=ENC_Methods[1], color='#c65102')
        rects3 = ax.bar(x_alg_3, Method_data[:, 2], width, label=ENC_Methods[2], color='#a9f971')
        rects4 = ax.bar(x_alg_4, Method_data[:, 3], width, label=ENC_Methods[3], color='#0485d1')
        rects5 = ax.bar(x_alg_5, Method_data[:, 4], width, label=ENC_Methods[4], color='#d2bd0a')
        ax.set_xticks(x_base + 2 * (width + spacing_within_group))
        ax.set_xticklabels(Cases)
        ax.set_xlabel('Cases', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
                  handleheight=1, handlelength=1, fontsize=10)
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        ax.spines['left'].set_color('lightgray')
        ax.spines['left'].set_linewidth(0.0)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Block Size vs ' + Terms[Graph_Term[n]])
        plt.tight_layout()
        path = "./Results/%s_Blocksize_bar_enc_MTD.png" % (Terms[Graph_Term[n]])
        plt.savefig(path)
        plt.show()


def plot_key_sencitivity():
    senc = np.load('Key_sensitivity.npy', allow_pickle=True)
    ENC_Algorithm = ['MRA-O-MA-ABE', 'FSA-O-MA-ABE', 'AOA-O-MA-ABE', 'PFOA-O-MA-ABE', 'MES-PFOA-O-MA-ABE']
    ENC_Methods = ['ECC', 'RSA', 'DES', 'MA-ABE', 'MES-PFOA-O-MA-ABE']
    Cases = ['1', '2', '3', '4', '5']
    Graph = senc[:5, :]
    Graphs = np.transpose(Graph)
    Graph_Time = senc[5:, :]
    Graphs_Time = np.transpose(Graph_Time)
    Algorithm_data = Graphs[:, :]
    width = 0.15
    spacing_within_group = 0.05
    group_spacing = 0.5
    x_base = np.arange(len(Cases)) * (5 * width + group_spacing)
    x_alg_1 = x_base
    x_alg_2 = x_base + (width + spacing_within_group)
    x_alg_3 = x_base + 2 * (width + spacing_within_group)
    x_alg_4 = x_base + 3 * (width + spacing_within_group)
    x_alg_5 = x_base + 4 * (width + spacing_within_group)
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x_alg_1, Algorithm_data[:, 0], width, label=ENC_Algorithm[0], color='#fe01b1')
    rects2 = ax.bar(x_alg_2, Algorithm_data[:, 1], width, label=ENC_Algorithm[1], color='#c65102')
    rects3 = ax.bar(x_alg_3, Algorithm_data[:, 2], width, label=ENC_Algorithm[2], color='#a9f971')
    rects4 = ax.bar(x_alg_4, Algorithm_data[:, 3], width, label=ENC_Algorithm[3], color='#0485d1')
    rects5 = ax.bar(x_alg_5, Algorithm_data[:, 4], width, label=ENC_Algorithm[4], color='#d2bd0a')
    ax.set_xticks(x_base + 2 * (width + spacing_within_group))
    ax.set_xticklabels(Cases)
    ax.set_xlabel('Cases', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
              handleheight=1, handlelength=1, fontsize=10)
    ax.spines['top'].set_color('lightgray')
    ax.spines['top'].set_linewidth(0.0)
    ax.spines['right'].set_color('lightgray')
    ax.spines['right'].set_linewidth(0.0)
    ax.spines['left'].set_color('lightgray')
    ax.spines['left'].set_linewidth(0.0)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Cases vs Key Sensitivity Algorithm Comparision')
    plt.tight_layout()
    path = "./Results/Key_sen_Enc_alg.png"
    plt.savefig(path)
    plt.show()

    Method_data = Graphs_Time[:, :]
    width = 0.15
    spacing_within_group = 0.05
    group_spacing = 0.5
    x_base = np.arange(len(Cases)) * (5 * width + group_spacing)
    x_alg_1 = x_base
    x_alg_2 = x_base + (width + spacing_within_group)
    x_alg_3 = x_base + 2 * (width + spacing_within_group)
    x_alg_4 = x_base + 3 * (width + spacing_within_group)
    x_alg_5 = x_base + 4 * (width + spacing_within_group)
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x_alg_1, Method_data[:, 0], width, label=ENC_Methods[0], color='#fe01b1')
    rects2 = ax.bar(x_alg_2, Method_data[:, 1], width, label=ENC_Methods[1], color='#c65102')
    rects3 = ax.bar(x_alg_3, Method_data[:, 2], width, label=ENC_Methods[2], color='#a9f971')
    rects4 = ax.bar(x_alg_4, Method_data[:, 3], width, label=ENC_Methods[3], color='#0485d1')
    rects5 = ax.bar(x_alg_5, Method_data[:, 4], width, label=ENC_Methods[4], color='#d2bd0a')
    ax.set_xticks(x_base + 2 * (width + spacing_within_group))
    ax.set_xticklabels(Cases)
    ax.set_xlabel('Cases', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
              handleheight=1, handlelength=1, fontsize=10)
    ax.spines['top'].set_color('lightgray')
    ax.spines['top'].set_linewidth(0.0)
    ax.spines['right'].set_color('lightgray')
    ax.spines['right'].set_linewidth(0.0)
    ax.spines['left'].set_color('lightgray')
    ax.spines['left'].set_linewidth(0.0)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Cases vs Key Sensitivity Encryption Algorithm Comparision')
    plt.tight_layout()
    path = "./Results/Key_sen_Enc_mtd.png"
    plt.savefig(path)
    plt.show()


def Plot_ENC_DEC():
    Y_Labels = ['Encryption time (S)', 'Decryption time (S)', 'Total Computational time (S)', 'Memory size (KB)']
    ENC_Algorithm = ['MRA-O-MA-ABE', 'FSA-O-MA-ABE', 'AOA-O-MA-ABE', 'PFOA-O-MA-ABE', 'MES-PFOA-O-MA-ABE']
    ENC_Methods = ['ECC', 'RSA', 'DES', 'MA-ABE', 'MES-PFOA-O-MA-ABE']
    Time_encryp = np.load('Time_encryp.npy', allow_pickle=True)
    Time_decryp = np.load('Time_decryp.npy', allow_pickle=True)
    Total_comput_time = np.load('Total_comput_time.npy', allow_pickle=True)
    Memory_size = np.load('Memory_Size.npy', allow_pickle=True)
    Values = [Time_encryp, Time_decryp, Total_comput_time, Memory_size]
    Block_size = ['5', '10', '15', '20', '25']
    for n in range(len(Values)):
        Time = Values[n]
        Graph = Time[:5, :]
        Graph_Time = Time[5:, :]

        Algorithm_data = Graph[:, :]
        width = 0.15
        spacing_within_group = 0.05
        group_spacing = 0.5
        x_base = np.arange(len(Block_size)) * (5 * width + group_spacing)
        x_alg_1 = x_base
        x_alg_2 = x_base + (width + spacing_within_group)
        x_alg_3 = x_base + 2 * (width + spacing_within_group)
        x_alg_4 = x_base + 3 * (width + spacing_within_group)
        x_alg_5 = x_base + 4 * (width + spacing_within_group)
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x_alg_1, Algorithm_data[:, 0], width, label=ENC_Algorithm[0], color='#fe01b1')
        rects2 = ax.bar(x_alg_2, Algorithm_data[:, 1], width, label=ENC_Algorithm[1], color='#c65102')
        rects3 = ax.bar(x_alg_3, Algorithm_data[:, 2], width, label=ENC_Algorithm[2], color='#a9f971')
        rects4 = ax.bar(x_alg_4, Algorithm_data[:, 3], width, label=ENC_Algorithm[3], color='#0485d1')
        rects5 = ax.bar(x_alg_5, Algorithm_data[:, 4], width, label=ENC_Algorithm[4], color='#d2bd0a')
        ax.set_xticks(x_base + 2 * (width + spacing_within_group))
        ax.set_xticklabels(Block_size)
        ax.set_xlabel('Block Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(Y_Labels[n], fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
                  handleheight=1, handlelength=1, fontsize=10)
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        ax.spines['left'].set_color('lightgray')
        ax.spines['left'].set_linewidth(0.0)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Block Size vs ' + Y_Labels[n])
        plt.tight_layout()
        path = "./Results/Block Size_vs_%s_enc_AlG.png" % (Y_Labels[n])
        plt.savefig(path)
        plt.show()

        Method_data = Graph_Time[:, :]
        width = 0.15
        spacing_within_group = 0.05
        group_spacing = 0.5
        x_base = np.arange(len(Block_size)) * (5 * width + group_spacing)
        x_alg_1 = x_base
        x_alg_2 = x_base + (width + spacing_within_group)
        x_alg_3 = x_base + 2 * (width + spacing_within_group)
        x_alg_4 = x_base + 3 * (width + spacing_within_group)
        x_alg_5 = x_base + 4 * (width + spacing_within_group)
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x_alg_1, Method_data[:, 0], width, label=ENC_Methods[0], color='#fe01b1')
        rects2 = ax.bar(x_alg_2, Method_data[:, 1], width, label=ENC_Methods[1], color='#c65102')
        rects3 = ax.bar(x_alg_3, Method_data[:, 2], width, label=ENC_Methods[2], color='#a9f971')
        rects4 = ax.bar(x_alg_4, Method_data[:, 3], width, label=ENC_Methods[3], color='#0485d1')
        rects5 = ax.bar(x_alg_5, Method_data[:, 4], width, label=ENC_Methods[4], color='#d2bd0a')
        ax.set_xticks(x_base + 2 * (width + spacing_within_group))
        ax.set_xticklabels(Block_size)
        ax.set_xlabel('Block Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(Y_Labels[n], fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
                  handleheight=1, handlelength=1, fontsize=10)
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        ax.spines['left'].set_color('lightgray')
        ax.spines['left'].set_linewidth(0.0)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Block Size vs ' + Y_Labels[n])
        plt.tight_layout()
        path = "./Results/Block Size_vs_%s_enc_MTD.png" % (Y_Labels[n])
        plt.savefig(path)
        plt.show()


def plot_Block_Chain_Security():
    Terms = ['Transactions per Second (TPS)', 'Transaction Latency (s)', 'Security (%)', 'Cost per Transaction']
    Stats = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Graph_Term = [0, 1, 2, 3]
    Algorithm = ['AHISM-B', 'BTFAT', 'B-DQCM', 'BUDTP - RPR', 'B-ADCRF']
    NL = np.load('RES.npy', allow_pickle=True)
    for n in range(NL.shape[0]):
        for k in range(NL.shape[1]):
            Graphs = NL[n, k, :, :]

            Graph = np.zeros((Graphs.shape[-2], 5))
            for j in range(len(Algorithm)):
                Graph[j, :] = Statastical(Graphs[j, :])

            Method_data = Graph
            width = 0.15
            spacing_within_group = 0.05
            group_spacing = 0.5
            x_base = np.arange(len(Stats)) * (5 * width + group_spacing)
            x_alg_1 = x_base
            x_alg_2 = x_base + (width + spacing_within_group)
            x_alg_3 = x_base + 2 * (width + spacing_within_group)
            x_alg_4 = x_base + 3 * (width + spacing_within_group)
            x_alg_5 = x_base + 4 * (width + spacing_within_group)
            fig, ax = plt.subplots(figsize=(12, 6))
            rects1 = ax.bar(x_alg_1, Method_data[0, :], width, label=Algorithm[0], color='#fe01b1')
            rects2 = ax.bar(x_alg_2, Method_data[1, :], width, label=Algorithm[1], color='#c65102')
            rects3 = ax.bar(x_alg_3, Method_data[2, :], width, label=Algorithm[2], color='#a9f971')
            rects4 = ax.bar(x_alg_4, Method_data[3, :], width, label=Algorithm[3], color='#0485d1')
            rects5 = ax.bar(x_alg_5, Method_data[4, :], width, label=Algorithm[4], color='#d2bd0a')
            ax.set_xticks(x_base + 2 * (width + spacing_within_group))
            ax.set_xticklabels(Stats)
            ax.set_xlabel('Statistical', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_Term[k]], fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, markerscale=1,
                      handleheight=1, handlelength=1, fontsize=10)
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            ax.spines['left'].set_color('lightgray')
            ax.spines['left'].set_linewidth(0.0)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Block Size vs ' + Terms[Graph_Term[k]])
            plt.tight_layout()
            path = "./Results/Block_chain_%s_bar_enc.png" % (Terms[Graph_Term[k]])
            plt.savefig(path)
            plt.show()


def Plot_encryption():
    plot_Attacks()
    plot_key_sencitivity()
    Plot_ENC_DEC()


if __name__ == '__main__':
    plot_convergence()
    plot_Crypto_convergence()
    ROC_curve()
    Plot_Confusion()
    Plot_Activation()
    Plot_KFold()
    Plot_encryption()
    plot_Block_Chain_Security()
