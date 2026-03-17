import numpy as np
from Global_Vars import Global_Vars
from Model_ADCRF import Model_ADCRF
from OMA_BE import OMA_BE


def objfun(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_ADCRF(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Fitn[i] = (1 / Eval[16]) + Eval[9]  # (1 / CSI) + FNR
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_ADCRF(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
        Fitn = (1 / Eval[16]) + Eval[9]  # (1 / CSI) + FNR
        return Fitn



def objfun_Crypt(Soln):
    Data = Global_Vars.Data
    Blocks = Global_Vars.Blocks
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, Encrypt_data, Decrypt_data = OMA_BE(Data, Blocks, sol=sol)
            Fitn[i] = Eval[2] + Eval[3]  # Time + Memory Size
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, Encrypt_data, Decrypt_data = OMA_BE(Data, Blocks, sol=sol)
        Fitn = Eval[2] + Eval[3]  # Time + Memory Size
        return Fitn

