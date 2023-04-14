##%%writefile /content/FINDER/code/FINDER_ND/testReal.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import graph
import csv
import numpy as  np 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def GetSolution(STEPRATIO, MODEL_FILE_CKPT):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = FINDER()
    data_test_path = '../real/'
    #data_test_name = ['Crime','HI-II-14','Digg','Enron','Gnutella31','Epinions','Facebook','Youtube','Flickr']
    data_test_name = [f for f in listdir(data_test_path) if isfile(join(data_test_path, f))]
    #model_file_path = '/content/FINDER/code/FINDER_ND/models/barabasi_albert/'
    #model_file_ckpt = MODEL_FILE_CKPT
    model_file = MODEL_FILE_CKPT
    ## save_dir
    save_dir = '../results/ND/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.arange(1*len(data_test_name)).reshape((1,len(data_test_name))),index=['time'], columns=data_test_name)
    #################################### modify to choose which stepRatio to get the solution
    stepRatio = STEPRATIO
    for j in range(len(data_test_name)):
        print ('\nTesting dataset %s'%data_test_name[j])
        data_test = data_test_path + data_test_name[j] 
        solution, time = dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        df.iloc[0,j] = time
        print('Data:%s, time:%.2f'%(data_test_name[j], time))
    save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
    df.to_csv(save_dir_local + '/sol_time.csv', encoding='utf-8', index=False)


def EvaluateSolution(STEPRATIO, MODEL_FILE_CKPT, STRTEGYID):
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
    dqn = FINDER()
    data_test_path = '../real/'
    #     data_test_name = ['Crime', 'HI-II-14', 'Digg', 'Enron', 'Gnutella31', 'Epinions', 'Facebook', 'Youtube', 'Flickr']
    data_test_name = [f for f in listdir(data_test_path) if isfile(join(data_test_path, f))]
    save_dir = '../results/ND/StepRatio_%.4f/'%STEPRATIO
    ## begin computing...
    df = pd.DataFrame(np.arange(2 * len(data_test_name)).reshape((2, len(data_test_name))), index=['solution', 'time'], columns=data_test_name)
    for i in range(len(data_test_name)):
        print('\nEvaluating dataset %s' % data_test_name[i])
        data_test = data_test_path + data_test_name[i] 
        solution = save_dir + data_test_name[i]
        t1 = time.time()
        # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
        ################################## modify to choose which strategy to evaluate
        strategyID = STRTEGYID
        score, MaxCCList = dqn.EvaluateSol(data_test, solution, strategyID, reInsertStep=0.001)
        t2 = time.time()
        df.iloc[0, i] = score
        df.iloc[1, i] = t2 - t1
        result_file = save_dir + '/MaxCCList_Strategy_' + data_test_name[i]
        with open(result_file, 'w') as f_out:
            for j in range(len(MaxCCList)):
                f_out.write('%.8f\n' % MaxCCList[j])
        '''if i < 5:
                    plt.plot(MaxCCList)
                    plt.title(data_test_name[i])
                    plt.show()'''
        print('Data: %s, score:%.6f' % (data_test_name[i], score))
    df.to_csv(save_dir + '/solution_score.csv', encoding='utf-8', index=False)

def findModel():
    NUM_MIN, NUM_MAX = 30, 50
    g_type = 'barabasi_albert'
    VCFile = './models/Model_%s/ModelVC_%d_%d.csv'%(g_type, NUM_MIN, NUM_MAX)
    vc_list = []
    with open(VCFile, newline='') as csvfile:
         #reader = csv.DictReader(csvfile)
         reader = csv.reader(csvfile, delimiter=',')
         for i, row in enumerate(reader):
            #print(row['first_name'], row['last_name'])
            vc_list.append(float(row[0]))
    start_loc = 0
    plt.plot(vc_list)
    plt.show()
    min_vc = start_loc + np.argmin(vc_list[start_loc:])
    best_model_iter = 500 * min_vc
    best_model = './models/Model_%s/nrange_%d_%d_iter_%d.ckpt' % (g_type, NUM_MIN, NUM_MAX, best_model_iter)
    return best_model


def main():
    model_file_ckpt = findModel()
    print(model_file_ckpt)
    GetSolution(0.01, model_file_ckpt)
    #EvaluateSolution(0.01, model_file_ckpt, 0)



if __name__=="__main__":
    main()