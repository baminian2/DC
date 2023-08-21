import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from math import sqrt, factorial, pow, floor

import scipy as sc
import ClassTest
from ClassParameter import *
from ClassTest import *
from ClassEleve import *
from ClassStatistics import *
from ClassInitialization import *



np.set_printoptions(suppress = True)

if __name__ == '__main__':
    sample = ['ClassA', 'ClassB'];
    #sample = ['ClassA'];
    #sample = ['TestData'];
    data, sexes, label_questions = Init_data(sample);
    param = np.empty(len(sample), dtype=Parametres)
    N_tot = 0;
    for k in range(len(sample)):
        param[k] = Parametres(data[k]);
        param[k].seuil = 0.55;
        N_tot += param[k].N_eleves;

    N_tests = param[0].N_tests;
    stats = Statistic_Data(data, sexes,param);


    labeled = [];
    labeled_b = [];
    labeled_m = [];
    labeled_h = [];


    for i in range(N_tot):
        if (stats.eleves[i].tests[0].Present):
            if (stats.eleves[i].tests[0].Score_G < 0.5 ): labeled_b.append(i)
            elif (stats.eleves[i].tests[0].Score_G >= 0.5 and stats.eleves[i].tests[0].Score_G <0.75): labeled_m.append(i);
            elif (stats.eleves[i].tests[0].Score_G >= 0.75): labeled_h.append(i);


    print('Number of students: ', len(labeled))
    RS_all = {};
    Calibration_all = {};
    Score_G_all = {};
    Score_DC_all = {};
    Score_idl_all = {};
    Score_Diff = {};

    p_RS_Shapiro = np.empty(N_tests)
    p_RS_Progress = np.empty(N_tests)
    F_RS_Progress = np.empty(N_tests)
    Diff_RS_Progress = np.empty(N_tests)
    Diff2_RS_Progress = np.empty(N_tests)

    p_Calibration_Shapiro = np.empty(N_tests)
    p_Calibration_0 = np.empty(N_tests)
    F_Calibration_0 = np.empty(N_tests)
    mean_Calibration_0 = np.empty(N_tests)

    Size_test = np.empty(N_tests)

    Mean_RS =  np.empty(N_tests);
    Mean_RS_b =  np.empty(N_tests);
    Mean_RS_m =  np.empty(N_tests);
    Mean_RS_h =  np.empty(N_tests);
    Mean_score =  np.empty(N_tests);
    Mean_score_b =  np.empty(N_tests);
    Mean_score_m =  np.empty(N_tests);
    Mean_score_h =  np.empty(N_tests);

    for i in range(N_tests):
        RS = stats.RS_Calib[:,i].flatten();
        RS_b = stats.RS_Calib[labeled_b,i].flatten();
        RS_m = stats.RS_Calib[labeled_m,i].flatten();
        RS_h = stats.RS_Calib[labeled_h,i].flatten();


        score = stats.Score[:,i].flatten();
        score_b = stats.Score[labeled_b,i].flatten();
        score_m = stats.Score[labeled_m,i].flatten();
        score_h = stats.Score[labeled_h,i].flatten();

        RS_all[i] = RS


        Mean_RS[i] = np.nanmean(RS)
        Mean_RS_b[i] = np.nanmean(RS_b)
        Mean_RS_m[i] = np.nanmean(RS_m)
        Mean_RS_h[i] = np.nanmean(RS_h)

        Mean_score[i] = np.nanmean(score)
        Mean_score_b[i] = np.nanmean(score_b)
        Mean_score_m[i] = np.nanmean(score_m)
        Mean_score_h[i] = np.nanmean(score_h)


    RS_tmp = stats.RS_Calib.flatten()
    RS_tmp =RS_tmp[np.logical_not(np.isnan(RS_tmp))]

    plt.figure('Histogramme')
    plt.hist(RS_tmp,50, density=True)
    plt.xlabel('Réalisme')
    plt.ylabel('Proportion')

    label_x =("TF","TS1","TA1","TA2", "TS2", "TA3","TS3");
    x = np.arange(len(label_x)) + 1;

    plt.figure('Moyenne réalisme par test')
    plt.plot(x,Mean_RS, '--r', label = 'Tous', marker=".", markersize=10)
    plt.plot(x, Mean_RS_b, '--b', label='Basse', marker=".", markersize=10)
    plt.plot(x, Mean_RS_m, '--g', label='Moyenne', marker=".", markersize=10)
    plt.plot(x, Mean_RS_h, '--C1', label= 'Haute', marker=".", markersize=10)
    plt.xticks(x, label_x)
    plt.grid(True)
    plt.xlabel('Tests')
    plt.ylabel('Réalisme')
    plt.ylim(0.75,1)
    plt.legend(loc = 'lower right')

    plt.figure('Note moyenne par test')
    plt.plot(x,(Mean_RS-Mean_score)/Mean_RS, '--r', label = 'Tous', marker=".", markersize=10)
    plt.plot(x, (Mean_RS_b-Mean_score_b)/Mean_RS_b, '--b', label='Basse', marker=".", markersize=10)
    plt.plot(x, (Mean_RS_m-Mean_score_m)/Mean_RS_m, '--g', label='Moyenne', marker=".", markersize=10)
    plt.plot(x, (Mean_RS_h-Mean_score_h)/Mean_RS_h, '--C1', label= 'Haute', marker=".", markersize=10)
    plt.xticks(x, label_x)
    plt.xticks(x, label_x)
    plt.xlabel('Tests')
    plt.ylabel('Score vs Réalisme')
    plt.show()