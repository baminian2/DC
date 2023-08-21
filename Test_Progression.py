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


    for i in range(N_tot):
        if (stats.eleves[i].tests[0].Present or True):
            #if (stats.eleves[i].tests[0].Score_G >= 0.5 and stats.eleves[i].tests[0].Score_G <0.75):  #0.2
            #if (stats.eleves[i].tests[0].Score_G >= 0.75):  #0.2
            #if (stats.eleves[i].tests[0].Score_G < 0.5 ):  #0.2
            if True:
                labeled.append(i);
    '''
    for i in range(N_tot):
        print(stats.eleves[i].Note)
        #if (stats.eleves[i].Note < 4):
        if (stats.eleves[i].Note >5.0):
        #if (stats.eleves[i].Note > 4 and stats.eleves[i].Note <=5):
            labeled.append(i);
    '''

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
    Mean_RS_Pros =  np.empty(N_tests);
    Mean_RS_Gilles =  np.empty(N_tests);
    Mean_RS_Mod =  np.empty(N_tests);
    Mean_RS_Mod_Cont =  np.empty(N_tests);

    Mean_Score = np.empty(N_tests);

    for i in range(N_tests):
        RS = stats.RS_Calib[labeled,i].flatten();
        RS = RS[np.logical_not(np.isnan(RS))];
        RS_Pros = stats.RS_Pros[labeled,i].flatten();
        RS_Gilles = stats.RS_Gilles[labeled,i].flatten();
        RS_Mod = stats.RS[labeled,i].flatten();
        RS_Mod_Cont = stats.RS_Mod_Continu[labeled,i].flatten();
        Calibration = stats.calibration[labeled,i].flatten();
        Calibration = Calibration[np.logical_not(np.isnan(Calibration))];
        Score_G = stats.Score[labeled,i].flatten();
        Score_G = Score_G[np.logical_not(np.isnan(Score_G))];
        Score_DC = stats.Score_DC[labeled,i].flatten();
        Score_DC = Score_DC[np.logical_not(np.isnan(Score_DC))];
        Score_idl = stats.Score_Ideal[labeled,i].flatten();
        Score_idl = Score_idl[np.logical_not(np.isnan(Score_idl))];
        RS_all[i] = RS
        Calibration_all[i] = Calibration
        Score_G_all[i] = Score_G
        Score_DC_all[i] = Score_DC
        Score_idl_all[i] = Score_idl
        Score_Diff[i] = Score_idl - Score_DC

        Mean_RS[i] = np.nanmean(RS)
        Mean_RS_Mod[i] = np.nanmean(RS_Mod)
        Mean_RS_Mod_Cont[i] = np.nanmean(RS_Mod_Cont)
        Mean_RS_Gilles[i] = np.nanmean(RS_Gilles)
        Mean_RS_Pros[i] = np.nanmean(RS_Pros)

        Mean_Score[i] = np.nanmean(Score_G)

    for i in range(N_tests):
        F, p = sc.stats.shapiro(Calibration_all[i])
        p_Calibration_Shapiro[i] = p
        print('Shapiro Test for calibration test ', i, ': ', p)
        meanCalib = np.mean(Calibration_all[i])
        if p>0.05:
            print('Calibration Test ', i, ' could be gaussian. One sample t-test')
            F, p = sc.stats.ttest_1samp(Calibration_all[i],0)
            p_Calibration_0[i] = p; F_Calibration_0[i] = F; mean_Calibration_0[i] = meanCalib;
            if p>0.05:
                print('p-value: ', p)
            else:
                print('Significant -------------------------> p-value: ', p, ' , F-value: ', F, ' , Size1: ', len(Calibration_all[i]))
        else:
            print('Calibration Test ', i, ' . Wilcox test')
            F, p = sc.stats.wilcoxon(Calibration_all[i])
            p_Calibration_0[i] = p; F_Calibration_0[i] = F; mean_Calibration_0[i] = meanCalib;
            if p>0.05:
                print('p-value: ', p)
            else:
                print('Significant -------------------------> p-value: ', p, ' , F-value: ', F, ' , Size1: ', len(Calibration_all[i]))
        print('Mean Calibration: ', meanCalib)


    F0, p0 = sc.stats.shapiro(RS_all[i])
    p_RS_Shapiro[0] = p0
    print('Shapiro Test for realisme test ', i, ': ', p0)
    for i in range(1,N_tests):
        F, p = sc.stats.shapiro(RS_all[i])
        p_RS_Shapiro[i] = p
        diff = np.mean(RS_all[i]) - np.mean(RS_all[0])
        print('Shapiro Test for realisme test ', i, ': ', p)
        RS0 = stats.RS_Calib[labeled, 0].flatten();
        RSi = stats.RS_Calib[labeled, i].flatten();
        tmp_present = np.logical_not(np.any([np.isnan(RS0),np.isnan(RSi)],0))
        RS0 = RS0[tmp_present];
        RSi = RSi[tmp_present];
        if p>0.05 and p0>0.05:
            print('Réalisme Test ', i, ' could be gaussian. Depedent t-test.')
            F, p = sc.stats.ttest_rel(RS0, RSi)
            p_RS_Progress[i] = p; F_RS_Progress[i] = F ; Diff_RS_Progress[i] = diff;
            Size_test[i] = len(RSi)
            if p>0.05:
                print('p-value: ', p)
            else:
                print('Significant -------------------------> p-value: ', p, ' , F-value: ', F, ' , Size1: ', len(RS_all[0]), ' , Size2: ', len(RS_all[i]))
            print('Diff = ' , diff)
        else:
            print('Réalisme Test ', i, '. Wilcox t-test.')
            #F, p = sc.stats.mannwhitneyu(RS_all[0], RS_all[i])
            F, p = sc.stats.wilcoxon(RS0,RSi)
            p_RS_Progress[i] = p; F_RS_Progress[i] = F ; Diff_RS_Progress[i] = diff; Diff2_RS_Progress[i] = np.mean(RSi-RS0);
            Size_test[i] = len(RSi)
            if p>0.05:
                print('p-value: ', p)
            else:
                print('Significant -------------------------> p-value: ', p, ' , F-value: ', F, ' , Size1: ', len(RS0), ' , Size2: ', len(RSi))
            print('Diff = ' , diff)


    print('Calibration: Shapiro / One sample test p / F / mean')
    print(np.round(p_Calibration_Shapiro,3))
    print(np.round(p_Calibration_0,3))
    print(np.round(F_Calibration_0,2))
    print(np.round(mean_Calibration_0,3))
    print('Realisme: Shapiro/ two sample test 0 vs i: p / F / Diff of mean / mean of diff / number ')
    print(np.round(p_RS_Shapiro,3))
    print(np.round(p_RS_Progress,3))
    print(np.round(F_RS_Progress,2))
    print(np.round(Diff_RS_Progress,3))
    print(np.round(Diff2_RS_Progress,3))
    print(np.rint(Size_test))


    label_x =("TF","TS1","TA1","TA2", "TS2", "TA3","TS3");
    x = np.arange(len(label_x)) + 1;

    plt.figure('Note moyenne par test')
    plt.plot(x, Mean_RS/Mean_Score, '--b', label = 'Score standard', marker=".", markersize=10)
    #plt.plot(x,Mean_RS, '--r', label = 'Réalisme', marker=".", markersize=10)
    plt.xticks(x, label_x)
    plt.xlabel('Tests')
    plt.ylabel('Score vs Réalisme')
    #plt.ylim(0,1)

    plt.figure('Moyenne réalisme par test')
    plt.plot(x,Mean_RS, '--r', label = 'utilisant la Centration (CS)', marker=".", markersize=10)
    plt.plot(x, Mean_RS_Mod_Cont, '--C6', label='Prosperi modifiée continue', marker=".", markersize=10)
    plt.plot(x, Mean_RS_Mod, '--b', label='Prosperi modifiée discrète', marker=".", markersize=10)
    plt.plot(x, Mean_RS_Gilles, '--g', label= 'Gilles', marker=".", markersize=10)
    plt.plot(x,Mean_RS_Pros, '--C1', label= 'Prosperi', marker=".", markersize=10)
    plt.xticks(x, label_x)
    plt.grid(True)
    plt.xlabel('Tests')
    plt.ylabel('Réalisme')
    plt.ylim(0.5,1)
    plt.legend(loc = 'lower right')

    '''
    for i in range(N_tests):
        plt.figure(i)
        plt.hist(RS_all[i],bins=20)
    '''

    x0 = [0, len(x)+1]
    y0 = np.zeros(len(x0))
    y1 = np.ones(len(x0))
    plt.figure('Réalisme')
    plt.boxplot([RS_all[0],RS_all[1],RS_all[2],RS_all[3],RS_all[4],RS_all[5],RS_all[6]], showmeans = True)
    plt.plot(x0,y1, '--C7')
    plt.xlabel('Tests')
    plt.ylabel('Réalisme')
    plt.xticks(x, label_x)
    plt.title('Réalisme par test')
    plt.figure('Calibration')
    plt.boxplot([Calibration_all[0],Calibration_all[1],Calibration_all[2],Calibration_all[3],Calibration_all[4],Calibration_all[5],Calibration_all[6]], showmeans = True)
    plt.plot(x0, y0, '--C7')
    plt.xlabel('Tests')
    plt.ylabel('Calibration')
    plt.title('Calibration par test')
    plt.xticks(x, label_x)


    plt.show()