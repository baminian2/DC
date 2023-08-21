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
import scipy




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sample = ['ClassA', 'ClassB'];
    #sample = ['ClassA'];
    #sample = ['TestData'];
    data, sexes, label_questions = Init_data(sample);
    param = np.empty(len(sample), dtype=Parametres)
    for k in range(len(sample)):
        param[k] = Parametres(data[k]);

    stats = Statistic_Data(data, sexes,param);

    meanRSEleve = np.nanmean(stats.RS_Calib, 1);
    meanScoreEleve= np.nanmean(stats.Score, 1);
    sdRSEleve = np.nanstd(stats.RS_Calib, 1);
    data_tmp = [];
    for i in range(len(sdRSEleve)):
        for j in range(stats.param[0].N_tests):
            if (stats.eleves[i].tests[j].Present):
                if (stats.eleves[i].tests[j].Score_G < 0.01 ):
                    print('ID eleve ', stats.eleves[i].ID, ' ID test', stats.eleves[i].tests[j].ID, ', score = ',stats.eleves[i].tests[j].Score_G ) #, 'mean realisme', meanRSEleve[i], ', std realisme', sdRSEleve[i])
    for i in range(len(sdRSEleve)):
        if (stats.eleves[i].tests[0].Present):
            if ((stats.eleves[i].tests[0].Score_G >= 0.5 and sdRSEleve[i] < 10.0) and False):  #0.2
                print('ID eleve ', stats.eleves[i].ID, ', Score = ', stats.eleves[i].tests[0].Score_G, 'mean realisme', meanRSEleve[i], ', std realisme', sdRSEleve[i])
            else:
                data_tmp.append(i);



    score = (stats.Score)[data_tmp, :].flatten()
    calib = (stats.calibration)[data_tmp, :].flatten()
    score_idl = (stats.Score_Ideal)[data_tmp, :].flatten()
    temps = (stats.Time)[data_tmp, :].flatten()
    RS = (stats.RS_Calib)[data_tmp, :].flatten()
    RS_comp = (stats.RS)[data_tmp, :].flatten()
    score = score[np.logical_not(np.isnan(RS))]
    calib = calib[np.logical_not(np.isnan(RS))]
    score_idl = score_idl[np.logical_not(np.isnan(RS))]
    temps = temps[np.logical_not(np.isnan(RS))]
    RS_comp = RS_comp[np.logical_not(np.isnan(RS))]
    RS = RS[np.logical_not(np.isnan(RS))]

    plt.figure(654)
    plt.plot(score, RS, '.k')
    plt.xlabel('Score')
    plt.ylabel('RS')

    X= np.array([score, temps])
    X = X.transpose();
    X2 = sm.add_constant(X);

    Y = RS;
    est = sm.OLS(Y, X2);
    est2 = est.fit();
    print(est2.summary())

    RS1 = stats.RS_Calib[data_tmp,0] ;#/stats.Score[data_tmp,0];
    RS1 = RS1.flatten()
    RS1 = RS1[np.logical_not(np.isnan(RS1))];
    RS2 = stats.RS_Calib[data_tmp,1];#/stats.Score[data_tmp,1];
    RS2 = RS2.flatten()
    RS2 = RS2[np.logical_not(np.isnan(RS2))];
    RS3 = stats.RS_Calib[data_tmp,2];#/stats.Score[data_tmp,2];
    RS3 = RS3.flatten()
    RS3 = RS3[np.logical_not(np.isnan(RS3))];
    RS4 = stats.RS_Calib[data_tmp,3];#/stats.Score[data_tmp,3];
    RS4 = RS4.flatten()
    RS4 = RS4[np.logical_not(np.isnan(RS4))];
    RS5 = stats.RS_Calib[data_tmp,4];#/stats.Score[data_tmp,4];
    RS5 = RS5.flatten()
    RS5 = RS5[np.logical_not(np.isnan(RS5))];
    RS6 = stats.RS_Calib[data_tmp,5];#/stats.Score[data_tmp,5];
    RS6 = RS6.flatten()
    RS6= RS6[np.logical_not(np.isnan(RS6))];
    RS7 = stats.RS_Calib[data_tmp,6];#/stats.Score[data_tmp,6];
    RS7 = RS7.flatten()
    RS7 = RS7[np.logical_not(np.isnan(RS7))];
    F, p = scipy.stats.f_oneway(RS1,RS2,RS3,RS4,RS5,RS6,RS7);
    print('Anova: F= ', F,' and p = ',p)
    print('Corrélation: ', np.corrcoef(RS,RS_comp))

    print('Test RS1 vs RS2: ', sc.stats.ttest_ind(RS1, RS2), ' Différence = ', np.nanmean(RS1)-np.nanmean(RS2))
    print('Test RS1 vs RS3: ', sc.stats.ttest_ind(RS1, RS3), ' Différence = ', np.nanmean(RS1)-np.nanmean(RS3))
    print('Test RS1 vs RS4: ', sc.stats.ttest_ind(RS1, RS4), ' Différence = ', np.nanmean(RS1)-np.nanmean(RS4))
    print('Test RS1 vs RS5: ', sc.stats.ttest_ind(RS1, RS5), ' Différence = ', np.nanmean(RS1)-np.nanmean(RS5))
    print('Test RS1 vs RS6: ', sc.stats.ttest_ind(RS1, RS6), ' Différence = ', np.nanmean(RS1)-np.nanmean(RS6))
    print('Test RS1 vs RS7: ', sc.stats.ttest_ind(RS1, RS7), ' Différence = ', np.nanmean(RS1)-np.nanmean(RS7))

    plt.figure(234)
    plt.plot(RS, np.maximum(score_idl - score,np.zeros(len(score))), '.b')
    plt.xlabel('RS')
    plt.ylabel('Error (score_idl - score)/score_idl')


    """
    X = np.array([stats.MC, stats.temps]);
    X = X.transpose();
    X2 = sm.add_constant(X);

    Y = stats.RS
    #X2 = np.ones(np.shape(Y))
    est = sm.OLS(Y, X2);
    est2 = est.fit();
    print(est2.summary())

    print(sc.stats.ttest_ind(stats.RS_Qrom, stats.RS_Qrol2))
    print('QROM: ', np.mean(stats.RS_Qrom))
    print('QROL: ', np.mean(stats.RS_Qrol2))

    print(sc.stats.ttest_ind(stats.RS_Garcon, stats.RS_Fille))
    print('Garçons: ', np.mean(stats.RS_Garcon))
    print('Fille: ', np.mean(stats.RS_Fille))
    """
    meanRS = np.nanmean(stats.RS[data_tmp, :],0);# np.nanmean(stats.RS[data_tmp, :]/stats.Score[data_tmp, :], 0);
    meanScore = np.nanmean(stats.Score[data_tmp, :], 0);
    meanRS_Gilles = np.nanmean(stats.RS_Gilles[data_tmp, :],0);#np.nanmean(stats.RS_Gilles[data_tmp, :]/stats.Score[data_tmp, :], 0);
    meanRS_Pros =np.nanmean(stats.RS_Pros[data_tmp, :] ,0);#np.nanmean(stats.RS_Pros[data_tmp, :]/stats.Score[data_tmp, :], 0);
    meanRS_Calib = np.nanmean(stats.RS_Calib[data_tmp, :],0);#np.nanmean(stats.RS_Calib[data_tmp, :]/stats.Score[data_tmp, :], 0);
    sdRS = np.nanstd(stats.RS[data_tmp, :],0);#np.nanstd(stats.RS[data_tmp, :]/stats.Score[data_tmp, :], 0);
    sdRS_Gilles = np.nanstd(stats.RS_Gilles[data_tmp, :],0);#np.nanstd(stats.RS_Gilles[data_tmp, :]/stats.Score[data_tmp, :], 0);
    sdRS_Pros = np.nanstd(stats.RS_Pros[data_tmp, :],0);#np.nanstd(stats.RS_Pros[data_tmp, :]/stats.Score[data_tmp, :], 0);



    label_x =("TF","TS1","TA1","TA2", "TS2", "TA3","TS3");
    x = np.arange(len(label_x));
    plt.figure(333)
    plt.plot(meanScoreEleve, meanRSEleve, linestyle='None', label=' Realisme ', marker="o", markersize=3)
    plt.legend()
    plt.xlabel('mean Score')
    plt.ylabel('mean RS')
    plt.figure(334)
    plt.plot(meanScoreEleve, sdRSEleve, linestyle='None', label=' Realisme ', marker="o", markersize=3)
    plt.xlabel('mean Score')
    plt.ylabel('std RS')
    plt.legend()

    label_x =("TF","TS1","TA1","TA2", "TS2", "TA3","TS3");
    x = np.arange(len(label_x));
    width = 0.25
    mlt = 0;
    realisme_means = {
        'Gilles': meanRS_Gilles ,
        'Prosperi modifié': meanRS,
        'Prosperi': meanRS_Pros,
    }
    yerror = (sdRS_Gilles, sdRS, sdRS_Pros);

    fig, ax = plt.subplots(1,1)
    for att, meas in realisme_means.items():
        offset = width*mlt;
        rects = ax.bar(x+offset, meas, yerr = yerror[mlt], width= width, label = att)
        #ax.bar_label(rects, padding=3)
        mlt +=1;
    ax.set_ylabel('Réalisme')
    ax.set_xlabel('Tests ordrés chronologiement')
    ax.set_title('Réalisme pour trois moddèles et 7 tests.')
    ax.set_xticks(x + width, label_x)
    plt.xticks(x+width, label_x)
    ax.legend()
    ax.set_ylim(0, 1)


    meanCentration = np.nanmean(stats.centration, 0);
    meanCalibration = np.nanmean(stats.calibration, 0);
    stdCentration = np.nanstd(stats.centration, 0);
    sdCalibration = np.nanstd(stats.calibration, 0);

    meanRS_cal_low = np.nanmean(1 - np.abs(stats.calibration_low), 0);
    sdRS_cal_low = np.nanstd(1-np.abs(stats.calibration_low), 0)
    plt.figure(231)
    plt.plot(score, calib, linestyle='None', label=' calibration', marker=".", markersize=15)
    plt.xlabel('Score')
    plt.ylabel('Calibation')
    plt.legend()

    meanRS_cal = np.nanmean(1 - np.abs(stats.calibration), 0);
    sdRS_cal = np.nanstd(1-np.abs(stats.calibration), 0)
    plt.figure(4)
    """
    plt.errorbar(x, meanRS_cal, yerr= sdRS_cal, label='Calibration', marker=".", markersize=15)
    plt.errorbar(x, meanRS, yerr= sdRS, label='Posperi modifié', marker=".", markersize=15)
    plt.errorbar(x, meanRS_Pros, yerr = sdRS_Pros, label='Proseri', marker=".", markersize=15)
    plt.errorbar(x, meanRS_Gilles, yerr= sdRS_Gilles, label='Gilles', marker=".", markersize=15)
    """
    plt.plot(x, meanRS_Calib, '--r', label='Calibration', marker=".", markersize=15)
    plt.plot(x, meanScore, '--C4', label='Score', marker=".", markersize=15)
    plt.plot(x, meanRS, '--b', label='Posperi modifié', marker=".", markersize=15)
    plt.plot(x, meanRS_Pros, '--g', label='Proseri', marker=".", markersize=15)
    plt.plot(x, meanRS_Gilles, '--C1', label='Gilles', marker=".", markersize=15)
    plt.title('Réalisme pour quatre moddèles (seuil de correction: '+str(param[0].seuil)+')')
    plt.xlabel('Test ordré chronologiquement')
    plt.ylabel('Réalisme')
    plt.xticks(x, label_x)
    #plt.ylim(0.5, 1)
    plt.grid(True)
    plt.legend()

    plt.figure(67)
    plt.boxplot([RS1,RS2,RS3,RS4,RS5,RS6,RS7])
    #plt.plot(x, meanScore, '--C4', label='Score', marker=".", markersize=15)
    plt.title('Réalisme pour quatre moddèles (seuil de correction: '+str(param[0].seuil)+')')
    plt.xlabel('Test ordré chronologiquement')
    plt.ylabel('Réalisme')
    plt.xticks(x, label_x)
    #plt.ylim(0.5, 1)
    #plt.grid(True)
    #plt.legend()




    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[0].hist(stats.Note.flatten(),np.linspace(1,6,11)-0.25)
    axs[1].hist(stats.Note_Seuil.flatten(),np.linspace(1,6,11)-0.25)
    axs[2].hist(stats.Note_DC.flatten(),np.linspace(1,6,11)-0.25)


    """
    plt.title('Réalisme')
    plt.xlabel('Test ordré chronologiquement')
    plt.ylabel('Réalisme')
    plt.legend()
    plt.figure(2)
    plt.xticks(x, label_x)
    plt.plot(x, stats.meanTE, label = 'Tout', marker=".", markersize=15)
    plt.plot(x, stats.meanTE_Qrom, label='QROM', marker=".", markersize=15)
    plt.plot(x, stats.meanTE_Qrol, label='QROL', marker=".", markersize=15)

    plt.title('Taux exactitude')
    plt.xlabel('Test ordré chronologiquement')
    plt.ylabel('TE')
    plt.legend()

    plt.figure(3)
    plt.xticks(x, label_x)
    plt.plot(x, stats.meanTE, label = 'TE', marker=".", markersize=15)
    plt.plot(x, stats.meanNote/6.0, label = 'Note', marker=".", markersize=15)
    plt.plot(x, stats.meanRS, label = 'RS', marker=".", markersize=15)
    plt.legend()

    plt.figure(4)
    plt.xticks(x, label_x)
    plt.plot(x, stats.meanRS - (0.7170 + 0.2975*stats.meanTE), label = 'TE', marker=".", markersize=15)
    plt.title('Error TE linreg')
    plt.legend()

    plt.figure(5)
    plt.xticks(x, label_x)
    plt.plot(x, stats.meanRS - (0.5526 + 0.0716*stats.meanNote), label = 'TE', marker=".", markersize=15)
    plt.title('Error Note lin reg')
    plt.legend()
    '''
    plt.figure(6)
    plt.xticks(x, label_x)
    plt.plot(stats.Note, '.b' ,label = ' Note', marker=".", markersize=5)
    plt.plot(stats.NoteDC, '.r' ,label = 'Note DC ', marker=".", markersize=5)
    plt.plot(stats.NoteDC_continu, '.g' ,label = 'Note DC cont', marker=".", markersize=5)
    plt.title('Error Note lin reg')
    plt.legend()

    diffDC = note_rounded - noteDC_rounded;
    diffDC = diffDC[diffDC<0.0]
    print(diffDC)
    diffDC_cont = note_rounded - noteDC_cont_rounded;
    diffDC_cont = diffDC_cont[diffDC_cont<0.0]
    print(diffDC_cont)
    plt.figure(9)
    plt.hist(note_rounded,np.linspace(1,6,11))
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[0].hist(note_rounded,np.linspace(1,6,11)-0.25)
    axs[1].hist(noteDC_rounded,np.linspace(1,6,11)-0.25)
    axs[2].hist(noteDC_cont_rounded,np.linspace(1,6,11)-0.25)
    """
    plt.show()
