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
    RS = stats.RS_Calib.flatten();
    RS =RS[np.logical_not(np.isnan(RS))]

    RS_Calib = stats.RS_Calib.flatten();
    RS_Calib =RS_Calib[np.logical_not(np.isnan(RS_Calib))]
    RS_Pros = stats.RS_Pros.flatten();
    RS_Pros = RS_Pros[np.logical_not(np.isnan(RS_Pros))];
    RS_Gilles = stats.RS_Gilles.flatten();
    RS_Gilles = RS_Gilles[np.logical_not(np.isnan(RS_Gilles))];
    RS_Mod = stats.RS.flatten();
    RS_Mod= RS_Mod[np.logical_not(np.isnan(RS_Mod))];
    Calibration = stats.calibration.flatten();
    Calibration = Calibration[np.logical_not((np.isnan(Calibration)))];
    Score = stats.Score.flatten()
    Score = Score[np.logical_not((np.isnan(Score)))];
    '''
    plt.figure('RS Calib')
    plt.hist(RS_Calib)
    plt.figure('RS Gilles')
    plt.hist(RS_Gilles)
    plt.figure('RS Prosperi')
    plt.hist(RS_Pros)
    plt.figure('RS Modified')
    plt.hist(RS_Mod)
    plt.figure('Calibration')
    plt.hist(Calibration)

    plt.show()
    '''

    N = 1000;
    Diff_Calib = np.zeros(N);
    Diff_Pros = np.zeros(N);
    Diff_Gilles = np.zeros(N);
    Diff_Mod = np.zeros(N);
    Diff_Score = np.zeros(N);
    Corr_Calib = np.zeros(N);
    Corr_Pros = np.zeros(N);
    Corr_Gilles = np.zeros(N);
    Corr_Mod = np.zeros(N);
    Corr_Mod_Cont = np.zeros(N);
    Corr_Score = np.zeros(N);

    tmp = 0;
    seuils = np.linspace(0.3,1.0,N);
    for s in seuils:
        print('Seuil = ', s)
        for i in range(len(param)):
            param[i].seuil = s;
        stats_tmp = Statistic_Data(data, sexes,param);
        RS = stats_tmp.Note.flatten();
        RS = RS[np.logical_not(np.isnan(RS))]
        RS_Calib = stats_tmp.RS_Calib.flatten();
        RS_Calib = RS_Calib[np.logical_not(np.isnan(RS_Calib))]
        RS_Pros = stats_tmp.RS_Pros.flatten();
        RS_Pros = RS_Pros[np.logical_not(np.isnan(RS_Pros))];
        RS_Gilles = stats_tmp.RS_Gilles.flatten();
        RS_Gilles = RS_Gilles[np.logical_not(np.isnan(RS_Gilles))];
        RS_Mod = stats_tmp.RS.flatten();
        RS_Mod= RS_Mod[np.logical_not(np.isnan(RS_Mod))];
        RS_Mod_Cont = stats_tmp.RS_Mod_Continu.flatten();
        RS_Mod_Cont= RS_Mod_Cont[np.logical_not(np.isnan(RS_Mod_Cont))];

        Diff_Calib[tmp] = np.mean(np.abs(RS-RS_Calib))
        Diff_Pros[tmp] = np.mean(np.abs(RS-RS_Pros));
        Diff_Gilles[tmp] = np.mean(np.abs(RS-RS_Gilles));
        Diff_Mod[tmp] = np.mean(np.abs(RS-RS_Mod));
        Diff_Score[tmp] = np.mean(np.abs(RS-Score));
        print(np.max(np.abs(RS_Mod-RS_Mod_Cont)))

        Corr_Calib[tmp],p = sc.stats.pearsonr(RS, RS_Calib);
        Corr_Pros[tmp],p = sc.stats.pearsonr(RS, RS_Pros);
        Corr_Gilles[tmp],p = sc.stats.pearsonr(RS, RS_Gilles);
        Corr_Mod[tmp],p = sc.stats.pearsonr(RS, RS_Mod);
        Corr_Mod_Cont[tmp],p = sc.stats.pearsonr(RS, RS_Mod_Cont);
        Corr_Score[tmp],p = sc.stats.pearsonr(RS, Score);

        tmp += 1;

    plt.figure('Difference')
    plt.plot(seuils, Diff_Calib, label='Calibration')
    plt.plot(seuils, Diff_Mod, label='Prosperi Modifié')
    plt.plot(seuils, Diff_Pros, label='Prosperi')
    plt.plot(seuils, Diff_Gilles, label='Gilles')
    plt.grid(True)
    plt.xlabel('Seuil')
    plt.ylabel('Différence')
    plt.legend()
    plt.figure('Correlation')
    #plt.plot(seuils, Corr_Calib, label = 'Calibration')
    plt.plot(seuils, Corr_Mod_Cont, 'C6', label = 'Modifiée continue')
    plt.plot(seuils, Corr_Mod, 'b', label = 'Modifiée discrète')
    plt.plot(seuils, Corr_Gilles, 'g', label = 'Gilles')
    plt.plot(seuils, Corr_Pros, 'C1', label = 'Prosperi')
    #plt.plot(seuils, Corr_Score, label = 'Score')
    plt.grid(True)
    plt.xlabel('Seuil')
    plt.ylabel('Corrélation')
    plt.xlim(0.3,1)
    plt.ylim(0.55,0.9)
    #plt.title('Corrélation des mesures de réalisme versus le seuil de correction')
    plt.legend(loc='upper right')
    plt.figure('Correlation Score')
    plt.plot(seuils, Corr_Calib, 'r', label = 'Centration (CS)')
    plt.plot(seuils, Corr_Mod_Cont, 'C6', label = 'Modifiée continue')
    plt.plot(seuils, Corr_Mod, 'b', label = 'Modifiée discrète')
    plt.plot(seuils, Corr_Gilles, 'g', label = 'Gilles')
    plt.plot(seuils, Corr_Pros, 'C1', label = 'Prosperi')
    plt.grid(True)
    plt.xlabel('Seuil')
    plt.ylabel('Corrélation')
    plt.xlim(0.3,1)
    plt.ylim(0.35,0.65)
    #plt.title('Corrélation des mesures de réalisme versus le seuil de correction')
    plt.legend()


    plt.show()


