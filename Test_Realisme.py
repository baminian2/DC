
from ClassStatistics import *
from ClassInitialization import *

import pandas as pd
import numpy as np
import seaborn
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

#np.set_printoptions(suppress = True)

if __name__ == '__main__':
    sample = ['ClassA', 'ClassB'];
    # sample = ['ClassA'];
    # sample = ['TestData'];
    data, sexes, label_questions = Init_data(sample);
    param = np.empty(len(sample), dtype=Parametres)
    N_tot = 0;
    for k in range(len(sample)):
        param[k] = Parametres(data[k]);
        param[k].seuil = 0.55;
        N_tot += param[k].N_eleves;

    N_tests = param[0].N_tests;
    stats = Statistic_Data(data, sexes, param);
    labeled = [];
    labeled_b = [];
    labeled_m = [];
    labeled_h = [];


    for i in range(N_tot):
        if (stats.eleves[i].tests[0].Present):
            if (stats.eleves[i].tests[0].Score_G < 0.5 ): labeled_b.append(i)
            elif (stats.eleves[i].tests[0].Score_G >= 0.5 and stats.eleves[i].tests[0].Score_G <0.75): labeled_m.append(i);
            elif (stats.eleves[i].tests[0].Score_G >= 0.75): labeled_h.append(i);

    RS_all = [];
    labeled = range(N_tot);

    RS_ALL = stats.RS_Calib.flatten()
    RS_ALL = RS_ALL[np.logical_not(np.isnan(RS_ALL))]


    for i in range(N_tests):
        RS = stats.RS_Calib[:,i].flatten();
        RS = RS[np.logical_not(np.isnan(RS))]
        RS_b = stats.RS_Calib[labeled_b,i].flatten();
        RS_b = RS_b[np.logical_not(np.isnan(RS_b))]
        RS_m = stats.RS_Calib[labeled_m,i].flatten();
        RS_m = RS_m[np.logical_not(np.isnan(RS_m))]
        RS_h = stats.RS_Calib[labeled_h,i].flatten();
        RS_h = RS_h[np.logical_not(np.isnan(RS_h))]

        print(np.max(RS))
        RS_all.append(RS_m)


    label_x = ("TF", "TS1", "TA1", "TA2", "TS2", "TA3", "TS3");

    label_x2 = ("Insuffisant", "Faible", "Satisfaisant", "Bon", "Excellent")
    x = [0.55, 0.77, 0.87, 0.925,0.9725]
    bins = [0.0, 0.705, 0.705, 0.835, 0.835, 0.905, 0.905, 0.95, 0.95, 1]


    """
    plt.figure('Realisme hist')
    plt.hist(RS_ALL, bins, label='Tous', density=True)
    plt.hist(RS_all[0], bins, label='TF', density=True, histtype='step')
    plt.xlabel('Note')
    plt.ylabel('Fréquence des notes')
    plt.legend()
    plt.xlim(0.5,1)
    plt.xticks(x, label_x2,rotation=45)
    plt.title('Note standard général versus TF')

    plt.figure('Realisme hist all')
    for i in range(N_tests):
        tmp = i/(N_tests-1);
        tmp2 = 0.1*i
        if i == 0:
            #plt.hist(RS_all[i], bins, label=label_x[i], rwidth=1.0-tmp2, density=True, fc=(tmp, 0, 1 - tmp, 0.1))
            plt.hist(RS_all[i], bins, label=label_x[i], rwidth=1.0-tmp2, density=True, fc=(tmp, 0, 1 - tmp, 1), histtype='step')
        else:
            plt.hist(RS_all[i], bins, label=label_x[i], rwidth=1.0-tmp2, density=True, fc=(tmp, 0, 1-tmp, 0.1))
    plt.xlabel('Note')
    plt.ylabel('Fréquence du Réalisme')
    plt.legend()
    plt.xlim(0.5,1)
    plt.xticks(x, label_x2, rotation=45)
    plt.title('Réalisme versus TF')
    plt.show
    ax = seaborn.displot(RS_all, kind='kde', palette= seaborn.color_palette("hls", 7), legend=False) # seaborn.color_palette("hls", 7)
    ax.set(xlabel='Réalisme', ylabel= 'Densité')
    plt.legend(title = 'Réalisme (Rs)',loc='upper left', labels = label_x)
    plt.xlim(0.5,1)
    """

    ax2 = seaborn.kdeplot(data=RS_all, palette= seaborn.color_palette("hls", 7), legend=False, cumulative=True, common_norm=False, common_grid=True) # seaborn.color_palette("hls", 7)
    ax2.set(xlabel='Réalisme', ylabel= 'Probabilité')
    plt.legend(title = 'Réalisme (Rs)',loc='upper left', labels = label_x)
    #plt.xlim(0.5,1)

    #seaborn.displot(RS_all[1], palette= seaborn.color_palette("hls", 7), legend=False, kde=True)




    plt.show()