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
    sampleA = ['ClassA'];
    sampleB = ['ClassB'];
    dataA, sexesA, label_questionsA = Init_data(sampleA);
    paramA = np.empty(len(sampleA), dtype=Parametres)
    N_tot = 0;
    for k in range(len(sampleA)):
        paramA[k] = Parametres(dataA[k]);
        paramA[k].seuil = 0.6;
        N_tot += paramA[k].N_eleves;

    N_tests = paramA[0].N_tests;
    statsA = Statistic_Data(dataA, sexesA,paramA);



    dataB, sexesB, label_questionsB = Init_data(sampleB);
    paramB = np.empty(len(sampleB), dtype=Parametres)
    N_totB = 0;
    for k in range(len(sampleB)):
        paramB[k] = Parametres(dataB[k]);
        paramB[k].seuil = 0.6;
        N_totB += paramB[k].N_eleves;

    N_tests = paramB[0].N_tests;
    statsB = Statistic_Data(dataB, sexesB,paramB);

    RSA = statsA.RS_Calib.flatten()
    RSA = RSA[np.logical_not(np.isnan(RSA))]
    RSB = statsB.RS_Calib.flatten()
    RSB = RSB[np.logical_not(np.isnan(RSB))]
    calibA = statsA.calibration.flatten()
    calibA = calibA[np.logical_not(np.isnan(calibA))]
    calibB = statsB.calibration.flatten()
    calibB = calibB[np.logical_not(np.isnan(calibB))]

    FA,pA = sc.stats.shapiro(RSA)
    print('pA : ', pA)
    FB,pB = sc.stats.shapiro(RSB)
    print('pB : ', pB)
    FA,pA = sc.stats.shapiro(calibA)
    skwA = sc.stats.skewtest(calibA)

    print('calib pA : ', pA, ' , skewness : ', skwA)
    FB,pB = sc.stats.shapiro(calibB)
    skwB = sc.stats.skewtest(calibB)
    print('calib pB : ', pB, ' , skewness : ', skwB)

    F, p = sc.stats.mannwhitneyu(RSA,RSB)
    print('A vs B : p = ', p, ' F : ', F, ' , SizeA : ', len(RSA), ' , SizeB : ', len(RSB))
    F, p = sc.stats.mannwhitneyu(calibA,calibB)
    print('Calib A vs B : p = ', p, ' F : ', F, ' , SizeA : ', len(calibA), ' , SizeB : ', len(calibB))

    plt.figure('Classe A')
    plt.hist(calibA,20)
    plt.figure('Classe B')
    plt.hist(calibB,20)
    plt.show()

