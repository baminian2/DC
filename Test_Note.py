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
        param[k].seuil = 0.55;

    stats = Statistic_Data(data, sexes,param);
    param09 = param;
    for k in range(len(param09)):
        param09[k].seuil = 0.9;


    stats09 = Statistic_Data(data, sexes,param09);

    Note = stats.Note.flatten();
    Note = Note[np.logical_not(np.isnan(Note))];
    Note_Seuil = stats.Note_Seuil.flatten();
    Note_Seuil = Note_Seuil[np.logical_not(np.isnan(Note_Seuil))];
    Note_Seuil09 = stats09.Note_Seuil.flatten();
    Note_Seuil09 = Note_Seuil09[np.logical_not(np.isnan(Note_Seuil09))];
    Note_DC = stats.Note_DC.flatten();
    Note_DC = Note_DC[np.logical_not(np.isnan(Note_DC))];

    Note_all = [];
    for i in range(param[0].N_tests):
        Note_tmp = stats.Note[:,i].flatten()
        Note_tmp = Note_tmp[np.logical_not(np.isnan(Note_tmp))]
        Note_all.append(Note_tmp)




    Id_seuil = np.sum( Note == Note_Seuil)
    Sup_seuil = np.sum(Note<Note_Seuil)
    Inf_seuil = np.sum(Note > Note_Seuil)
    Id_seuil09 = np.sum( Note == Note_Seuil09)
    Sup_seuil09 = np.sum(Note<Note_Seuil09)
    Inf_seuil09 = np.sum(Note > Note_Seuil09)
    Id_DC = np.sum(Note==Note_DC)
    Sup_DC = np.sum(Note<Note_DC)
    Inf_DC = np.sum(Note > Note_DC)

    Sup_Note_seuil = Note_Seuil[Note_Seuil > Note];
    Sup_Note_seuil09 = Note_Seuil09[Note_Seuil09 > Note];
    Sup_Note_DC = Note_DC[Note_DC > Note];

    label_x = ("TF", "TS1", "TA1", "TA2", "TS2", "TA3", "TS3");

    print('Note Seuil 0.6: Identique ', Id_seuil,' , Mieux ', Sup_seuil, ' , Inférieur ' , Inf_seuil, ' , Total ', len(Note))
    print('Note Seuil 0.9: Identique ', Id_seuil09,' , Mieux ', Sup_seuil09, ' , Inférieur ' , Inf_seuil09, ' , Total ', len(Note))
    print('Note DC: Identique ', Id_DC,' , Mieux ', Sup_DC, ' , Inférieur ' , Inf_DC, ' , Total ', len(Note))

    print('Mean Note ', np.mean(Note), ' , Mean Note seuil06 ', np.mean(Note_Seuil), ' , Mean Note seuil09 ', np.mean(Note_Seuil09), ' , Mean Note DC ', np.mean(Note_DC))

    '''
    plt.figure('Diff Note Seuil 0.6 hist')
    plt.hist(Sup_Note_seuil, np.linspace(1,6,11)+0.25, rwidth=0.8,label='Diff note seuil')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    plt.title('Nombre de meilleures note avec DC avec un seuil = 0.6')
    plt.figure('Diff Note Seuil 0.9 hist')
    plt.hist(Sup_Note_seuil09, np.linspace(1,6,11)+0.25, rwidth=0.8, label='Diff note seuil')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    plt.title('Nombre de meilleures note avec DC avec un seuil = 0.9')
    plt.figure('Diff  Note DC hist')
    plt.hist(Sup_Note_DC, np.linspace(1,6,11)+0.25, rwidth=0.8, label='Diff note seuil')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    plt.title('Nombre de meilleures note avec DC sans seuil')
    '''
    plt.figure('Note hist')
    plt.hist(Note,np.linspace(1,6,11)+0.25,rwidth=0.8, label='Note standard')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    plt.title('Note standard')

    plt.figure('NoteTF hist 1')
    plt.hist(Note_all[1],np.linspace(1,6,11)+0.25,rwidth=0.8, label='Post-test', density=True, lw = 1)
    plt.hist(Note_all[0],np.linspace(1,6,11)+0.25,rwidth=0.5, label='Pré-test', density=True, color='r', alpha=0.2, lw=1, ls= 'dotted')
    plt.xlabel('Note')
    plt.ylabel('Fréquence des notes')
    plt.legend()
    plt.title('Note standard général Post-test versus Pre-test')
    plt.figure('NoteTF hist 2')
    plt.hist(Note_all[4],np.linspace(1,6,11)+0.25,rwidth=0.8, label='Post-test', density=True, lw = 1)
    plt.hist(np.concatenate((Note_all[2],Note_all[3])),np.linspace(1,6,11)+0.25,rwidth=0.5, label='Pré-test', density=True, color='r', alpha=0.2, lw=1, ls= 'dotted')
    plt.xlabel('Note')
    plt.ylabel('Fréquence des notes')
    plt.legend()
    plt.title('Note standard général Post-test versus Pre-test')

    plt.figure('NoteTF hist 3')
    plt.hist(Note_all[6],np.linspace(1,6,11)+0.25,rwidth=0.8, label='Post-test', density=True, lw = 1)
    plt.hist(Note_all[5],np.linspace(1,6,11)+0.25,rwidth=0.5, label='Pré-test', density=True, color='r', alpha=0.2, lw=1, ls= 'dotted')
    plt.xlabel('Note')
    plt.ylabel('Fréquence des notes')
    plt.legend()
    plt.title('Note standard général Post-test versus Pre-test')

    plt.figure('All')
    N_tests = param[0].N_tests
    for i in range(param[0].N_tests):
        tmp = i/(N_tests-1);
        plt.hist(Note_all[i],np.linspace(1,6,11)+0.25,rwidth=0.8, label=label_x[i], density=True,fc=(tmp, 0, 1-tmp, 0.1))
    plt.xlabel('Note')
    plt.ylabel('Fréquence des notes')
    plt.legend()
    plt.title('Note standard par tests')

    plt.figure('Note Seuil 0.6 hist')
    plt.hist(Note_Seuil,np.linspace(1,6,11)+0.25, rwidth=0.8,label='Seuil: 0.55')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    #plt.title('Note DC avec seuil = 0.55')
    plt.legend()
    plt.figure('Note Seuil 0.75 hist')
    plt.hist(Note_Seuil09,np.linspace(1,6,11)+0.25, rwidth=0.8, label='Seuil: 0.9')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    #plt.title('Note DC avec seuil = 0.9')
    plt.legend()
    plt.figure('Note DC hist')
    plt.hist(Note_DC,np.linspace(1,6,11)+0.25,rwidth=0.8, label='Diff note seuil')
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    plt.title('Note DC sans seuil de correction')


    plt.figure('Note Seuil all hist')
    plt.hist(Note_Seuil,np.linspace(1,6,11)+0.25, rwidth=0.8,label='S_a = 0.55',density=True, fc=(0, 0, 1, 0.5))
    plt.hist(Note_Seuil09,np.linspace(1,6,11)+0.25, rwidth=0.7, label='S_a = 0.9',density=True, fc=(1, 0, 0, 0.5))
    plt.xlabel('Note')
    plt.ylabel('Nombre de notes')
    plt.title('Note DC avec seuil de correction')
    plt.legend()

    plt.show()

