import scipy as sc
import ClassTest
from ClassParameter import *
from ClassTest import *
from ClassEleve import *
from ClassStatistics import *
from ClassInitialization import *
import pandas as pd
import xlsxwriter
import openpyxl

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
        param[k].seuil = 0.45;
        N_tot += param[k].N_eleves;

    N_tests = param[0].N_tests;
    stats = Statistic_Data(data, sexes, param);

    writer = pd.ExcelWriter('Data_baminian_seuil_45.xlsx', engine='xlsxwriter')
    label_x = ("TF", "TS1", "TA1", "TA2", "TS2", "TA3", "TS3");
    for i in range(N_tests):
        Page = [];
        for k in range(len(sample)):
            for j in range(param[k].N_eleves):
                line = [];
                eleve = stats.eleves[k*param[0].N_eleves + j]
                test = eleve.tests[i]
                if (test.Present):
                    test = eleve.tests[i]
                    if k == 0:
                        line.append("A")
                    else:
                        line.append("B")
                    line.append(stats.eleves[k*param[0].N_eleves + j].ID)
                    for q in range(test.N_questions):
                        if ((test.Pts_obtenu[q] /test.Pts_questions[q]) > param[0].seuil):
                            line.append(1)
                        else:
                            line.append(0)
                        line.append(test.DC[q])
                    Page.append(line)


        df = pd.DataFrame(Page)
        df.to_excel(writer, sheet_name=label_x[i], index=False, header=False)

    writer.save()