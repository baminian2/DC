import ClassParameter
from ClassTest import *
import numpy as np

class Eleve:
    ID = None;
    sexe = None;
    tests = None;
    RS = None;
    Note = None;
    TE = None;
    def __init__(self, id, groupe, data, sexe, param, label = None, label_type=None):
        self.ID = id + 1000*groupe;
        self.sexe = sexe;
        self.tests = np.empty(param.N_tests, dtype=Test)
        tmp = 0.0; N_tmp = 0;
        for i in range(param.N_tests):
            if label==None:
                self.tests[i] = Test(i,data[i][id], param);
            else:
                self.tests[i] = Test(i,data[i][id], param, label[i], label_type);
            if self.tests[i].Present:
                tmp += self.tests[i].Note
                N_tmp +=1 ;

        self.Note = tmp/N_tmp;




