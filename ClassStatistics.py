import numpy as np
from ClassEleve import *


class Statistic_Data:
    param = None;
    eleves = None;

    Note = [];
    Note_Seuil = [];
    Note_DC = [];

    NoteDC = [];
    NoteDC_continu = [];
    TE = [];
    temps = [];
    TS = [];            # 1 si c'est un TS, 0 sinon
    RS = [];
    RS_Gilles = [];
    RS_Pros = [];
    RS_Calib = [];
    RS_Mod_Continu = [];
    Score = [];
    Score_Ideal = [];
    Score_DC = [];
    Time = [];
    RS_Qrom = [];
    RS_Qrol = [];
    RS_Qrom2 = [];
    RS_Qrol2 = [];
    RS_Fille = [];
    RS_Garcon = [];

    meanRS = None;
    meanRS_Qrom = None;
    meanRS_Qrol = None;

    meanTE = None;
    meanTE_Qrom = None;
    meanTE_Qrol = None;

    meanNote = None;

    DiffRS_Qrolm = None;

    MC = [];
    centration = [];
    calibration_low = None;


    def computeRS(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.RS = np.empty([N_tot, self.param[0].N_tests])
        self.RS[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.RS[i][j] = test.RS;
                j += 1;
            i += 1;
    def computeRS_Mod_Continu(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.RS_Mod_Continu = np.empty([N_tot, self.param[0].N_tests])
        self.RS_Mod_Continu[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.RS_Mod_Continu[i][j] = test.RS_Mod_Continu;
                j += 1;
            i += 1;
    def computeRS_Gilles(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.RS_Gilles = np.empty([N_tot, self.param[0].N_tests])
        self.RS_Gilles[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.RS_Gilles[i][j] = test.RS_Gilles;
                j += 1;
            i += 1;
    def computeRS_Pros(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.RS_Pros = np.empty([N_tot, self.param[0].N_tests])
        self.RS_Pros[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.RS_Pros[i][j] = test.RS_Pros;
                j += 1;
            i += 1;

    def computeRS_Calib(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.RS_Calib = np.empty([N_tot, self.param[0].N_tests])
        self.RS_Calib[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.RS_Calib[i][j] = test.RS_Calib;
                j += 1;
            i += 1;
    def computeScore(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Score = np.empty([N_tot, self.param[0].N_tests])
        self.Score[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.Score[i][j] = test.Score_G;

                j += 1;
            i += 1;

    def computeScore_DC(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Score_DC = np.empty([N_tot, self.param[0].N_tests])
        self.Score_DC[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.Score_DC[i][j] = test.Score_DC;

                j += 1;
            i += 1;
    def computeScore_Ideal(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Score_Ideal = np.empty([N_tot, self.param[0].N_tests])
        self.Score_Ideal[:] = np.nan;
        i = 0;
        j = 0;
        for eleve in self.eleves:
            j = 0;
            for test in eleve.tests:
                if (test.Present):
                    self.Score_Ideal[i][j] = test.Score_ideal;

                j += 1;
            i += 1;

    def computeTime(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Time = np.empty([N_tot, self.param[0].N_tests])
        self.Time[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.Time[i][j] = test.ID;
                j += 1;
            i += 1;

    def computeCalibration(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.calibration = np.empty([N_tot, self.param[0].N_tests])
        self.calibration[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.calibration[i][j] = test.calibration;
                j += 1;
            i += 1;

    def computeCalibration_lowMarks(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.calibration_low = np.empty([N_tot, self.param[0].N_tests])
        self.calibration_low[:] = np.nan;
        i = 0; j = 0; tmp = 0;
        for eleve in self.eleves:
            j=0;
            if (eleve.tests[0].Present):
                if(eleve.tests[0].Note < 7):
                    tmp +=1;
                    for test in eleve.tests:
                        if (test.Present) :
                            self.calibration_low[i][j] = test.calibration;
                        j += 1;
            i += 1;

    def computeCentration(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.centration = np.empty([N_tot, self.param[0].N_tests])
        self.centration[:] = np.nan;
        i = 0; j = 0;
        for eleve in self.eleves:
            j=0;
            for test in eleve.tests:
                if (test.Present) :
                    self.centration[i][j] = test.centration;
                j += 1;
            i += 1;

    def computeNote(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Note = np.empty([N_tot, self.param[0].N_tests])
        self.Note[:] = np.nan;
        i = 0;
        j = 0;
        for eleve in self.eleves:
            j = 0;
            for test in eleve.tests:
                if (test.Present):
                    self.Note[i][j] = test.Note;
                j += 1;
            i += 1;

    def computeNote_Seuil(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Note_Seuil = np.empty([N_tot, self.param[0].N_tests])
        self.Note_Seuil[:] = np.nan;
        i = 0;
        j = 0;
        for eleve in self.eleves:
            j = 0;
            for test in eleve.tests:
                if (test.Present):
                    self.Note_Seuil[i][j] = test.Note_seuil;
                j += 1;
            i += 1;

    def computeNote_DC(self):
        N_tot = 0;
        for i in range(len(self.param)):
            N_tot += self.param[i].N_eleves
        self.Note_DC = np.empty([N_tot, self.param[0].N_tests])
        self.Note_DC[:] = np.nan;
        i = 0;
        j = 0;
        for eleve in self.eleves:
            j = 0;
            for test in eleve.tests:
                if (test.Present):
                    self.Note_DC[i][j] = test.Note_DC;
                j += 1;
            i += 1;


    def computeRS_FG(self):
        for eleve in self.eleves:
            for test in eleve.tests:
                if (test.Present) :
                    if (eleve.sexe == 'F'):
                        self.RS_Fille.append(test.centration);
                    else:
                        self.RS_Garcon.append(test.centration);


    def computeDiffRS(self):
        RS_Qrom = np.array(self.RS_Qrom)
        RS_Qrol = np.array(self.RS_Qrol)
        RS_Qrom2 = RS_Qrom[np.logical_not(np.isnan(RS_Qrol))]
        RS_Qrol2 = RS_Qrol[np.logical_not(np.isnan(RS_Qrol))]
        self.DiffRS_Qrolm = (RS_Qrom2-RS_Qrol2);
        self.RS_Qrom2 = RS_Qrom2;
        self.RS_Qrol2 = RS_Qrol2;




    def __init__(self, data, sexes, param, label=None, label_type= None):
        self.param = param;
        tmp = 0;
        for i in range(len(self.param)):
            tmp += param[i].N_eleves;
        self.eleves = np.empty(tmp, dtype=Eleve);

        tmp = 0;
        for k in range(len(self.param)):
            for i in range((self.param[k].N_eleves)):
                if label==None:
                    self.eleves[i + tmp] = Eleve(i, k, data[k], sexes[k][i],self.param[k])
                else:
                    self.eleves[i + tmp] = Eleve(i, k, data[k], sexes[k][i],self.param[k], label[k], label_type)
            tmp += self.param[k].N_eleves;


        self.computeDiffRS();

        self.computeRS_FG();

        self.computeRS();
        self.computeRS_Pros();
        self.computeRS_Gilles();
        self.computeRS_Calib();
        self.computeRS_Mod_Continu();

        self.computeCentration();
        self.computeCalibration();
        self.computeCalibration_lowMarks();

        self.computeNote();
        self.computeNote_Seuil();
        self.computeNote_DC();

        self.computeScore();
        self.computeScore_DC();
        self.computeScore_Ideal();
        self.computeTime();
