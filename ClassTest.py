import numpy as np
from math import sqrt, factorial, pow, floor
np.seterr(invalid='ignore')
import ClassParameter

def confidenceWilson(ns, n):
    #n = ups + downs

    if n == 0:
        return np.array([0.0])

    z = 1.281551565545;
    z2 = z*z;

    N = float(n);
    Ns = float(ns)
    gauche = (Ns+0.5*z2)/(N+z2);
    droite = z/(N+z2)*sqrt( Ns*(N-Ns)/N + z2/4.0);
    #left = p + 1/(2*n)*z*z
    #right = z*sqrt(p*(1-p)/n + z*z/(4*n*n))
    #under = 1+1/n*z*z
    a = gauche - droite;
    b = gauche + droite;
    return np.array([a,b]);
def confidenceGillesModified(ns, n):
    if n == 0:
        return np.array([0.0])
    N = float(n);
    Ns = float(ns)
    p0 = Ns/N;

    a = p0;
    b = p0;
    return np.array([a,b])
def confidenceProsperiModified(ns, n, DC_intervals):
    if n == 0:
        return np.array([0.0])
    N = float(n);
    Ns = float(ns)
    p_hat = Ns/N;
    d = (DC_intervals[1]- DC_intervals[0])*0.5;
    a = p_hat - d;
    b = p_hat + d;

    return np.array([a,b])
def BinomialCoefficient(n,k):
    return factorial(n)/factorial(k)/factorial(n-k)

def BinomialDistribution(ns,n, p):
    return BinomialCoefficient(n,ns)*pow(p,ns)*pow(1-p,n-ns);
def BinomialError(ns, n, p):
    if (n==0):
        return 0.0;
    mode = floor((n+1)*p);
    tmp = 0.0
    return abs(mode-ns)
    if (ns == mode):
        return 0.0;
    elif (ns < mode):
        for i in range(int(mode-ns)):
            tmp += BinomialCoefficient(n,ns+i)*pow(p,ns+i)*pow(1-p,n-ns-i);
        return tmp;
    else:
        for i in range(int(ns-mode)):
            tmp += BinomialCoefficient(n,ns-i)*pow(p,ns-i)*pow(1-p,n-ns+i);
        return tmp;
    #a = BinomialCoefficient(n,ns)*pow(p,ns)*pow(1-p,n-ns);
    #b = BinomialCoefficient(n,mode)*pow(p,mode)*pow(1-p,n-mode);
    #return a/b;
def BinomialError2(ns, n, VC):
    if (n==0):
        return 0.0;
    tmp = 0.0;
    p_hat = float(ns)/float(n);
    for i in range(int(n)):
        tmp += abs(BinomialDistribution(i,n,p_hat)-BinomialDistribution(i,n,VC));
    return tmp*0.5

def Distance_Intervals(x,y):
    if (x[0] > y[1]):
        return x[0]-y[1];
    elif (x[1] < y[0]):
        return y[0]-x[1];
    else:
        return 0.0;
def Distance_Intervals_rel(x,y):
    if (x[0] > y[1]):
        return x[0]-y[1];
    elif (x[1] < y[0]):
        return x[1]-y[0];
    else:
        return 0.0;

def Give_DC(p,param):
    for i in range(len(param.VC)):
        if (p <= 0.0):
            return 0;
        if (p >= 1.0):
            return 5;
        if (p >param.DC_intervals[i][0]  and p <= param.DC_intervals[i][1]):
            return i
class Test:
    ID = None;
    N_questions = None;
    Pts_questions = None;
    Pts_total = None;
    Pts_obtenu = None;
    DC = None;

    NC = None;                      #size of N_DC
    NU = None;                      #size of N_DC
    TE = None;                      #size of N_DC
    NC_cont = None;                      #size of N_DC
    NU_cont = None;                      #size of N_DC
    TE_cont = None;
    RS = None;
    RS_Gilles =None;
    RS_Pros = None;
    RS_Calib = None;
    RS_Mod_Continu = None;


    Note = None;
    Note_DC = None;
    Note_seuil= None;
    Note_ideal = None;
    Score_G = None;
    Score_DC = None;
    Score_Seuil = None;
    Score_ideal = None;
    TEG = None;
    TEG_sup = None;
    TEG_moy = None;
    TEG_inf = None;

    calibration = None;
    centration = None;

    Present = False;                     #size of N_questions
    label_questions = None;         #size of N_questions
    label_taxonomie = None;
    reponse_correct = None;


    Pts_total_Qrom = 0.0;
    Pts_total_Qrol = 0.0;

    NC_Qrom = None;                      #size of N_DC
    NU_Qrom = None;                      #size of N_DC
    TE_Qrom = None;                      #size of N_DC
    RS_Qrom = None;
    Note_Qrom = 0.0;
    MC = 0.0;


    NC_Qrol = None;                      #size of N_DC
    NU_Qrol = None;                      #size of N_DC
    TE_Qrol = None;                      #size of N_DC
    RS_Qrol = None;
    Note_Qrol = 0.0;

    meanTE = None;
    meanTE_Qrom = None;
    meanTE_Qrol = None;

    def computeCentration(self):
        return self.TEG_moy-self.TEG;

    def computeCalibration(self):
        return Distance_Intervals_rel([self.TEG_inf,self.TEG_sup], [self.TEG,self.TEG])
    def computeNCNUTE(self, param):
        for i in range(self.N_questions):
            deg_conf = self.DC[i];
            self.NU[deg_conf] += 1;
            self.NU_cont[deg_conf] += self.Pts_questions[i]
            self.NC_cont[deg_conf] += self.Pts_obtenu[i]
            if (self.Pts_obtenu[i] / self.Pts_questions[i] >= param.seuil):
                self.reponse_correct[i] = True;
                self.NC[deg_conf] += 1;
            else:
                self.reponse_correct[i] = False;

        self.TE = np.divide(self.NC, self.NU);
        #self.TE[np.where( self.NU == 0)] = None;

    def computeNCNUTE_label(self):
        self.Pts_total_Qrom = 0.0;
        self.Pts_total_Qrol = 0.0;
        for i in range(self.N_questions):
            deg_conf = self.DC[i];
            if (self.label_questions[i] == 'QROM'):
                self.NU_Qrom[deg_conf] += 1;
                self.Pts_total_Qrom += self.Pts_questions[i];
                if (self.reponse_correct[i]):
                    self.NC_Qrom[deg_conf] += 1;
            else:
                self.NU_Qrol[deg_conf] += 1;
                self.Pts_total_Qrol += self.Pts_questions[i];
                if (self.reponse_correct[i]):
                    self.NC_Qrol[deg_conf] += 1;

        self.TE_Qrom = np.divide(self.NC_Qrom,self.NU_Qrom)
        #self.TE_Qrom[np.where( self.NU_Qrom == 0)] = None;

        self.TE_Qrol = np.divide(self.NC_Qrol,self.NU_Qrol)
        #self.TE_Qrol[np.where( self.NU_Qrol == 0)] = None;


    def computeScore(self):
        return np.sum(self.Pts_obtenu)/self.Pts_total;
    def computeNote(self):
        note = self.computeScore()* 5.0 + 1.0;
        return np.rint(np.nextafter(note*2.0, note*2.0+1))/2.0

    def computeScore_seuil(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            DC = self.DC[i];
            if (self.reponse_correct[i]):
                tmp +=self.Pts_questions[i]*param.DC_points[0][DC]/20.0;
            else:
                tmp +=self.Pts_questions[i]*param.DC_points[1][DC]/20.0;

        tmp = np.maximum(tmp,0)
        return tmp/self.Pts_total;
    def computeNote_seuil(self, param):
        note =  self.computeScore_seuil(param)* 5.0 + 1.0;
        return np.rint(np.nextafter(note*2.0, note*2.0+1))/2.0

    def computeScore_DC(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            DC = self.DC[i];
            p_p = param.DC_points[0][DC];
            p_m = param.DC_points[1][DC];
            tmp +=self.Pts_questions[i]*(self.Pts_obtenu[i]/self.Pts_questions[i]*(p_p-p_m) + p_m)/20.0;
        tmp = np.maximum(tmp, 0)
        return tmp/self.Pts_total;
    def computeScore_Ideal(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            p = self.Pts_obtenu[i]/self.Pts_questions[i];
            DC = Give_DC(p,param)

            p_p = param.DC_points[0][DC];
            p_m = param.DC_points[1][DC];
            tmp +=self.Pts_questions[i]*(p*(p_p-p_m) + p_m)/20.0;
        return tmp/self.Pts_total;

    def computeNote_DC(self,param):
        note = self.computeScore_DC(param)* 5.0 + 1.0;
        return np.rint(np.nextafter(note*2.0, note*2.0+1))/2.0

    def computeTEG(self):
        return self.Score_G;

    def computeTEG_sup(self, param):
        tmp= 0.0;
        for i in range(self.N_questions):
            DC = self.DC[i];
            tmp += self.Pts_questions[i]*param.DC_intervals[DC][1];
        return tmp/self.Pts_total;

    def computeTEG_moy(self, param):
        tmp= 0.0;
        for i in range(self.N_questions):
            DC = self.DC[i];
            tmp += self.Pts_questions[i]*param.VC[DC];
        return tmp/self.Pts_total;

    def computeTEG_inf(self, param):
        tmp= 0.0;
        for i in range(self.N_questions):
            DC = self.DC[i];
            tmp += self.Pts_questions[i]*param.DC_intervals[DC][0];
        return tmp/self.Pts_total;



    def computeNote_JF(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            DC = self.DC[i];
            if (self.reponse_correct[i]):
                tmp +=self.Pts_questions[i];
            else:
                tmp += 0.0 ; #self.Pts_questions[i]*param.DC_points[1][DC]/20.0;
        return tmp/self.Pts_total * 5.0 + 1.0;
    def setNote(self):
        self.Note = self.computeNote();

    def computeRS(self, param):
        error = np.zeros(len(self.NU));
        for i in range(len(self.NU)):
            conf_interval = confidenceProsperiModified(self.NC[i], self.NU[i], param.DC_intervals[i]);
            # conf_intervals = confidence3(self.NC[i], self.NU[i], i, param);
            if (len(conf_interval) == 1):
                error[i] = 0.0;
            else:
                error[i] =  Distance_Intervals(conf_interval, param.DC_intervals[i]);
        self.RS = (0.95 - np.sum(error * self.NU) / self.N_questions) / 0.95;
    def computeRS_Mod_Continu(self, param):
        error = np.zeros(len(self.NU));
        for i in range(len(self.NU)):
            conf_interval = confidenceProsperiModified(self.NC_cont[i], self.NU_cont[i], param.DC_intervals[i]);
            # conf_intervals = confidence3(self.NC[i], self.NU[i], i, param);
            if (len(conf_interval) == 1):
                error[i] = 0.0;
            else:
                error[i] =  Distance_Intervals(conf_interval, param.DC_intervals[i]);
        self.RS_Mod_Continu = (0.95 - np.sum(error * self.NU_cont) / self.Pts_total) / 0.95;
    def computeRS_Gilles(self, param):
        error = np.zeros(len(self.NU));
        for i in range(len(self.NU)):
            conf_interval = confidenceGillesModified(self.NC[i], self.NU[i]);
            # conf_intervals = confidence3(self.NC[i], self.NU[i], i, param);
            if (len(conf_interval) == 1):
                error[i] = 0.0;
            else:
                error[i] =  Distance_Intervals(conf_interval, [param.VC[i],param.VC[i]]);
        self.RS_Gilles = ((1.0 - np.sum(error * self.NU) / self.N_questions) - param.beta)*param.alpha;
    def computeRS_Pros(self, param):
        error = np.zeros(len(self.NU));
        for i in range(len(self.NU)):
            conf_interval = confidenceWilson(self.NC[i], self.NU[i]);
            # conf_intervals = confidence3(self.NC[i], self.NU[i], i, param);
            if (len(conf_interval) == 1):
                error[i] = 0.0;
            else:
                error[i] =  Distance_Intervals(conf_interval, param.DC_intervals[i]);
        self.RS_Pros = (0.95 - np.sum(error * self.NU) / self.N_questions) / 0.95;

    def computeRS_Calib(self):
        error = np.abs(self.calibration);
        self.RS_Calib = 1-error;

    def computeRS_label(self, NC, NU, param):
        error = np.zeros(len(NU));
        for i in range(len(NU)):
            conf_interval = confidenceProsperiModified(NC[i], NU[i], param.DC_intervals[i]);
            # conf_intervals = confidence3(self.NC[i], self.NU[i], i, param);
            if (len(conf_interval) == 1):
                error[i] = 0.0;
            else:
                error[i] =  Distance_Intervals(conf_interval, param.DC_intervals[i]);

        N_questions = np.nansum(NU);
        return (0.95 - np.sum(error * NU) / N_questions) / 0.95;

    def __init__(self, id, data_eleve, param, label = None, label_type = None):
        self.ID = id;
        N_tmp = (data_eleve.shape[0])//3;
        if label_type==None:
            labeled = np.full(N_tmp, True, dtype=bool)
        else:
            labeled = np.full(N_tmp, False, dtype=bool)
            for i in range(N_tmp):
                if label[2+3*i] > 5.5:
                    labeled[i] = True

        if label_type=='QROM':
            labeled = np.logical_not(labeled)

        self.N_questions = np.sum(labeled);

        if (np.isnan(data_eleve[2]) or  self.N_questions == 0):
            self.Present = False;
        else:
            self.Pts_questions = np.zeros(self.N_questions);
            self.Pts_obtenu = np.zeros(self.N_questions);
            self.DC = np.zeros(self.N_questions, dtype=int);
            self.label_questions = np.empty(self.N_questions, dtype = 'U10')
            self.label_taxonomie = np.empty(self.N_questions, dtype = float)
            self.reponse_correct = np.empty(self.N_questions, dtype = bool)
            self.Present = True;
            self.Pts_total = 0.0;
            ii = 0;
            for i in range(0,len(labeled)): #self.N_questions):
                if labeled[i]:
                    #print('i and ii', i, ii)
                    self.Pts_questions[ii] = data_eleve[3*i];
                    self.Pts_obtenu[ii] = data_eleve[3*i + 1];         ### check this line too
                    self.Pts_total += self.Pts_questions[ii];
                    self.DC[ii] = int(data_eleve[3*i + 2]);
                    ii += 1
                    #self.label_taxonomie[ii] = label[2+3*i];
                #if (self.label_taxonomie[i]>4.6):
                #    self.label_questions[i] = 'QROL'
                #else:
                #    self.label_questions[i] = 'QROM'

            self.NU = np.zeros(len(param.VC));  # Nombre de d'utilisation du degrée de certitude (0,...,5)
            self.NC = np.zeros(len(param.VC));
            self.TE = np.zeros(len(param.VC));

            self.NU_cont = np.zeros(len(param.VC));  # Nombre de d'utilisation du degrée de certitude (0,...,5)
            self.NC_cont = np.zeros(len(param.VC));
            self.TE_cont = np.zeros(len(param.VC));

            self.NU_Qrom = np.zeros(len(param.VC));  # Nombre de d'utilisation du degrée de certitude (0,...,5)
            self.NC_Qrom = np.zeros(len(param.VC));
            self.TE_Qrom = np.zeros(len(param.VC));

            self.NU_Qrol = np.zeros(len(param.VC));  # Nombre de d'utilisation du degrée de certitude (0,...,5)
            self.NC_Qrol = np.zeros(len(param.VC));
            self.TE_Qrol = np.zeros(len(param.VC));

            self.computeNCNUTE(param);
            #self.setNote();
            self.computeRS(param);
            self.computeRS_Pros(param);
            self.computeRS_Gilles(param);

            self.Score_G = self.computeScore();
            self.Score_ideal = self.computeScore_Ideal(param);
            self.Score_seuil = self.computeScore_seuil(param);
            self.Score_DC = self.computeScore_DC(param);
            self.Note = self.computeNote();
            self.Note_seuil = self.computeNote_seuil(param);
            self.Note_DC = self.computeNote_DC(param);
            self.Note_JF  = self.computeNote_JF(param);

            self.computeNCNUTE_label();
            self.RS_Qrom = self.computeRS_label(self.NC_Qrom, self.NU_Qrom, param);
            self.RS_Qrol = self.computeRS_label(self.NC_Qrol, self.NU_Qrol, param);

            self.meanTE = np.nanmean(self.TE);
            self.meanTE_Qrom = np.nanmean(self.TE_Qrom);
            self.meanTE_Qrol = np.nanmean(self.TE_Qrol);


            self.TEG = self.computeTEG();
            self.TEG_sup = self.computeTEG_sup(param)
            self.TEG_moy = self.computeTEG_moy(param)
            self.TEG_inf = self.computeTEG_inf(param)

            self.centration = self.computeCentration();
            self.calibration = self.computeCalibration();
            self.computeRS_Calib();
            self.computeRS_Mod_Continu(param);

