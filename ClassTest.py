import numpy as np
from math import sqrt, factorial, pow, floor
np.seterr(invalid='ignore')

# Intervelle de confiance de Wilson (mesure de Prosperi (2015))
"""
Entrées:
    ns: nombre de succées
    n: nombre d'essais
Sortie: intervalle 90% de Wilson (a,b) 
"""
def confidenceWilson(ns, n):
    if n == 0:
        return np.array([0.0])

    z = 1.281551565545;
    #z = 0.4
    #z = 0.55
    #z = 1.645
    z2 = z*z
    N = float(n)
    Ns = float(ns)
    gauche = (Ns+0.5*z2)/(N+z2)
    droite = z/(N+z2)*sqrt( Ns*(N-Ns)/N + z2/4.0)
    a = gauche - droite
    b = gauche + droite

    return np.array([a,b])

# Estimateur de la mesure de Gilles (2002)
"""
Entrées:
    ns: nombre de succées
    n: nombre d'essais
Sotite:   
    Estimateur hat(P) = ns/n
    (est un intervalle pour des raisons de programmation)
"""
def confidenceGillesModified(ns, n):
    if n == 0:
        return np.array([0.0])
    N = float(n);
    Ns = float(ns)
    p0 = Ns/N;

    a = p0;
    b = p0;
    return np.array([a,b])

# Intervalle de confiance de la mesure de Prosperi modifiées discrètes
"""
Entrées:
    ns: nombre de succées
    n: nombre d'essais
    DC_interval: intervalles de certitude pour un certain DC (Taille: 2x1)
Sotite:   
    Intervalle de confiance centré en ns/n de la taille de DC_interval
"""
def confidenceProsperiModified(ns, n, DC_interval):
    if n == 0:
        return np.array([0.0])
    N = float(n);
    Ns = float(ns)
    p_hat = Ns/N;
    d = (DC_interval[1]- DC_interval[0])*0.5;
    a = p_hat - d;
    b = p_hat + d;

    return np.array([a,b])

# Distance entre deux intervalles
"""
Entrées: Intervalles x et y
Sortie: distance absolue entre x et y 
    (0 si x intersection y non-vide, sinon plus courte distance entre x et y)
"""
def Distance_Intervals(x,y):
    if (x[0] > y[1]):
        return x[0]-y[1];
    elif (x[1] < y[0]):
        return y[0]-x[1];
    else:
        return 0.0;

# Distance relative entre deux intervalle
"""
Identique a Distance_Intervals() mais est négatif si x<y.
"""
def Distance_Intervals_rel(x,y):
    if (x[0] > y[1]):
        return x[0]-y[1];
    elif (x[1] < y[0]):
        return x[1]-y[0];
    else:
        return 0.0;

# DC correcpondant à une probabilité
"""
Entrées: 
    p: probabilité (0,1)
    param: parametres de l'échantillon 
           (utilisé pour les intervalles de certitude)
Sortie: DC correspondant à la probabilité p (0, ..., 5)
"""
def Give_DC(p,param):
    for i in range(len(param.VC)):
        if (p <= 0.0):
            return 0;
        if (p >= 1.0):
            return 5;
        if (p >param.DC_intervals[i][0]  and p <= param.DC_intervals[i][1]):
            return i

# Classe Test
"""
La classe Test représente le test d'un élèves.
Il contient toutes les informations concernant les points possibles d'obtenir, obtenus 
et le choix des DC pour un test et pour un élève.
La classe Test contient également les valeurs de centration et réalisme (pour différentes mesures)
"""
class Test:
    # Données sur le test
    Present = False                # Si l'élève est présent à ce test (Vrai ou Faux)
    ID = None                      # ID du test (mélange entre ID élève et numéro chronologique du test)
    N_questions = None             # Nombre de questions au test (Taille: 1)
    Pts_questions = None           # Nombre de points possibles d'obtenir à une question (Taille: N_questions x 1 )
    Pts_total = None               # Nombre de points possibles d'obtenir au total (Taille: 1)
    Pts_obtenu = None              # Nombre de points obtenus par questions (Taille: N_questions x 1 )
    DC = None                      # DC choisi par questions (Taille: N_questions x 1 )
    label_questions = None         # Label de la question (QROM ou QROL) (Taille: N_questions x 1 )
    label_taxonomie = None         # Nombre d'élément de la taxonomie d'Anderson par question (Taille: N_questions x 1)
    reponse_correct = None         # Réponse correcte ou incorrect (Vrai ou Faux) (Taille: N_questions x 1 )

    # Nombre d'utilisation, de réponse correcte par DC avec la version continue pour QRO
    NC = None                      # Nombre de réponses correctes par DC (Taille: 6 x 1)
    NU = None                      # Nombre d'utilisation par DC (Taille: 6 x 1)
    TE = None                      # Taux d'exactitude par DC (= NC/NU) (Taille: 6 x 1)
    NC_cont = None                 # Nombre de points obtenus par DC (Taille 6 x 1)
    NU_cont = None                 # Nombre de points possible par DC (Taille 6 x 1)
    TE_cont = None                 # Taux d'exactitude par DC (= NC_cont/NU_cont) (Taille 6 x 1 )

    # Centration et Réalismes
    centration = None              # Centration (Cs) selon Gilles (2002)
    calibration = None             # Centration (Cs) selon projet de MAS
    RS_Calib = None                # Mesure utilisant la centration
    RS_Gilles =None                # Mesure de Gilles (2002)
    RS_Pros = None                 # Mesure de Prosperi (2015)
    RS = None                      # Mesure de Prosperi Modifiée Discrète
    RS_Mod_Continu = None          # Mesure de Prosperi Modifiée Continue

    # Scores et Notes
    Score_G = None                 # Score standard
    Score_DC = None                # Score avec DC (nécessite un seuil d'acceptation)
    Score_Seuil = None             # Score avec DC mais sans seuil d'acceptation
    Score_ideal = None             # Score idéal avec DC et sans seuil (si l'élève choisi les bons DC)
    Note = None                    # Note utilisant le score standard
    Note_DC = None                 # Note utilisant le score avec DC et avec seuil
    Note_seuil= None               # Note utilisant le score avec DC et sans seuil
    Note_ideal = None              # Note utilisant le score idéal avec DC et sans seuil

    # Taux d'exactitude
    TEG = None                     # Taux d'exactitude général (équivalent au score général)
    TEG_sup = None                 # Estimation du TEG supérieur (le meilleur des cas)
    TEG_moy = None                 # Estimation du TEG moyen
    TEG_inf = None                 # Estimation du TEG inférieur (le pire de cas)

    # Centration selon Gilles (2002)
    def computeCentration(self):
        return self.TEG_moy-self.TEG

    # Centration selon projet de MAS
    def computeCalibration(self):
        return Distance_Intervals_rel([self.TEG_inf,self.TEG_sup], [self.TEG,self.TEG])

    # Calcule NC, NU, TE
    """
    Intialise reponse_correct d'après le seuil d'acceptation défini dans les paramètres (cf. class Parametres)
    Calcule NC, NU, TE (avec seuil) et les versions continues (sans seuil)
    """
    def computeNCNUTE(self, param):
        for i in range(self.N_questions):
            deg_conf = self.DC[i]
            self.NU[deg_conf] += 1
            self.NU_cont[deg_conf] += self.Pts_questions[i]
            self.NC_cont[deg_conf] += self.Pts_obtenu[i]
            if (self.Pts_obtenu[i] / self.Pts_questions[i] >= param.seuil):
                self.reponse_correct[i] = True
                self.NC[deg_conf] += 1
            else:
                self.reponse_correct[i] = False
        self.TE = np.divide(self.NC, self.NU)

    # Score standard
    def computeScore(self):
        return np.sum(self.Pts_obtenu)/self.Pts_total

    # Note standard
    def computeNote(self):
        note = self.computeScore()* 5.0 + 1.0
        return np.rint(np.nextafter(note*2.0, note*2.0+1))/2.0

    # Score avec DC et avec un seuil d'acceptation
    def computeScore_seuil(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            DC = self.DC[i]
            if (self.reponse_correct[i]):
                tmp +=self.Pts_questions[i]*param.DC_points[0][DC]/20.0
            else:
                tmp +=self.Pts_questions[i]*param.DC_points[1][DC]/20.0

        tmp = np.maximum(tmp, 0)
        return tmp/self.Pts_total

    # Note avec DC et avec un seuil d'acceptation
    def computeNote_seuil(self, param):
        note =  self.computeScore_seuil(param)* 5.0 + 1.0;
        return np.rint(np.nextafter(note*2.0, note*2.0+1))/2.0

    # Score avec DC et sans un seuil d'acceptation
    def computeScore_DC(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            DC = self.DC[i];
            p_p = param.DC_points[0][DC];
            p_m = param.DC_points[1][DC];
            tmp +=self.Pts_questions[i]*(self.Pts_obtenu[i]/self.Pts_questions[i]*(p_p-p_m) + p_m)/20.0;
        tmp = np.maximum(tmp, 0)
        return tmp/self.Pts_total;

    # Score idéal avec DC et sans un seuil d'acceptation
    def computeScore_Ideal(self, param):
        tmp = 0.0
        for i in range(self.N_questions):
            p = self.Pts_obtenu[i]/self.Pts_questions[i];
            DC = Give_DC(p,param)

            p_p = param.DC_points[0][DC];
            p_m = param.DC_points[1][DC];
            tmp +=self.Pts_questions[i]*(p*(p_p-p_m) + p_m)/20.0;
        return tmp/self.Pts_total;

    # Note avec DC et sans un seuil d'acceptation
    def computeNote_DC(self,param):
        note = self.computeScore_DC(param)* 5.0 + 1.0;
        return np.rint(np.nextafter(note*2.0, note*2.0+1))/2.0

    def computeTEG(self):
        return self.Score_G
    def computeTEG_sup(self, param):
        tmp= 0.0;
        for i in range(self.N_questions):
            DC = self.DC[i]
            tmp += self.Pts_questions[i]*param.DC_intervals[DC][1]
        return tmp/self.Pts_total
    def computeTEG_moy(self, param):
        tmp= 0.0;
        for i in range(self.N_questions):
            DC = self.DC[i]
            tmp += self.Pts_questions[i]*param.VC[DC]
        return tmp/self.Pts_total
    def computeTEG_inf(self, param):
        tmp= 0.0;
        for i in range(self.N_questions):
            DC = self.DC[i]
            tmp += self.Pts_questions[i]*param.DC_intervals[DC][0]
        return tmp/self.Pts_total

    def setNote(self):
        self.Note = self.computeNote()

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
        error = np.ones(len(self.NU));
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
        self.ID = id
        # Nombre de questions
        N_tmp = (data_eleve.shape[0])//3
        # Si pas de label: considérer toutes les questions
        # Sinon: label True pour les QUROL (5 ou plus d'élément de la taxonomie)
        if label_type==None:
            labeled = np.full(N_tmp, True, dtype=bool)
        else:
            labeled = np.full(N_tmp, False, dtype=bool)
            for i in range(N_tmp):
                if label[2+3*i] > 4.5:
                    labeled[i] = True
        # Si seul les QROM sont considéré
        if label_type=='QROM':
            labeled = np.logical_not(labeled)

        self.N_questions = np.sum(labeled)

        # Vérifie si l'élève est présent au test (pas de réponse (NaN) pour le premier DC)
        if (np.isnan(data_eleve[2]) or  self.N_questions == 0):
            self.Present = False
        else:
            self.Pts_questions = np.zeros(self.N_questions)
            self.Pts_obtenu = np.zeros(self.N_questions)
            self.DC = np.zeros(self.N_questions, dtype=int)
            self.label_questions = np.empty(self.N_questions, dtype = 'U10')
            self.label_taxonomie = np.empty(self.N_questions, dtype = float)
            self.reponse_correct = np.empty(self.N_questions, dtype = bool)
            self.Present = True
            self.Pts_total = 0.0
            ii = 0
            for i in range(0,len(labeled)):
                if labeled[i]:
                    self.Pts_questions[ii] = data_eleve[3*i];
                    self.Pts_obtenu[ii] = data_eleve[3*i + 1];
                    self.Pts_total += self.Pts_questions[ii];
                    self.DC[ii] = int(data_eleve[3*i + 2]);
                    ii += 1

            self.NU = np.zeros(len(param.VC));
            self.NC = np.zeros(len(param.VC));
            self.TE = np.zeros(len(param.VC));

            self.NU_cont = np.zeros(len(param.VC));
            self.NC_cont = np.zeros(len(param.VC));
            self.TE_cont = np.zeros(len(param.VC));

            self.computeNCNUTE(param);


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

            self.TEG = self.computeTEG();
            self.TEG_sup = self.computeTEG_sup(param)
            self.TEG_moy = self.computeTEG_moy(param)
            self.TEG_inf = self.computeTEG_inf(param)

            self.centration = self.computeCentration();
            self.calibration = self.computeCalibration();
            self.computeRS_Calib();
            self.computeRS_Mod_Continu(param);

