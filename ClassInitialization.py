import pandas as pd
import numpy as np

# Initialisation des données
"""
Ignore les deux premières colonnes (Prénom, Nom)
Les cases vides sont considérées comme NaN.
Entrées: les noms des échantillons (doit être dans 
         le dossier data et de format ods.
Sorties: 
    data: les données (nombre de points au total par questions,
    nombre de points obtenus par questions, degré de certitude)
    Label_questions: première ligne du tableau excel.
    sexes: sexe des participants.
"""

def Init_data(sample):
    data = [];
    label_questions = [];
    sexes = [];
    for k in range(len(sample)):
        filename ='data/'+sample[k]+'.ods';
        print(filename)
        xls = pd.ExcelFile(filename, engine='odf')
        data_tmp = {};
        label_questions_tmp = {};
        i = 0;
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(filename, sheet_name=sheet_name)
            df = pd.DataFrame(df);
            tmp = np.delete(df.to_numpy(),[0,1],1)
            tmp_sx = tmp[1::,0]
            tmp = np.delete(tmp,0,1)
            label_questions_tmp[i] = tmp[0][:]
            data_tmp[i] = np.delete(tmp,0,0);
            i += 1;
        sexes.append(tmp_sx);
        data.append(data_tmp);
        label_questions.append(label_questions_tmp);
    return data, sexes, label_questions;