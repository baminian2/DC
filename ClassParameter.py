import numpy as np

class Parametres:
    alpha = 20./19.;
    beta = 0.025;
    VC = np.array([0.125, 0.375, 0.6, 0.775, 0.9, 0.975]);
    DC_points = np.array([[13, 16, 17, 18, 19, 20], [4, 3, 2, 0, -6, -20]]);
    DC_intervals = np.array([[0, 0.25],[0.25, 0.5], [0.5,0.7],[0.7,0.85],[0.85,0.95],[0.95,1.0]]);
    seuil = 0.6;
    N_tests = None;
    N_eleves = None;
    def __init__(self, data):
        self.N_tests = len(data);
        self.N_eleves = data[0].shape[0];
