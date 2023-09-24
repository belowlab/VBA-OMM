import numpy as np


def f_model(X, th, u, inF):
    # Function defining the C-Peptide OMM

    k01 = inF["k01"]
    k12 = inF["k12"]
    k21 = inF["k21"]

    n_theta = 5
    n = 3

    t = u[0, 0] # time
    G = u[1, 0] # glucose
    dG = u[2, 0] # change in glucose
    GLP1 = u[3, 0] # GLP1

    SR = f_SR(X,th,dG)[0]

    # Model Equations
    dx = np.zeros((n, 1))
    dx[0, 0] = -(k01+k21)*X[0] + k12*X[1] + SR
    dx[1, 0] = k21*X[0] - k12*X[1]
    dx[2, 0] = -np.exp(-th[0]) * (X[2] - np.exp(th[1])*1e-3 * (G - np.exp(th[2])))
    dx[3, 0] = SR * (th[4] * GLP1 + 1) # th[4] is Pi, new term we're solving for

    # Derivatives of the ODEs w.r.t
        # Model states
    dFdX = np.zeros((n, n))
    dFdX[0, 0] = -(k01+k21)
    dFdX[0, 1] = k12
    dFdX[0, 2] = 1
    dFdX[1, 0] = k21
    dFdX[1, 1] = -k12
    dFdX[1, 2] = 0
    dFdX[2, 0] = 0
    dFdX[2, 1] = 0
    dFdX[2, 2] = -np.exp(-th[0])
        # Evolution parameters
    dFdTh = np.zeros((n, n_theta))
    dFdTh[0, 0] = 0
    dFdTh[0, 1] = 0
    dFdTh[0, 2] = 0
    dFdTh[0, 3] = get_dTh_3_4(th,dG)
    dFdTh[0, 4] = 0
    dFdTh[1, 0] = 0
    dFdTh[1, 1] = 0
    dFdTh[1, 2] = 0
    dFdTh[1, 3] = 0
    dFdTh[1, 4] = 0
    dFdTh[2, 0] = np.exp(-th[0]) * (X[2] - np.exp(th[1])*1e-3 * (G - np.exp(th[2])))
    dFdTh[2, 1] = np.exp(-th[0]) * np.exp(th[1])*1e-3 * (G - np.exp(th[2]))
    dFdTh[2, 2] = -np.exp(-th[0]) * np.exp(th[1])*1e-3 * np.exp(th[2])
    dFdTh[2, 3] = 0
    dFdTh[2, 4] = 0
    dFdTh[3, 0] = 0
    dFdTh[3, 1] = 0
    dFdTh[3, 2] = 0 # Deriving with respect to glucose, maybe wrong
    dFdTh[3, 3] = 0
    dFdTh[3, 4] = SR * (GLP1 + 1)
    
    return dx, dFdX, dFdTh


def f_obs(X, phi, u, inG):
    # Observation of the first c-peptide state

    n_phi = 0       # No of Observation Parameters
    nY = 1          # No of Observations
    n = 3           # Model order

    # Observation Equation
    gx = np.zeros((nY, 1))
    gx[0, 0] = X[0]

    # Derivatives of the Observation equation w.r.t
        # - Model states
    dGdX = np.zeros((nY, n))
    dGdX[0, 0] = 1
    dGdX[0, 1] = 0
    dGdX[0, 2] = 0
        # - Observation Parameters
    dGdPhi = np.zeros((nY, n_phi))

    return gx, dGdX, dGdPhi


def f_SR(X,th,dG):

    kD = np.exp(th[3])*1e-3
    SRs = X[2]    
    if dG >= 0:
        SRd = kD*dG
    else:
        SRd = 0
    
    SR = SRs + SRd

    return SR, SRs, SRd

def f_SR_GLP(GLP1: float, SR: float, Pi: float):
    """
    Calculates the secretion rate of GLP-1 based on
    the secretion rate of C-peptide, current GLP1 concentration
    and GLP1-sensitvity index (Pi)
    """

    return SR * (Pi * GLP1 + 1)
    

def get_dTh_3_4(th,dG):

    if dG>0:
        dth3 = np.exp(th[3])*1e-3*dG
    else:
        dth3 = 0
    
    return dth3



