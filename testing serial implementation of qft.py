import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from scipy import sparse
from scipy import integrate
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
import csv
import time
import random

def CPhase(n,theta,i,k):
    # controlled phase gate between i-th and j-th qubit in a circuit
    return sparse.identity(2**n) + sparse.csc_matrix((np.array([np.exp(-1j*2*np.pi*theta)]), (np.array([x]), np.array([y]))), shape=(2**n, 2**n))

def all_CPhase(n):
    # return array with all CPhase gates we will need
    CPhase_gates = []
    delta_theta = np.random.normal(0, 1, n^2) #model noise in gate
    # thetas = []
    for i in range(n):
        for j in range(n):
            theta_ij = 2 * np.pi / 2 ** (j - i + 1)
            # thetas.append(theta_ij*(1+delta_theta[i,j]))
            CPhase_gates.append(CPhase(n,theta_ij*(1+delta_theta[i,j]),i,j))
    return CPhase_gates

def QFT_series(n):
    state = sparse.csc_matrix((np.array(1), (np.array([0]), np.array([0]))), shape=(2**n, 1))
    for i in range(len(state)):
        for j in range(len(state)):
            atom_loss_prob = np.random.uniform(0,1,)

def Hamming_weight(a,b):
#     a,b are sparse column vectors with the same dimension
    weight = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            weight += 1
    return weight


def eval_QFT_series_error(n, trials):
    for m in range(trials):
        init_state = sparse.csc_matrix((np.array(1), (np.array([0]), np.array([0]))), shape=(2 ** n, 1))
        fin_state = sparse.csc_matrix((np.array(1), (np.array([0]), np.array([0]))), shape=(2 ** n, 1))
        error_l2_norm = Hamming_weight(init_state, fin_state)
