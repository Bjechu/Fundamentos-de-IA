# Modulos necesarios
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#=========================
# Minimos CUadrados
#=========================
def minimos_cuadrados(X,Y):
    N = len(X)
    sumx = np.sum(X)
    sumy = np.sum(Y)
    sumXY = np.sum(X*Y)
    sumx2 = np.sum(X*X)
    w1 = (N*sumxy - sumx*sumy)/(N*sumx2 - sumx*sumx)
    w0 = (sumy - w1*sumx)/N
    Ybar = w0 + w1*X
#================================
# Programa Principal 
#================================
if __name__ == ¨__main__¨