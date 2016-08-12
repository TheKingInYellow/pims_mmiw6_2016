# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:04:37 2016

@author: gaudre
"""
from __future__ import division, print_function
import matplotlib
import numpy as np
from scipy.integrate import quad
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import minimize_scalar
#import seaborn as sns


# define all time related constants
T = 60 # Time of one period
dt = .01   #time step in cumsum method

#sns.set(font_scale=1, rc={'text.usetex' : True}, font="serif")
# Returns the Solution to our Model for a whole Period starting with Green Light
def qG(t,q0,RHO,LAMBDA,MU,G,Y,T):
    qG = max([(LAMBDA-RHO)*G+q0,0.0])
    qY = max([(LAMBDA-MU)*Y+qG,0.0])
    if t >=0 and t < G:
        return max([(LAMBDA-RHO)*t+q0,0.0])
    elif t >= G and t < G+Y:
        return max([(LAMBDA-MU)*(t-G)+qG,0.0])
    elif t >= G+Y and t <= T:
        return LAMBDA*(t-G-Y)+qY
qG = np.vectorize(qG,otypes=[np.float64], excluded=['q0','RHO','LAMBDA','MU','Tg','Y','T'])
# Returns the Solution to our Model for a whole Period starting with Red Light
def qR(t,q0,RHO,LAMBDA,MU,G,Y,T,C):
    qY = max([(LAMBDA-RHO)*(T-G-2*Y-C)+LAMBDA*(G+Y+C)+q0,0.0])
    qZ = max([(LAMBDA-MU)*(Y-C)+qY,0.0])
    if t >=0 and t < G+Y+C:
        return LAMBDA*t+q0
    elif t >= G+Y+C and t < T-Y:
        return max([(LAMBDA-RHO)*(t-G-Y-C)+LAMBDA*(G+Y+C)+q0,0.0])
    elif t >= T-Y and t <= T-C:
        return max([(LAMBDA-MU)*(t-T+Y)+qY,0.0])
    elif t >= T-C and t <= T:
        return LAMBDA*(t-T+C)+qZ
qR = np.vectorize(qR,otypes=[np.float64], excluded=['q0','RHO','LAMBDA','MU','Tg','Y','T','C'])

# Returns the Average waiting time for any one lane
def tauG(q0,RHO,LAMBDA,MU,G,Y,T):
    val1 = quad(qG,a=0.0,b=G,args=(q0,RHO,LAMBDA,MU,G,Y,T))[0]
    val2 = quad(qG,a=G,b=G+Y,args=(q0,RHO,LAMBDA,MU,G,Y,T))[0]
    val3 = quad(qG,a=G+Y,b=T,args=(q0,RHO,LAMBDA,MU,G,Y,T))[0]
    qbar = (val1 + val2 + val3)/T
    return (qbar+1)/(RHO*G+MU*Y)
def tauR(q0,RHO,LAMBDA,MU,G,Y,T,C):
    val1 = quad(qR,a=0.0,b=G+Y+C,args=(q0,RHO,LAMBDA,MU,G,Y,T,C))[0]
    val2 = quad(qR,a=G+Y+C,b=T-Y,args=(q0,RHO,LAMBDA,MU,G,Y,T,C))[0]
    val3 = quad(qR,a=T-Y,b=T-C,args=(q0,RHO,LAMBDA,MU,G,Y,T,C))[0]
    val4 = quad(qR,a=T-C,b=T,args=(q0,RHO,LAMBDA,MU,G,Y,T,C))[0]
    qbar = (val1 + val2 + val3 + val4)/T
    return (qbar+1)/(RHO*(T-G-2*Y-2*C)+MU*Y)

# Returns the Weighted Average waiting time for all four lanes
def WAWT(G,q0_Vec,RHO_Vec,LAMBDA_Vec,MU_Vec,C,Y,T):
    tau0 = tauG(q0_Vec[0],RHO_Vec[0],LAMBDA_Vec[0],MU_Vec[0],G,Y,T)
    tau1 = tauR(q0_Vec[1],RHO_Vec[1],LAMBDA_Vec[1],MU_Vec[1],G,Y,T,C)
    tau2 = tauG(q0_Vec[2],RHO_Vec[2],LAMBDA_Vec[2],MU_Vec[2],G,Y,T)
    tau3 = tauR(q0_Vec[3],RHO_Vec[3],LAMBDA_Vec[3],MU_Vec[3],G,Y,T,C)
    return (LAMBDA_Vec[0]*tau0 + LAMBDA_Vec[1]*tau1 + \
     LAMBDA_Vec[2]*tau2 + LAMBDA_Vec[3]*tau3)/sum(LAMBDA_Vec)

# Finds the optimal Green times for all 4 lanes by minimizing WAWT
def OPTIM_G(q0_Vec,RHO_Vec,LAMBDA_Vec,MU_Vec,C,Y,T):
    minG = 5.5
    G0 = minimize_scalar(WAWT, args=(q0_Vec,RHO_Vec,LAMBDA_Vec,MU_Vec,C,Y,T), bounds=(0+minG, T-minG), method='bounded').x
    G1 = T-2*Y-2*C-G0
    G2 = G1
    G3 = G2
    R0 = T-Y-G0
    R1 = T-Y-G1
    R2 = T-Y-G2
    R3 = T-Y-G3
    G = np.array([G0,G1,G2,G3])
    R = np.array([R0,R1,R2,R3])
    return G,R

# determine rho for random cases
def giveRho(q):
    rho = np.random.uniform(low=0.5, high=1)
    buses = np.random.binomial(q, 0.05)
    rcars = np.random.binomial(q-buses, 0.3)
    rbus = np.random.binomial(buses, 0.3)
    return np.max(((rho-rcars*0.01-rbus*0.02-(buses-rbus)*0.01), 0.01))
giveRho = np.vectorize(giveRho,otypes=[np.float64])

def giveMu(q):
    mu = np.random.uniform(low=0.08, high=0.1)
    buses = np.random.binomial(q, 0.05)
    rcars = np.random.binomial(q-buses, 0.3)
    rbus = np.random.binomial(buses, 0.3)
    return np.max(((mu-rcars*0.01-rbus*0.02-(buses-rbus)*0.01), 0.01))
giveMu = np.vectorize(giveMu,otypes=[np.float64])

# Running a random examples
def main(q0,T,Y,C,i):
    lamb = np.random.uniform(low=0.01, high=0.5, size=4)
    rho = giveRho(q0)
    mu =  giveMu(q0)
    G,R = OPTIM_G(q0,rho,lamb,mu,C,Y,T)

    time = np.asarray(np.arange(0,T,dt))
    queues = []
    queue0 = qG(time,q0[0],rho[0],lamb[0],mu[0],G[0],Y,T)
    queue1 = qR(time,q0[1],rho[1],lamb[1],mu[1],G[0],Y,T,C)
    queue2 = qG(time,q0[2],rho[2],lamb[2],mu[2],G[0],Y,T)
    queue3 = qR(time,q0[3],rho[3],lamb[3],mu[3],G[0],Y,T,C)
    fig = plt.plot(time,queue0, 'k-', label="Street 1", alpha=0.8)
    plt.plot(time, queue1, 'k--', label="Street 2", alpha=0.8)
    plt.plot(time, queue2, 'k-', label="Street 3", alpha=0.8)
    plt.plot(time, queue3, 'k--', label="Street 4", alpha=0.8)
    ax = plt.gca()
    ax.set_title("Queue Length vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Length (\# Cars)")
    print(G[0])
    plt.savefig(str(i)+".png")
    plt.clf()



def generatePlots(time, queues, GreenLightTime):
    fig = plt.plot(time,queues[:,0], 'k-', label="Street 1", alpha=0.8)
    plt.plot(time, queues[:,1], 'k--', label="Street 2", alpha=0.8)
    ax = plt.gca()
    ax.set_title("Queue Size vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Size (\# Cars)")

    points1 = np.array([time, queues[:,0]]).T.reshape(-1,1,2)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
    cmap1 = ListedColormap(['g','r'])
    lc1 = LineCollection(segments1, cmap=cmap1, \
            norm=BoundaryNorm(GreenLightTime, cmap1.N), \
            linewidth = 3, linestyle = 'dashed')
    lc1.set_array(np.mod(time,T))
    ax.add_collection(lc1)

    points2 = np.array([time, queues[:,1]]).T.reshape(-1,1,2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
    cmap2 = ListedColormap(['r','g'])
    lc2 = LineCollection(segments2, cmap=cmap2, \
            norm=BoundaryNorm(GreenLightTime, cmap2.N), \
            linewidth = 3, linestyle = 'solid')
    lc2.set_array(np.mod(time,T))
    lc2.set_linewidth(3)
    ax.add_collection(lc2)

    print(GreenLightTime)

    plt.xlim(time.min(), time.max())
    ax.grid()
    plt.savefig(str(i)+".png")
    plt.close()

