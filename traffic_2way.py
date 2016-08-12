# coding: utf-8

# import libraries
from __future__ import division, print_function
import matplotlib
import numpy as np
import scipy.integrate as spi
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import minimize_scalar
import seaborn as sns


# define all time related constants
T = 60
dt = .01    #time step in cumsum method

sns.set(font_scale=1, rc={'text.usetex' : True}, font="serif")

def tau(q0,rho,lamb,T,TIME_G):
    if TIME_G > -q0/(lamb-rho):
        qbar =  0.5*( q0**2 / (rho-lamb)  + lamb *(T-TIME_G)**2)/T
    else:
        qbar =  0.5*( rho*(T-TIME_G)**2 - (rho-lamb)*T**2)/T + q0
    return (qbar + 1)/(rho*TIME_G)

def WAWT(t,lamb,ro1,ro2,initial,T):
   return (lamb[0]*tau(initial[0],ro1,lamb[0],T,t) + \
           lamb[1]*tau(initial[1],ro2,lamb[1],T,T-t))/(lamb[0]+lamb[1])


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


def generateContour(q0,T):
    c = 10.0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    lam = np.random.uniform(low=0.0,high=0.1,size=2)
    rho = np.random.uniform(low=0.01,high=0.1,size=2)
    green = []
    q0 = np.vstack((np.arange(0,20),np.arange(0,20)))
    for i in q0[0]:
        gr = []
        for j in q0[1]:
            tg = minimize_scalar(WAWT, args=(lam,rho[0],rho[1],[i,j],T), \
                bounds=(0+c, T-c), method='bounded').x
            gr.append(tg)
        green.append(gr)
    green = np.squeeze(np.asarray(green)).reshape((len(q0[0]),len(q0[1])))
    print(lam.shape, rho.shape, green.shape)
    x,y=np.meshgrid(q0[0], q0[1])
    ax.plot_surface(x, y, green, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contour(x, y, green, zdir='z', offset=-20, cmap=cm.RdYlGn)
    cset = ax.contour(x, y, green, zdir='x', offset=-10, cmap=cm.coolwarm)
    cset = ax.contour(x, y, green, zdir='y', offset=30, cmap=cm.coolwarm)

    ax.set_xlabel("Queue 1")
    ax.set_xlim(-10,30)
    ax.set_ylabel("Queue 2")
    ax.set_ylim(-10,30)
    ax.set_zlabel(r"$t_G$ for $\lambda={:.3f}$, $\rho={:.3f}$"\
            .format(lam[0],rho[0]))
    ax.set_zlim(-20,70)
    plt.show()

# get_ipython().magic('matplotlib inline')
def main(q0, t0, T, dt, i):
    """
    inputs: q size (2d array),
            initial t (float),
            total time T (float),
            time step dt (float),
            i iterator index
    """
    # define indicator functions
    def I1(t): return 1.0 if np.mod(t,T) <= TIME_G_1 else 0.0
    def I2(t): return 1.0 if np.mod(t,T) >= TIME_G_1 else 0.0

    # define our right hand side
    def f(lam,rho,ind): return lam - rho*ind

    def RHS(t,lam1,lam2,ro1,ro2):
        return [f(lam1,ro1,I1(t)), f(lam2,ro2,I2(t))]

    def giveRho(q):
        # determine rho
        rho = np.random.uniform(low=0.01, high=0.1)
        buses = np.random.binomial(q, 0.05)
        rcars = np.random.binomial(q-buses, 0.3)
        rbus = np.random.binomial(buses, 0.3)
        return np.max(((rho-rcars*0.01-rbus*0.02-(buses-rbus)*0.01), 0.01))

    def Solution(time,initial):
        # calculate solution
        queue = []
        lamb = np.random.uniform(low=0.0, high=0.07, size=2)
        rho1 = giveRho(initial[0])
        rho2 = giveRho(initial[1])
        # iterate through time step
        for k,t in enumerate(time):
            # add right turning on red light???
            queue.append(RHS(t,lamb[0],lamb[1],rho1,rho2))
            #print(tau(initial[0],rho1,lamb[0],T,t))
        fcn = np.reshape(queue,[len(time),2])
        sol = np.asarray(np.cumsum(fcn,axis=0)*dt + initial).clip(min=0)
        return sol, lamb, rho1, rho2


    # First part of the solution: time interval (t0,TIME_G_1)
    initial = q0
    c = 10.
    GreenLightTime = []
    lamb = np.random.uniform(low=0.0, high=0.07, size=2)
    rho1 = giveRho(initial[0])
    rho2 = giveRho(initial[1])
    TIME_G_1 = minimize_scalar(WAWT, args=(lamb,rho1,rho2,initial,T), \
            bounds=(0+c, T-c), method='bounded').x
    GreenLightTime.append(TIME_G_1)
    time = np.asarray(np.arange(t0,TIME_G_1,dt))
    sol,lamb,rho1,rho2 = Solution(time,q0)
    # Lets try two cycles:
    numcycles = 3
    initial = sol[-1,]
    for j in range(2*numcycles-1):
        index = j+1
        start = np.mod(index,2)*TIME_G_1 + np.floor(index/2)*T
        end = np.mod(index+1,2)*TIME_G_1 + np.floor((index+1)/2)*T
        time_temp =  np.asarray(np.arange(start,end,dt))
        sol_temp,lamb,rho1,rho2 = Solution(time_temp,initial)
        time = np.concatenate((time,time_temp))
        sol = np.concatenate((sol,sol_temp))
        initial = sol_temp[-1,]
        if np.mod(j,2)==0:
            GreenLightTime.append(T-TIME_G_1)
        else:
            GreenLightTime.append(TIME_G_1)
        TIME_G_1 =  minimize_scalar(WAWT, args=(lamb,rho1,rho2,initial,T), \
                bounds=(0+c, T-c), \
                method='bounded').x if np.mod(j,2)==0 else TIME_G_1
    queues = np.squeeze(sol)
    GreenLightTime=np.squeeze(GreenLightTime)

    # generate pretty plots
    generatePlots(time,queues,GreenLightTime)


q0 = [[7,2], [5,5], [2,1], [1,3], [6,0], [10,3]]
for i, qq in enumerate(q0):
    main(qq,0,T,dt,i)

#generateContour([1,2],T)


