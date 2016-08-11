
# coding: utf-8

# import libraries
import matplotlib
import numpy as np
import scipy.integrate as spi
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import seaborn as sns

def main(q0, t0 ,i):
    # define all time related constants
    TIME_G_1 = 20 # seconds
    TIME_R_1 = 20 # seconds
#    TIME_G_2 = TIME_R_1 # seconds
#    TIME_R_2 = TIME_G_1 # seconds
    T = TIME_G_1 + TIME_R_1 # seconds
    dt = .01    #time step in cumsum method

    # define arrival constraints
    #LAMBDA_1 = lamb[0]
    #LAMBDA_2 = lamb[1]
    # define departure rate
    #RHO_1 = np.random.uniform(low=2.0, high=4.0)
    #RHO_2 = np.random.uniform(low=2.0, high=4.0)

    # define initial conditions
    # q0, t0 = [7, 20], 0

    # arrivals dependent on poisson process
 #   ARRIVAL_1 = np.random.poisson(lam=LAMBDA_1)
 #   ARRIVAL_2 = np.random.poisson(lam=LAMBDA_2)

    # define indicator functions
    def I1(t): return 1.0 if np.mod(t,T) <= TIME_G_1 else 0.0
    def I2(t): return 1.0 if np.mod(t,T) >= TIME_G_1 else 0.0

    # define our right hand side
    def f(lam,rho,ind): return lam - rho*ind

    def RHS(t,lam1,lam2,ro1,ro2):
        return [f(lam1,ro1,I1(t)), f(lam2,ro2,I2(t))]

    def giveRho(q):
        rho = np.random.uniform(low=2.0, high=5.0)
        buses = np.random.binomial(q, 0.05)
        rcars = np.random.binomial(q-buses, 0.3)
        rbus = np.random.binomial(buses, 0.3)
        return (rho1 - rcar1*0.1 - rbus1*0.2 - (buses1-rbus1)*0.1).clip(min=1.0)

    def Solution(time,initial):
        queue = []
        lamb = np.random.uniform(low=0.0, high=7.0, size=2)
        rho1 = giveRho(initial[0])
        rho2 = giveRho(initial[1])
        # iterate through time step
        for k,t in enumerate(time):
            # add right turning on red light????
            queue.append(RHS(t,lamb[0],lamb[1],rho1,rho2))
        fcn = np.reshape(queue,[len(time),2])
        sol = np.asarray(np.cumsum(fcn,axis=0)*dt + initial)
        sol[sol<0] = 0
        return sol

    # First part of the solution: time interval (t0,TIME_G_1)
    time = np.asarray(np.arange(t0,TIME_G_1,dt))
    Slope = RHS(time[1])
    sol = Solution(time,q0)
    # Lets try two cycles:
    numcycles = 3
    initial = sol[-1,]
    for j in range(2*numcycles-1):
        index = j+1
        start = np.mod(index,2)*TIME_G_1 + np.floor(index/2)*T
        end = np.mod(index+1,2)*TIME_G_1 + np.floor((index+1)/2)*T
        time_temp =  np.asarray(np.arange(start,end,dt))
        Slope.append(RHS(time_temp[1]))
        sol_temp = Solution(time_temp,initial)
        initial = sol_temp[-1,]
        time = np.concatenate((time,time_temp))
        sol = np.concatenate((sol,sol_temp))

    queues = np.squeeze(sol)
    print(Slope)
#    # set up scipy ode integrator
#    q=spi.ode(RHS).set_integrator("vode", method="bdf")
#    q.set_initial_value(q0, t0)
#
#    # iterate to get results
#    time = []
#    queues = []
#    while q.successful() and q.t < T:
#        step = q.t+dt
#        resp = q.integrate(q.t+dt)
#        resp[resp<0]=0
##        resp = max([resp,0])
#        time.append(step)
#        queues.append(resp)
#    #    print(step, resp)

    # generate pretty plots
    fig = plt.plot(time,queues[:,0], 'k-', label="Street 1", alpha=0.7)
    plt.plot(time, queues[:,1], 'k-', label="Street 2", alpha=0.7)
    ax = plt.gca()
    #fig = plt.plot(time, np.zeros(len(time)), 'w')
    #ax = plt.gca()
    ax.set_title("Queue Size vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Size (# Cars)")
    # ax.legend(loc='best')
    # plt.axvspan(0, TIME_G_1, facecolor='r', alpha=0.2)
    # plt.axvspan(TIME_G_1, T, facecolor='g', alpha=0.2)
    # plt.axvline(x=TIME_G_1, color='g')

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax.set_title("Queue Size vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Size (# Cars)")

    points1 = np.array([time, queues[:,0]]).T.reshape(-1,1,2)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
    lc1 = LineCollection(segments1, cmap=ListedColormap(['g','r']))
    lc1.set_array(time)
    lc1.set_linewidth(2)
    lc1.set_linestyle('-')
    ax.add_collection(lc1)

    points2 = np.array([time, queues[:,1]]).T.reshape(-1,1,2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
    lc2 = LineCollection(segments2, cmap=ListedColormap(['r','g']))
    lc2.set_array(time)
    lc2.set_linewidth(2)
    lc2.set_linestyle('--')
    ax.add_collection(lc2)

    plt.xlim(time.min(), time.max())
    ax.grid()
    plt.savefig(str(i)+".png")
    plt.close()

q0 = [[7,20], [5,5], [15,12], [1,3], [6,0], [14,3]]
for i, qq in enumerate(q0):
    main(qq,0,i)

