
# coding: utf-8

# 

# In[1]:

# import libraries
import matplotlib
from __future__ import division, print_function
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[2]:

# define all time related constants
TIME_G_1 = 20 # seconds
TIME_R_1 = 20 # seconds
TIME_G_2 = 20 # seconds
TIME_R_2 = 20 # seconds
T = TIME_G_1 + TIME_R_1 # seconds
dt = 1

# define arrival constraints
LAMBDA_1 = 20 
LAMBDA_2 = 25

# define departure rate
RHO_1 = 19
RHO_2 = 26

# define initial conditions
q0, t0 = [0, 0], 0


# In[3]:

# arrivals dependent on poisson process
ARRIVAL_1 = np.random.poisson(lam=LAMBDA_1)
ARRIVAL_2 = np.random.poisson(lam=LAMBDA_2)


# In[4]:

# define indicator functions
def I1(t): return 1.0 if np.mod(t,T) <= TIME_G_1 else 0.0
def I2(t): return 1.0 if np.mod(t,T) >= TIME_G_1 else 0.0


# In[5]:

# define our right hand side
def f(t,lam,rho,ind):
    return lam - rho*ind

def RHS(t): return [f(t,LAMBDA_1,RHO_1,I1(t)), f(t,LAMBDA_2,RHO_2,I2(t))]


# In[6]:

queue_t = []
for t in np.arange(0,T,dt):
    queue_t.append(RHS(t))
fcn = np.reshape(queue_t,[T*dt,2])
fcn


# In[14]:

sol=np.floor(np.cumsum(fcn,axis=0)*dt)
sol


# In[9]:

# set up scipy ode integrator
q=spi.ode(RHS).set_integrator("vode", method="bdf")
q.set_initial_value(q0, t0)


# In[10]:

# iterate to get results
time = []
queues = []
while q.successful() and q.t < T:
    step = q.t+dt
    resp = q.integrate(q.t+dt)
    time.append(step)
    queues.append(resp)
    print(step, resp)


# In[17]:

# array manipulation
time = np.asarray(time)
queues = np.squeeze(queues)


# In[32]:

# generate pretty plots
fig = plt.plot(time,queues[:,0], 'b-', )
plt.plot(time, queues[:,1], 'r--')
ax = plt.gca()
ax.set_title("Queue Size vs. Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Queue Size (# Cars)")
ax.grid()
plt.show()


# In[ ]:



