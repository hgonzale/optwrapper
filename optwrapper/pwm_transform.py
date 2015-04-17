import numpy as np 
import matplotlib.pyplot as plt 


def pwm(t, d):
    """
    this function generates the pwm transformation of the discrete input vectors 

    arguments:
    t: the time vector of Haar sample times; size: (2**N)+1 (where N corresponds to the Haar transform)
    d: the matrix of discrete modal values at each Haar time sample; size: #modes x 2**N 

    """

    t = np.array(t)

    #Nmodes = number of modes
    #N = number of Haar samples 
    (Nmodes, N) = np.shape(d)

    t_pwm = np.array( [ ] )
    #d_pwm = np.zeros( (Nmodes, Nmodes * N ) )

    deltaT = (1.0 - 0.0) / float(N)

    for k in range(1, N+1):
        t_pwm = np.append(t_pwm, t[k-1])
        if k == 1:
            d_pwm = np.identity(Nmodes)
        else:
            d_pwm = np.hstack( (d_pwm, np.identity(Nmodes) ) )
        for j in range(Nmodes - 1 ):
            if j == 0:
                mode_sum = d[j,k-1]
            t_pwm = np.append(t_pwm, t[k-1] + deltaT*mode_sum)
            mode_sum += d[j+1,k-1]

    t_pwm = np.append(t_pwm, 1.0)

    # print t_pwm
    # print d_pwm

    return (t_pwm, d_pwm)

##main

t = [0, 1.0/4.0, 2.0/4.0, 3.0/4.0, 1.0]

#t = [0.0, 1.0/8.0, 2.0/8.0, 3.0/8.0, 4.0/8.0, 5.0/8.0, 6.0/8.0, 7.0/8.0, 1]

#d = np.array( [ [.3, .1, .2, 1.0, 0.0, .5, .7, .6], [.7, .9, .8, 0.0, 1.0, .5, .3, .4] ] )

d = np.array( [ [.2, 1.0, 0.0, 0.5], [0.7, 0.0, 0.5, 0.0], [0.1, 0.0, 0.5, 0.5] ] )

(t_pwm, d_pwm) = pwm(t,d)

print t_pwm 
print d_pwm

plt.step(t_pwm, np.hstack( (d_pwm[0,:], d[0,-1]) ), where='post' )
plt.title('d1')
plt.show()

plt.step(t_pwm, np.hstack( (d_pwm[1,:], d[1,-1]) ), where='post' )
plt.title('d2')
plt.show()

plt.step(t_pwm, np.hstack( (d_pwm[2,:], d[2,-1]) ), where='post' )
plt.title('d3')
plt.show()