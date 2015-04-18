import numpy as np
import math
import matplotlib.pyplot as plt

def haar_coeff(t, d, N):
    """
    this function calculates the Haar coefficients when the input function is not uniformly sampled

    inputs:
    t: an array of the times that the function is sampled at
    d: an array of the function values at the sampled times in t

    """
    t = np.array(t)
    d = np.array(d)

    if( t.size - 1 != d.size ):
        raise ValueError("error")

    if( t[0] != 0.0 ):
        raise ValueError("error")

    tf = 1.0
    t0 = 0.0

    deltaT = ( tf - t0 ) / (2**N)
    t_new = np.zeros( (2**N + t.size - 1,) )
    d_new = np.zeros( (t_new.size - 1,) )

    oidx = 0
    hidx = 1

    for k in range( 0, t_new.size ):
        # print( "t: {0}, th: {1}".format( t[oidx], deltaT*hidx ) )
        if( oidx < d.size and t[oidx] < deltaT*hidx ):
            t_new[k] = t[oidx]
            d_new[k] = d[oidx]
            oidx += 1
        else:
            t_new[k] = deltaT*hidx
            if( k < d_new.size ):
                d_new[k] = d[oidx-1]
            hidx += 1

    # print( "t_new: {0}".format( t_new ) )
    # print( "d_new: {0}".format( d_new ) )
    # print( "oidx: {0}".format( oidx ) )

    coeff = np.zeros( ( 2**N,))

    for k in range( d.size ):
        coeff[0] += ( t[k+1] - t[k] ) * d[k]

    cidx = 1
    for k in range( N ):
        for j in range( 2**k ):
            low = float(j) / 2**k
            mid = (.5 + j) / 2**k
            high = (1. + j) / 2**k

            idx = 0
            while( t_new[idx] < low ):
                idx += 1

            while( t_new[idx] < mid ):
                coeff[cidx] += ( t_new[idx+1] - t_new[idx] ) * d_new[idx]
                idx += 1

            while( t_new[idx] < high ):
                coeff[cidx] -= ( t_new[idx+1] - t_new[idx] ) * d_new[idx]
                idx += 1

            cidx += 1

    return coeff

def inv_haar(coeff):
    """
    this function calculates the inverse haar transform, given the coefficients of the haar transform

    input:
    coeff: the coefficients of the haar transform; vector of size 2**N, where N is the number of Haar samples

    """

    coeff = np.array(coeff)

    N = coeff.size
    n = int( math.log(N, 2) )

    p = np.array( [0.0, 0.0] )
    q = np.array( [0.0, 1.0] )

    for i in range(1, n):
        p = np.append(p, i*np.ones(2**i))
        t = np.arange(1, (2**i) + 1)
        q = np.append(q, t)

    H = np.zeros( (N, N) )
    H[0,:] = np.ones(N)

    for i in range(1,N):
        P = p[i]
        Q = q[i]
        for j in range( int((N*(Q-1)/(2**P))) , int((N*((Q-0.5)/(2**P)))) ):
            H[i,j] = 2**(P)
        for j in range( int((N*((Q-0.5)/(2**P)))), int((N*(Q/(2**P)))) ):
            H[i,j] = -2**(P)

    H = np.transpose(H)

    print( H )

    return np.dot(H, coeff)


##main
t = [ 0, 0.7/4, 1.4/2, 3.2/4, 1]
d = [ 1, 2, 4, -7 ]
N = 5

(coeff) = haar_coeff( t, d, N)

print( "coeff: {0}".format( coeff ) )

#dinv = wave.idwt( X = coeff, wf = 'h', k=2, centered=True )
#print( np.cumsum(dinv) )

dinv = inv_haar(coeff)
print( "dinv: {0}".format( dinv ) )

tinv = np.linspace( 0.0, 1.0, 2**N + 1)

plt.plot( t, np.hstack( (d,d[-1])), 'ro', tinv[:-1], dinv, 'bs')
plt.show()


# t1 = [0, 1.0/6, 2.0/6, .5, 4.0/6, 5.0/6, 1]
# d1 = [1, 2, 3, 4, 3, 2, 1]

# (coeff1) = haar_coeff(t1, d1)

# d1_ = wave.idwt(X = coeff1, wf = 'h', k=2)

# print( "d1_: " +  str(d1_) )
