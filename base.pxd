cimport numpy as np

cdef class Soln:
    cdef public float value
    cdef public np.ndarray final
    cdef public int retval


cdef class Solver:
    cdef public object prob

    cdef public dict printOpts
    cdef public dict solveOpts
