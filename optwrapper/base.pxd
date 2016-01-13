cimport numpy as np
from .utils cimport Options

cdef class Soln:
    cdef public float value
    cdef public np.ndarray final
    cdef public int retval


cdef class Solver:
    cdef public object prob

    cdef public Options options
    cdef public int debug
