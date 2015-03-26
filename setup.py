from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

major = 0
minor = 3

extensions = [ Extension( "optwrapper.utils",
                          sources = [ "optwrapper/utils.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.base",
                          sources = [ "optwrapper/base.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.npsol",
                          sources = [ "optwrapper/npsol.pyx",
                                      # "optwrapper/dummy.c",
                                      "optwrapper/filehandler.c" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "npsol",
                                        "lssol",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ),
               Extension( "optwrapper.snopt",
                          sources = [ "optwrapper/snopt.pyx",
                                      # "optwrapper/dummy.c",
                                      "optwrapper/filehandler.c" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "snopt",
                                        "snprint",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ) ]

setup( name = "OptWrapper",
       version = "%d.%d" % ( major, minor ),
       description = "Common optimization interface and wrappers for different solvers",
       license = "BSD 2-Clause",
       author = "Jingdao Chen, Humberto Gonzalez, Christa Stathopoulos",
       author_email = "hgonzale@ese.wustl.edu",
       packages = [ "optwrapper" ],
       ext_modules = cythonize( extensions ) )
