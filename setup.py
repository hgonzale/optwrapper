from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

major = 0
minor = 2

extensions = [ Extension( "optwrapper.utils", [ "src/utils.pyx" ],
                          include_dirs = [ np.get_include() ] ),
               Extension( "optwrapper.base", [ "src/base.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.npsol", [ "src/npsol.pyx" ],
                          include_dirs = [ np.get_include(), "." ],
                          extra_objects = [ "src/dummy.o", "src/filehandler.o" ],
                          libraries = [ "npsol",
                                        "lssol",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ),
               Extension( "optwrapper.snopt", [ "src/snopt.pyx" ],
                          include_dirs = [ np.get_include(), "." ],
                          extra_objects = [ "src/dummy.o", "src/filehandler.o" ],
                          libraries = [ "snopt",
                                        "snprint",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ) ]

setup( name = "OptWrapper",
       version = "%d.%d" % ( major, minor ),
       description = "Common optimization interface and wrappers for different solvers",
       author = "Jingdao Chen, Humberto Gonzalez",
       author_email = "hgonzale@ese.wustl.edu",
       py_modules = "src/__init__",
       ext_modules = cythonize( extensions ) )
