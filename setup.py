from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

major = 0
minor = 2

extensions = [ Extension( "optwrapper.utils",
                          sources = [ "src/utils.pyx" ],
                          include_dirs = [ np.get_include() ] ),
               Extension( "optwrapper.base",
                          sources = [ "src/base.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.npsol",
                          sources = [ "src/npsol.pyx", "src/dummy.c", "src/filehandler.c" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "npsol",
                                        "lssol",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ),
               Extension( "optwrapper.snopt",
                          sources = [ "src/snopt.pyx", "src/dummy.c", "src/filehandler.c" ],
                          include_dirs = [ np.get_include(), "." ],
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
       package_dir = { "optwrapper": "src" },
       packages = [ "optwrapper" ],
       ext_modules = cythonize( extensions ) )
