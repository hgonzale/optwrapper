from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

major = "0"
minor = "3"

extensions = [ Extension( "optwrapper.utils",
                          sources = [ "optwrapper/utils.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.base",
                          sources = [ "optwrapper/base.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.npsol",
                          sources = [ "optwrapper/npsol.pyx",
                                      "optwrapper/filehandler.c" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "npsol",
                                        "lssol",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ),
               Extension( "optwrapper.snopt",
                          sources = [ "optwrapper/snopt.pyx",
                                      "optwrapper/filehandler.c" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "snopt",
                                        "snprint",
                                        "blas",
                                        "f2c", ## Used to convert filehandler.f
                                        "m" ] ) ]

setup( name = "OptWrapper",
       version = "{0}.{1}".format( major, minor ),
       description = "Common interface for numerical optimization solvers",
       license = "BSD 2-Clause",
       maintainer = "Humberto Gonzalez",
       maintainer_email = "hgonzale@wustl.edu",
       download_url = "https://github.com/hgonzale/optwrapper",
       packages = [ "optwrapper" ],
       ext_modules = cythonize( extensions ) )
