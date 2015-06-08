from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

major = "0"
minor = "3+"

extensions = [ Extension( "optwrapper.utils",
                          sources = [ "optwrapper/utils.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.base",
                          sources = [ "optwrapper/base.pyx" ],
                          include_dirs = [ np.get_include(), "." ] ),
               Extension( "optwrapper.lssol",
                          sources = [ "optwrapper/lssol.pyx" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "lssol",
                                        "blas",
                                        "m" ] ),
               Extension( "optwrapper.npsol",
                          sources = [ "optwrapper/npsol.pyx" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "npsol",
                                        "lssol",
                                        "blas",
                                        "m" ] ),
               Extension( "optwrapper.snopt",
                          sources = [ "optwrapper/snopt.pyx" ],
                          include_dirs = [ np.get_include(), "." ],
                          libraries = [ "snopt",
                                        "snprint",
                                        "blas",
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
