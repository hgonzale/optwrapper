from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# setup( cmdclass = {'build_ext': build_ext},
#        ext_modules = [ Extension( "optw_snopt", [ "optw_snopt.pyx" ],
#                                   extra_objects = ["dummy.o"],
#                                   libraries = [ "snopt_c",
#                                                 "snprint_c",
#                                                 "snopt",
#                                                 "snprint",
#                                                 "blas",
#                                                 "f2c",
#                                                 # "gfortran",
#                                                 "m"] ) ] )

setup( cmdclass = {'build_ext': build_ext},
       ext_modules = [ Extension( "utils", [ "utils.pyx" ],
                                  include_dirs = [ np.get_include() ] ) ] )

setup( cmdclass = {'build_ext': build_ext},
       ext_modules = [ Extension( "base", [ "base.pyx" ],
                                  include_dirs = [ np.get_include() ] ) ] )

setup( cmdclass = {'build_ext': build_ext},
       ext_modules = [ Extension( "npsol", [ "npsol.pyx" ],
                                  include_dirs = [ np.get_include() ],
                                  extra_objects = [ "dummy.o",
                                                    "filehandler.o" ],
                                  libraries = [ "npsol_c",
                                                "lssol_c",
                                                "npsol",
                                                "lssol",
                                                "blas",
                                                "f2c",
                                                "m" ] ) ] )

setup( cmdclass = {'build_ext': build_ext},
       ext_modules = [ Extension( "snopt", [ "snopt.pyx" ],
                                  include_dirs = [ np.get_include() ],
                                  extra_objects = [ "dummy.o",
                                                    "filehandler.o" ],
                                  libraries = [ "snopt_c",
                                                "snprint_c",
                                                "snopt",
                                                "snprint",
                                                "blas",
                                                "f2c",
                                                "m" ] ) ] )
