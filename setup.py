from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup( cmdclass = {'build_ext': build_ext},
       ext_modules = [ Extension( "optw_snopt", [ "optw_snopt.pyx" ],
                                  extra_objects = ["dummy.o"],
                                  libraries = [ "snopt_c",
                                                "snprint_c",
                                                "snopt",
                                                "snprint",
                                                "blas",
                                                "f2c",
                                                # "gfortran",
                                                "m"] ) ] )

setup( cmdclass = {'build_ext': build_ext},
       ext_modules = [ Extension( "optw_npsol", ["optw_npsol.pyx"],
                                  extra_objects = ["dummy.o"],
                                  libraries = [ "npsol_c",
                                                "lssol_c",
                                                "npsol",
                                                "lssol",
                                                "blas",
                                                "f2c",
                                                # "gfortran",
                                                "m"] ) ] )
