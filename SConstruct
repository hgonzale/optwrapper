from subprocess import call
import os

def CheckProg( context, cmd ):
    context.Message( "Checking for {0} command... ".format( cmd ) )
    result = WhereIs( cmd )
    result = result is not None
    context.Result( result )
    return result

def CheckPythonLib( context, lib ):
    context.Message( "Checking for Python {0} library... ".format( lib ) )
    fp = open( os.devnull, "w" )
    result = call( [ "python", "-c", "import {0}".format( lib ) ], stdout=fp, stderr=fp )
    result = ( result == 0 )
    context.Result( result )
    return result


env = Environment()
conf = Configure( env,
                  custom_tests = { "CheckProg": CheckProg,
                                   "CheckPythonLib": CheckPythonLib } )

if( not env.GetOption( "clean" ) ):
    if( not conf.CheckLib( "blas" ) or
        not conf.CheckLib( "m" ) or
        not conf.CheckPythonLib( "numpy" ) or
        not conf.CheckProg( "cython" ) ):
        Exit(1)

    repl = {}
    libs = ( "lssol", "npsol", "snopt", "ipopt" )
    for lib in libs:
        repl[ lib ] = conf.CheckLib( lib )

env.SubstInFile( "setup.py", "setup.py.in", SUBST_DICT=repl )

env = conf.Finish()
