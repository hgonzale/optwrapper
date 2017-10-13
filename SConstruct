from subprocess import call
from os import environ, devnull

def CheckProg( context, cmd ):
    context.Message( "Checking for {} command... ".format( cmd ) )
    result = ( WhereIs( cmd ) is not None )
    context.Result( result )
    return result

def CheckPythonLib( context, lib, python_exe ):
    context.Message( "Checking for Python {} library... ".format( lib ) )
    fp = open( devnull, "w" )
    result = ( call( ( python_exe, "-c", "import {}".format( lib ) ), stdout=fp, stderr=fp ) == 0 )
    fp.close()
    context.Result( result )
    return result

def CheckSizeOf( context, dtype ):
    context.Message( "Getting size of " + dtype + "... " )
    program = """
      #include <stdlib.h>
      #include <stdio.h>
      int main() {
        printf( "%d", (int) sizeof( """ + dtype + """ ) );
        return 0;
      }
      """
    ret = context.TryRun( program, ".c" )
    context.Result( ret[1] )
    return int( ret[1] )

### Main
python_exe = "python3"
env = Environment( ENV = environ,
                   tools = ( "default", "textfile" ) )
conf = Configure( env,
                  custom_tests = { "CheckProg": CheckProg,
                                   "CheckPythonLib": CheckPythonLib,
                                   "CheckSizeOf": CheckSizeOf } )

### Configure
## add folders to include path
repl = { "@headers@": [ "/usr/local/include" ],
         "@cc@": "gcc",   ## we need an openmp-compatible compiler, using gcc by default
         "@cxx@": "g++" } ## we need an openmp-compatible compiler, using g++ by default

env[ "CC" ] = repl[ "@cc@" ]
env[ "CXX" ] = repl[ "@cxx@" ]
env.Append( CPPPATH = repl[ "@headers@" ] )

### Process arguments
AddOption( "--manifest",
           dest = "manifest_file",
           type = "string",
           nargs = 1,
           action = "store",
           default = "install_manifest.txt",
           help = "Manifest to record installed files" )

AddOption( "--local",
           dest = "install_local",
           default = False,
           action = "store_true",
           help = "Install locally in user's home directory" )

AddOption( "--fortran-64int",
           dest = "fort_64int",
           default = False,
           action = "store_true",
           help = "Set Fortran integers to int64_t, otherwise use int32_t" )

setup_args = []
if( env.GetOption( "no_exec" ) ):
    setup_args.append( "--dry-run" )

if( env.GetOption( "silent" ) ):
    setup_args.append( "--quiet" )

if( not env.GetOption( "clean" ) and
    not env.GetOption( "help" ) ):
    ## numpy is absolutely required
    if( not conf.CheckPythonLib( "numpy", python_exe ) ):
        Exit(1)

    ## Check double size
    if( conf.CheckSizeOf( "double" ) != 8 ):
        print( "unsupported double variable size" )
        Exit(1)

    ## Set Fortran integer type
    repl[ "@integer_type@" ] = ( "int64_t" if env.GetOption( "fort_64int" ) else "int32_t" )
    repl[ "@uinteger_type@" ] = ( "uint64_t" if env.GetOption( "fort_64int" ) else "uint32_t" )

    ## List of shared libraries and their headers to check
    ## these define string substitutions in setup.py.in
    check_cc = conf.CheckCC()
    check_cython = conf.CheckProg( "cython" )
    check_f2c = conf.CheckHeader( "f2c.h" )
    repl[ "@lssol@" ] = ( check_cc and
                          check_f2c and
                          check_cython and
                          conf.CheckLib( "lssol" ) and
                          conf.CheckHeader( "lssol.h" ) )
    repl[ "@npsol@" ] = ( repl[ "@lssol@" ] and
                          conf.CheckLib( "npsol" ) and
                          conf.CheckHeader( "npsol.h" ) )
    repl[ "@snopt@" ] = ( check_cc and
                          check_f2c and
                          check_cython and
                          conf.CheckLib( "snopt7" ) and
                          conf.CheckHeader( "snopt.h" ) )
    repl[ "@ipopt@" ] = ( check_cc and
                          check_cython and
                          conf.CheckLib( "ipopt" ) and
                          conf.CheckHeader( "coin/IpStdCInterface.h" ) )
    repl[ "@qpoases@" ] = ( check_cython and
                            conf.CheckCXX() and
                            conf.CheckLib( "qpoases", language="C++" ) and
                            conf.CheckHeader( "qpOASES.hpp", language="C++" ) )
    repl[ "@glpk@" ] = ( check_cc and
                         check_cython and
                         conf.CheckLib( "glpk" ) and
                         conf.CheckHeader( "glpk.h" ) )
    repl[ "@scipy_optimize@" ] = ( conf.CheckPythonLib( "scipy.optimize", python_exe ) and
                                   conf.CheckPythonLib( "functools", python_exe ) )

env = conf.Finish()

### Create targets
spy = env.Substfile( "setup.py.in", SUBST_DICT=repl )
init = env.Substfile( "optwrapper/__init__.py.in", SUBST_DICT=repl )
typedefs = env.Substfile( "optwrapper/typedefs.pxd.in", SUBST_DICT=repl )

spy_str = python_exe + " setup.py {}".format( " ".join( setup_args ) )
spy_build = env.Command( "build", None, spy_str + " build" ) ## target "build" is *not* a file

spy_install_str = spy_str + " install --record={}".format( env.GetOption( "manifest_file" ) )
if( env.GetOption( "install_local" ) ):
    spy_install_str += " --user"
spy_inst = env.Command( "install", None, spy_install_str ) ## target "install" is *not* a file

### Determine cleans
env.Clean( spy_build, "./build" )

if( env.FindFile( env.GetOption( "manifest_file" ), "." ) ):
    with open( env.GetOption( "manifest_file" ) ) as mfile:
        for line in mfile:
            env.Clean( spy_inst, line.rstrip( "\n" ) )
    env.Clean( spy_inst, env.GetOption( "manifest_file" ) )

### Hierarchy
env.AlwaysBuild( spy_build ) ## run setup.py in case the source files have changed
env.Depends( spy_build, init )
env.Depends( spy_build, typedefs )
env.Depends( spy_build, spy )
env.Depends( spy_inst, spy )
env.Default( spy_build )
