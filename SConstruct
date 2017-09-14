from subprocess import call
from os import environ, devnull

def CheckProg( context, cmd ):
    context.Message( "Checking for {0} command... ".format( cmd ) )
    result = WhereIs( cmd )
    result = result is not None
    context.Result( result )
    return result

def CheckPythonLib( context, lib ):
    context.Message( "Checking for Python {0} library... ".format( lib ) )
    fp = open( devnull, "w" )
    result = ( call( ( "python", "-c", "import {0}".format( lib ) ),
                   stdout=fp, stderr=fp ) == 0 )
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
env = Environment( ENV = environ,
                   tools = ( "default", "textfile" ) )
conf = Configure( env,
                  custom_tests = { "CheckProg": CheckProg,
                                   "CheckPythonLib": CheckPythonLib,
                                   "CheckSizeOf": CheckSizeOf } )

### Configure
## add folders to include path
repl = { "@headers@": [ "/usr/local/include" ],
         "@cc@": "gcc" } ## we need an openmp-compatible compiler, using gcc by default
env[ "CC" ] = repl[ "@cc@" ]
env.Append( CPPPATH = repl[ "@headers@" ] )

if( not env.GetOption( "clean" ) and
    not env.GetOption( "help" ) ):
    if( not conf.CheckCC() or
        not conf.CheckPythonLib( "numpy" ) or
        not conf.CheckProg( "cython" ) ):
        Exit(1)

    ## Check sizes of integer and doublereal from typedefs.pxd
    if( conf.CheckSizeOf( "long int" ) != 8 or
        conf.CheckSizeOf( "double" ) != 8 ):
        Exit(1)

    ## List of shared libraries and their headers to check
    ## these define string substitutions in setup.py.in
    repl[ "@lssol@" ] = ( conf.CheckLib( "lssol" ) and
                          conf.CheckHeader( "lssol.h" ) )
    repl[ "@npsol@" ] = ( repl[ "@lssol@" ] and
                          conf.CheckLib( "npsol" ) and
                          conf.CheckHeader( "npsol.h" ) )
    repl[ "@snopt@" ] = ( conf.CheckLib( "snopt" ) and
                          conf.CheckHeader( "snopt.h" ) )
    repl[ "@ipopt@" ] = ( conf.CheckLib( "ipopt" ) and
                          conf.CheckHeader( "coin/IpStdCInterface.h" ) )

env = conf.Finish()

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

args = []
if( env.GetOption( "no_exec" ) ):
    args.append( "--dry-run" )
if( env.GetOption( "silent" ) ):
    args.append( "--quiet" )

spy_str = "python setup.py {0}".format( " ".join( args ) )
spy_build_str = spy_str + " build"
spy_install_str = spy_str + " install --record={0}".format( env.GetOption( "manifest_file" ) )
if( env.GetOption( "install_local" ) ):
    spy_install_str += " --user"

### Create targets
spy = env.Substfile( "setup.py.in", SUBST_DICT=repl )
init = env.Substfile( "optwrapper/__init__.py.in", SUBST_DICT=repl )
spy_build = env.Command( "build", None, spy_build_str ) ## target "build" is *not* a file
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
env.Depends( spy_build, spy )
env.Depends( spy_inst, spy )
env.Default( spy_build )
