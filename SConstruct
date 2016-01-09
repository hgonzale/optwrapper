from subprocess import call
import os

clibs = ( "lssol", "npsol", "snopt", "ipopt" )

def CheckProg( context, cmd ):
    context.Message( "Checking for {0} command... ".format( cmd ) )
    result = WhereIs( cmd )
    result = result is not None
    context.Result( result )
    return result

def CheckPythonLib( context, lib ):
    context.Message( "Checking for Python {0} library... ".format( lib ) )
    fp = open( os.devnull, "w" )
    result = call( ( "python", "-c", "import {0}".format( lib ) ),
                   stdout=fp, stderr=fp )
    fp.close()
    result = ( result == 0 )
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
    context.Result( ret[0] )
    return int( ret[1] )


### Main
env = Environment( ENV = os.environ,
                   tools = ( "default", "textfile" ) )
conf = Configure( env,
                  custom_tests = { "CheckProg": CheckProg,
                                   "CheckPythonLib": CheckPythonLib,
                                   "CheckSizeOf": CheckSizeOf } )

### Configure
repl = {}
if( not env.GetOption( "clean" ) or
    not env.GetOption( "help" ) ):
    if( not conf.CheckLib( "blas" ) or
        not conf.CheckLib( "m" ) or
        not conf.CheckPythonLib( "numpy" ) or
        not conf.CheckProg( "cython" ) ):
        Exit(1)

    ## Check sizes of integer and doublereal from typedefs.pxd
    if( conf.CheckSizeOf( "long int" ) != 8 or
        conf.CheckSizeOf( "double" ) != 8 ):
        Exit(1)

    ## List of shared libraries to check, these define the string substitutions in setup.py.in
    for lib in clibs:
        repl[ "@{0}@".format( lib ) ] = conf.CheckLib( lib )

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
spy_install_str = spy_str + " install --record={0}".format( GetOption( "manifest_file" ) )
if( GetOption( "install_local" ) ):
    spy_install_str += " --user"

### Create targets
spy = env.Substfile( "setup.py.in", SUBST_DICT=repl )
spy_build = env.Command( "build", None, spy_build_str ) ## target "build" is *not* a file
spy_inst = env.Command( "install", None, spy_install_str ) ## target "install" is *not* a file

### Determine cleans
env.Clean( spy_build, "./build" )

if( FindFile( GetOption( "manifest_file" ), "." ) ):
    with open( GetOption( "manifest_file" ) ) as mfile:
        for line in mfile:
            env.Clean( spy_inst, line.rstrip( "\n" ) )
    env.Clean( spy_inst, GetOption( "manifest_file" ) )

### Hierarchy
env.AlwaysBuild( spy_build ) ## run setup.py in case the source files have changed
env.Depends( spy_build, spy )
env.Depends( spy_inst, spy_build )
env.Default( spy_build )
