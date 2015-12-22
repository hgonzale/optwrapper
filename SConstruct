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
    fp.close()
    result = ( result == 0 )
    context.Result( result )
    return result

def uninstall( target, manifest, env ):
    print( "hola" )



### Main
uninst = Builder( action = uninstall, suffix = None, src_suffix = ".txt" )
env = Environment( ENV = os.environ,
                   tools = [ "default", "textfile" ],
                   BUILDERS = { "Uninstall": uninstall } )
conf = Configure( env,
                  custom_tests = { "CheckProg": CheckProg,
                                   "CheckPythonLib": CheckPythonLib } )

### Configure
repl = {}
if( not env.GetOption( "clean" ) ):
    if( not conf.CheckLib( "blas" ) or
        not conf.CheckLib( "m" ) or
        not conf.CheckPythonLib( "numpy" ) or
        not conf.CheckProg( "cython" ) ):
        Exit(1)

    libs = ( "lssol", "npsol", "snopt", "ipopt" )
    for lib in libs:
        repl[ "@{0}@".format( lib ) ] = conf.CheckLib( lib )

env = conf.Finish()

### Process arguments
AddOption( "--manifest",
           dest = "manifest_file",
           type = "string",
           nargs = 1,
           action = "store",
           default = "install_manifest.txt",
           help = "Install manifest file" )

args = []
if( env.GetOption( "no_exec" ) ):
    args.append( "--dry-run" )

if( env.GetOption( "silent" ) ):
    args.append( "--quiet" )

### Create targets
str = "python setup.py {0} ".format( " ".join( args ) )
spy = env.Substfile( "setup.py.in", SUBST_DICT=repl )
spy_build = env.Command( "spy_build", None,
                         str + "build" )
spy_inst = env.Command( "spy_install", None,
                        str + "install --record={1}".format( " ".join( args ),
                                                             GetOption( "manifest_file" ) ) )
spy_inst_loc = env.Command( "spy_install", None,
                            str +
                            "install --record={1} --user".format( " ".join( args ),
                                                                  GetOption( "manifest_file" ) ) )

env.Depends( spy_build, spy )
env.Depends( spy_inst, spy_build )
env.Depends( spy_inst_loc, spy_build )
env.Uninstall( "uninst", GetOption( "manifest_file" ) )
env.Default( spy_build )

env.Clean( spy_build, "./build" )

env.Alias( "install", spy_inst )
env.Alias( "install_local", spy_inst_loc )
env.Alias( "uninstall", uninst )
