def CheckProg( context, cmd ):
    context.Message( "Checking for {0} command... ".format( cmd ) )
    result = WhereIs( cmd )
    context.Result( result is not None )
    return result

env = Environment()
conf = Configure( env,
                  custom_tests = {'CheckProg' : CheckProg} )


if( not env.GetOption('clean') ):
    if( not conf.CheckLib( "blas" ) ):
        print( "Could not find library 'blas'." )
        Exit(1)

    conf.CheckProg( "cython" )

env = conf.Finish()
print( "done" )
