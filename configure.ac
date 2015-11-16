AC_PREREQ([2.69])

AC_INIT([OptWrapper],[0.5+])

AC_CONFIG_SRCDIR([README])
AC_CONFIG_AUX_DIR(config)

AC_CANONICAL_HOST

AC_PROG_CC
AX_PYTHON

# CFLAGS="-g -O2"
# CXXFLAGS="-g -O2"
# FFLAGS="-g -O2"


# configure path to cython using "./configure --with-cython=/path/to/cython"
# or else find it on the path
AC_ARG_WITH([cython],
            [AS_HELP_STRING([--with-cython],
              [path to cython binary to use (otherwise find in PATH) @<:@default=check@:>@])],
            [],
            [with_cython=yes])
AS_IF([test "x$with_cython" == xyes],[AC_PATH_PROG([CYTHON], [cython], [cython])
                                      cython_path="$ac_cv_path_CYTHON"],
      [test "x$with_cython" != xno],[cython_path="$with_cython"])
AS_IF([test "x$cython_path" == x],
      [AC_MSG_ERROR([Cannot find cython.])])
AC_SUBST([CYTHON],[$cython_path])

AC_CONFIG_FILES([setup.py])

AC_OUTPUT
