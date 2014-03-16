/* Josh Griffin ... modeled after npsol.h written by:          */
/* Mike Gertz - 2-Aug-98                                       */
/* Function prototypes for functions in the snopt distribution */

#ifndef SNOPT
#define SNOPT

#pragma once

#define INFBND (1.0e20)
#define STR_MINIMIZE "Minimize"
#define STR_MAXIMIZE "Maximize"
#define STR_DERIVATIVE_OPTION "Derivative option"
#define STR_MAJOR_PRINT_LEVEL "Major print level"
#define STR_ITERATIONS_LIMIT "Iterations limit"
#define STR_WARM_START "Warm start"
#define STR_MAJOR_FEASIBILITY_TOLERANCE "Major feasibility tolerance"
#define STR_MAJOR_OPTIMALITY_TOLERANCE "Major optimality tolerance"

typedef long int integer;
typedef double doublereal;
typedef long int flag;
typedef long int ftnlen;
// typedef long int ftnint;

typedef int (*U_fp)( integer *Status, integer *n, doublereal *x,
                     integer *needf, integer *nF, doublereal *f,
                     integer *needG, integer *lenG, doublereal *G,
                     char *cu, integer *lencu, integer *iu, integer *leniu,
                     doublereal *ru, integer *lenru );

/*
typedef struct
{ flag oerr;
  ftnint ounit;
  char *ofnm;
  ftnlen ofnmlen;
  char *osta;
  char *oacc;
  char *ofm;
  ftnint orl;
  char *oblnk;
} olist;
*/
// extern integer f_open(olist* a);

extern int snopta_( integer *start, integer *nf, integer *n,
                    integer *nxname, integer *nfname, doublereal *objadd, integer *objrow,
                    char *prob, U_fp usrfun, integer *iafun, integer *javar,
                    integer *lena, integer *nea, doublereal *a, integer *igfun,
                    integer *jgvar, integer *leng, integer *neg,
                    doublereal *xlow, doublereal *xupp,
                    char *xnames, doublereal *flow, doublereal *fupp, char *fnames,
                    doublereal *x, integer *xstate, doublereal *xmul, doublereal *f,
                    integer *fstate, doublereal *fmul, integer *info, integer *mincw,
                    integer *miniw, integer *minrw, integer *ns, integer *ninf,
                    doublereal *sinf, char *cu, integer *lencu, integer *iu, integer *leniu,
                    doublereal *ru, integer *lenru, char *cw, integer *lencw,
                    integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
                    ftnlen prob_len, ftnlen xnames_len, ftnlen fnames_len, ftnlen cu_len,
                    ftnlen cw_len );

extern int sninit_( integer *iprint, integer *isumm, char *cw,
                    integer *lencw, integer *iw, integer *leniw,
                    doublereal *rw, integer *lenrw, ftnlen cw_len );

extern int sngeti_( char *buffer, integer *ivalue, integer *errors,
                    char *cw, integer *lencw, integer *iw, integer *leniw, doublereal *rw,
                    integer *lenrw, ftnlen buffer_len, ftnlen cw_len );

extern int snset_( char *buffer, integer *iprint, integer *isumm,
                   integer *errors, char *cw, integer *lencw, integer *iw, integer *leniw,
                   doublereal *rw, integer *lenrw, ftnlen buffer_len, ftnlen cw_len );

extern int snseti_( char *buffer, integer *ivalue, integer *iprint,
                    integer *isumm, integer *errors, char *cw, integer *lencw,
                    integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
                    ftnlen buffer_len, ftnlen cw_len );

extern int snsetr_( char *buffer, doublereal *rvalue, integer *iprint, integer *isumm,
                    integer *errors, char *cw, integer *lencw, integer *iw,
                    integer *leniw, doublereal *rw, integer *lenrw, ftnlen buffer_len,
                    ftnlen cw_len );

extern int snspec_( integer *ispecs, integer *iexit, char *cw,
                    integer *lencw, integer *iw, integer *leniw, doublereal *rw,
                    integer *lenrw, ftnlen cw_len );

extern int snmema_( integer *iexit, integer *nf, integer *n, integer *nxname,
                    integer *nfname, integer *nea, integer *neg, integer *mincw,
                    integer *miniw, integer *minrw, char *cw, integer *lencw, integer *iw,
                    integer *leniw, doublereal *rw, integer *lenrw, ftnlen cw_len );

extern int snjac_( integer *iexit, integer *nf, integer *n, U_fp userfg, integer *iafun,
                   integer *javar, integer *lena, integer *nea, doublereal *a,
                   integer *igfun, integer *jgvar, integer *leng, integer *neg,
                   doublereal *x, doublereal *xlow, doublereal *xupp, integer *mincw,
                   integer *miniw, integer *minrw, char *cu, integer *lencu, integer *iu,
                   integer *leniu, doublereal *ru, integer *lenru, char *cw,
                   integer *lencw, integer *iw, integer *leniw, doublereal *rw,
                   integer *lenrw, ftnlen cu_len, ftnlen cw_len );

#endif
