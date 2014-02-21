#ifndef NPSOL
#define NPSOL

#pragma once

#define STR_WARM_START "Warm start"
#define STR_PRINT_FILE "Print file"
#define STR_MAJOR_PRINT_LEVEL "Major print level"
#define STR_FEASIBILITY_TOLERANCE "Feasibility tolerance"
#define STR_OPTIMALITY_TOLERANCE "Optimality tolerance"
#define STR_MINOR_ITERATIONS_LIMIT "Minor iterations limit"

typedef long int integer;
typedef double doublereal;
typedef long int ftnlen;
// typedef long int flag;
// typedef long int ftnint;
// typedef int /* Unknown procedure type */ (*U_fp)();
typedef int (*c_fp)( integer*, integer*, integer*, integer*,
                     integer*, doublereal*, doublereal*, doublereal*, integer*);
typedef int (*o_fp)( integer*, integer*, doublereal*, doublereal*, doublereal*, integer* );

/* typedef struct */
/* {	flag oerr; */
/* 	ftnint ounit; */
/* 	char *ofnm; */
/* 	ftnlen ofnmlen; */
/* 	char *osta; */
/* 	char *oacc; */
/* 	char *ofm; */
/* 	ftnint orl; */
/* 	char *oblnk; */
/* } olist; */
/* extern integer f_open(olist* a); */

extern int npsol_( integer *n, integer *nclin, integer *ncnln, integer *lda,
                   integer *ldju, integer *ldr, doublereal *a, doublereal *	bl, doublereal *bu,
                   c_fp funcon, o_fp funobj, integer *inform__,	integer *iter,
                   integer *istate, doublereal *c__, doublereal *cjacu,	doublereal *clamda, doublereal *objf,
                   doublereal *gradu, doublereal *r__, doublereal *x, integer *iw, integer *leniw,
                   doublereal *w,	integer *lenw );

extern int npopti_( char *string, integer *ivalue, ftnlen string_len );
extern int npoptr_(char *string, doublereal *rvalue, ftnlen string_len);
extern int npoptn_(char *string, ftnlen string_len);
// extern int npfile_( integer* file, integer* inform );
// extern int npfilewrapper_( char *name__, integer *inform__, ftnlen name_len );

#endif
