#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#define Real double //TODO:replace this by include defs.in file in athena

/*----------------------------------------------------------------------------*/
/* PUBLIC FUCNTIONS                                                           */
/*----------------------------------------------------------------------------*/
void get_abundances(const Real nH, const Real T, const Real dvdr, const Real Z,
                    const Real xi_CR, const Real G_PE, const Real G_CI,
                    const Real G_CO, const Real G_H2,
                    Real *px_e, Real *px_HI, Real *px_H2, Real *px_Cplus,
                    Real *px_CI, Real *px_CO, Real *px_OI);
void get_abundances_fast(const Real nH, const Real T, const Real dvdr, const Real Z,
			 const Real xi_CR, const Real G_PE, const Real G_CI,
			 const Real G_CO, const Real G_H2,
			 Real *px_e, Real *px_HI, Real *px_H2, Real *px_Cplus,
			 Real *px_CI, Real *px_CO, Real *px_OI);
Real get_heating(const Real x_e, const Real x_HI, const Real x_H2,
		 const Real nH, const Real T, const Real Z,
		 const Real xi_CR, const Real G_PE, const Real G_H2);
Real get_cooling(const Real x_e, const Real x_HI, const Real x_H2,
		 const Real x_Cplus, const Real x_CI,
		 const Real x_CO, const Real x_OI, 
		 const Real nH, const Real T, const Real dvdr,
		 const Real Z, const Real G_PE);
