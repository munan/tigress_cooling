/*******************************************************************************
 * This file is taken from CMacIonize
 * https://github.com/bwvdnbro/CMacIonize/blob/master/src/LineCoolingData.hpp
 * commit: c2885a0fe9176ecbb730476347d7aa9140a45ac5
 * 
 * Copyright (C) 2016 Bert Vandenbroucke (Original Author)
 * 
 ******************************************************************************/

#ifndef LINECOOL_HPP
#define LINECOOL_HPP

#include <string>
#include <vector>

#define M_PI 3.141592653589793

#ifndef LINECOOL_ENUM
#define LINECOOL_ENUM

// Names of supported five level elements.
enum LineCool5LvElem {
		      NI = 0,
		      NII,
		      OI,
		      OII,
		      OIII,
		      NeIII,
		      SII,
		      SIII,
		      CII,
		      CIII,
		      LINECOOL_5LV_NELEM
};

// Names of supported two level elements.
// Note that we start counting from the number of five level elements!
enum LineCool2LvElem {
		      NIII = LINECOOL_5LV_NELEM,
		      NeII,
		      SIV,
		      // Counter. Should always be the last element!
		      LINECOOL_NELEM
};

#define LINECOOL_2LV_NELEM (LINECOOL_NELEM - LINECOOL_5LV_NELEM)

// Convenient names for transitions between levels.
enum LineCoolTransition {
			 TRANS_0_to_1 = 0,
			 TRANS_0_to_2,
			 TRANS_0_to_3,
			 TRANS_0_to_4,
			 TRANS_1_to_2,
			 TRANS_1_to_3,
			 TRANS_1_to_4,
			 TRANS_2_to_3,
			 TRANS_2_to_4,
			 TRANS_3_to_4,
			 NTRANS
};
#endif  // LINECOOL_ENUM

/**
 * Internal representation of the line cooling data in "atom4.dat".
 *
 * The cooling by collisionally excited line radiation is based on section 3.5
 * of Osterbrock, D. E. & Ferland, G. J. 2006, Astrophysics of Gaseous Nebulae
 * and Active Galactic Nuclei, 2nd edition
 * (http://adsabs.harvard.edu/abs/2006agna.book.....O).
 *
 * We use data from a large number of different sources, many of which are part
 * of the the IRON project and were located using their online database
 * (http://cdsweb.u-strasbg.fr/tipbase/home.html). We also used the extensive
 * list of data sources in Lykins, M. L., Ferland, G. J., Kisielius, R.,
 * Chatzikos, M., Porter, R. L., van Hoof, P. A. M., Williams, R. J. R., Keenan,
 * F. P. & Stancil, P. C. 2015, ApJ, 807, 118
 * (http://adsabs.harvard.edu/abs/2015ApJ...807..118L) to locate data for ions
 * that seem to be missing from the IRON database.
 *
 * The actual data used comes from (in alphabetical order):
 *  - Berrington, K. A. 1988, JPhB, 21, 1083
 *    (http://adsabs.harvard.edu/abs/1988JPhB...21.1083B) (OI)
 *  - Berrington, K. A., Burke, P. G., Dufton, P. L. & Kingston, A. E. 1985,
 *    ADNDT, 33, 195 (http://adsabs.harvard.edu/abs/1985ADNDT..33..195B) (CIII)
 *  - Blum, R. D. & Pradhan, A. K. 1992, ApJS, 80, 425
 *    (http://adsabs.harvard.edu/abs/1992ApJS...80..425B) (NIII)
 *  - Butler, K. & Zeippen, C. J. 1994, A&AS, 108, 1
 *    (http://adsabs.harvard.edu/abs/1994A%26AS..108....1B) (NeIII)
 *  - Froese Fischer, C. & Tachiev, G. 2004, ADNDT, 87, 1
 *    (http://adsabs.harvard.edu/abs/2004ADNDT..87....1F) (NI, OII, CII, CIII)
 *  - Galavis, M. E., Mendoza, C. & Zeippen, C. J. 1997, A&AS, 123, 159
 *    (http://adsabs.harvard.edu/abs/1997A%26AS..123..159G) (NII, OI, OIII,
 *    NeIII)
 *  - Galavis, M. E., Mendoza, C. & Zeippen, C. J. 1998, A&AS, 131, 499
 *    (http://adsabs.harvard.edu/abs/1998A%26AS..131..499G) (NIII)
 *  - Griffin, D. C, Mitnik, D. M., Badnell, N. R. 2001, JPhB, 34, 4401
 *    (http://adsabs.harvard.edu/abs/2001JPhB...34.4401G) (NeII)
 *  - Hudson, C. E., Ramsbottom, C. A. & Scott, M. P. 2012, ApJ, 750, 65
 *    (http://adsabs.harvard.edu/abs/2012ApJ...750...65H) (SIII)
 *  - Kaufman, V. & Sugar, J. 1986, JPCRD, 15, 321
 *    (http://adsabs.harvard.edu/abs/1986JPCRD..15..321K) (NeII)
 *  - Kisielius, R., Storey, P. J., Ferland, G. J. & Keenan, F. P. 2009, MNRAS,
 *    397, 903 (http://adsabs.harvard.edu/abs/2009MNRAS.397..903K) (OII)
 *  - Lennon, D. J. & Burke, V. M. 1994, A&AS, 103, 273
 *    (http://adsabs.harvard.edu/abs/1994A%26AS..103..273L) (NII, OIII)
 *  - Martin, W. C., Zalubas, R. & Musgrove, A. 1990, JPCRD, 19, 821
 *    (http://adsabs.harvard.edu/abs/1990JPCRD..19..821M) (SIV)
 *  - Mendoza, C. & Zeippen, C. J. 1982, MNRAS, 199, 1025
 *    (http://adsabs.harvard.edu/abs/1982MNRAS.199.1025M) (SIII)
 *  - Pradhan, A. 1995, The Analysis of Emission Lines: A Meeting in Honor of
 *    the 70th Birthdays of D. E. Osterbrock & M. J. Seaton, proceedings of the
 *    Space Telescope Science Institute Symposium, held in Baltimore, Maryland
 *    May 16--18, 1994, Eds.: Robert Williams and Mario Livio, Cambridge
 *    University Press, p. 8.
 *    (http://adsabs.harvard.edu/abs/1995aelm.conf....8P) (SIV)
 *  - Saraph, H. E. & Storey, P. J. 1999, A&AS, 134, 369
 *    (http://adsabs.harvard.edu/abs/1999A%26AS..134..369S) (SIV)
 *  - Saraph, H. E. & Tully, J. A. 1994, A&AS, 107, 29
 *    (http://adsabs.harvard.edu/abs/1994A%26AS..107...29S) (NeII)
 *  - Tayal, S. S. 2000, ADNDT, 76, 191
 *    (http://adsabs.harvard.edu/abs/2000ADNDT..76..191T) (NI)
 *  - Tayal, S. S. 2008, A&A, 486, 629
 *    (http://adsabs.harvard.edu/abs/2008A%26A...486..629T) (CII)
 *  - Tayal, S. S. & Zatsarinny, O. 2010, ApJS, 188, 32
 *    (http://adsabs.harvard.edu/abs/2010ApJS..188...32T) (SII)
 *  - Zatsarinny, O. & Tayal, S. S. 2003, ApJS, 148, 575
 *    (http://adsabs.harvard.edu/abs/2003ApJS..148..575Z) (OI)
 *
 * Adding new lines is straightforward: add an entry in the corresponding enum
 * (LineCool5LvElem or LineCool2LvElem; before
 * the counter element), and initialize the data in the constructor. To compute
 * line strengths, add relevant code to linestr(). Adding new elements will
 * break some unit tests in testLineCool, but should work fine.
 */

class LineCool {
private:
  // Collision strength fit parameters for the five level elements.
  double _5lv_coll_str[LINECOOL_5LV_NELEM][NTRANS][7];

  // Transition probabilities for deexcitation between different
  // levels for the five level elements.
  double _5lv_A[LINECOOL_5LV_NELEM][NTRANS];

  // Energy differences for the transitions between different levels
  // for the five level elements (in K).
  double _5lv_energy_diff[LINECOOL_5LV_NELEM][NTRANS];

  // Inverse statistical weights for the different levels for the five
  // level elements.
  double _5lv_ginv[LINECOOL_5LV_NELEM][5];

  // Collision strength fit parameters for the two level elements.
  double _2lv_coll_str[LINECOOL_2LV_NELEM][7];

  // Transition probabilities for deexcitation between different
  // levels for the two level elements.
  double _2lv_A[LINECOOL_2LV_NELEM];

  // Energy differences for the transitions between different levels
  // for the two level elements (in K).
  double _2lv_energy_diff[LINECOOL_2LV_NELEM];

  // Inverse statistical weights for the different levels for the two
  // level elements.
  double
  _2lv_ginv[LINECOOL_2LV_NELEM][2];

  // Prefactor for collision strengths: $\frac{h^2}{\sqrt{k}
  // \left(2\pi{}m_e\right)^\frac{3}{2}$ (in K^0.5 m^3 s^-1). */
  double _coll_str_prefactor;

  void compute_5lvpop(LineCool5LvElem element,
		      double coll_str_prefactor, double T,
		      double Tinv, double logT, double lvpop[5]) const;

  double compute_2lvpop(LineCool2LvElem element,
			double coll_str_prefactor, double T,
			double Tinv, double logT) const;

public:
  LineCool();

  double get_EinsteinA(LineCool5LvElem element,
		       LineCoolTransition transition) const;
  double get_energy_diff(LineCool5LvElem element,
			 LineCoolTransition transition) const;
  double get_statistical_weight(LineCool5LvElem element,
                                uint_fast8_t level) const;

  static int solve_system_of_linear_equations(double A[5][5], double B[5]);

  double get_linecooling_all(double temperature, double electron_density,
			     const double abundances[LINECOOL_NELEM]) const;
    
  double get_linecool_5lv(LineCool5LvElem element, const double temperature,
			  const double electron_density, const double abundance) const;

  void get_linecool_all(const double temperature, const double electron_density,
			double abundances[LINECOOL_NELEM],
			double *linecool_5lv, double *linecool_2lv);

};

#endif // LINECOOL_HPP
