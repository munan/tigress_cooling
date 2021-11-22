#include <cstdlib>
#include <stdint.h>

#include "linecool_c_wrapper.hpp"
#include "linecool.hpp"

#ifdef __cplusplus
extern "C" {
#endif

  static LineCool *LineCool_instance = NULL;

  void lazyLineCool() {
    if (LineCool_instance == NULL) {
      LineCool_instance = new LineCool();
    }
  }

  double get_EinsteinA_(LineCool5LvElem element, LineCoolTransition transition) {
    lazyLineCool();
    return LineCool_instance->get_EinsteinA(element, transition);
  }

  double get_energy_diff_(LineCool5LvElem element,
                          LineCoolTransition transition) {
    lazyLineCool();
    return LineCool_instance->get_energy_diff(element, transition);
  }

  double get_statistical_weight_(LineCool5LvElem element,
                                 uint_fast8_t level) {
    lazyLineCool();
    return LineCool_instance->get_statistical_weight(element, level);
  }

  double get_linecool_5lv_(LineCool5LvElem element, const double temperature,
			   const double electron_density, const double abundance) {
    lazyLineCool();
    return LineCool_instance->get_linecool_5lv(element, temperature,
					       electron_density, abundance);
  }

  void get_linecool_all_(const double temperature,
			     const double electron_density,
			     double abundances[],
			     double *linecool_5lv, double *linecool_2lv) {
    lazyLineCool();
    LineCool_instance->get_linecool_all(temperature, electron_density,
					abundances, linecool_5lv, linecool_2lv);
  }

#ifdef __cplusplus
}
#endif
