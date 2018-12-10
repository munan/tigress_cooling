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

  double get_A_(LineCool5LvElem element, LineCoolTransition transition) {
    lazyLineCool();
    return LineCool_instance->get_A(element, transition);
  }

  double get_energy_diff_(LineCool5LvElem element,
                          LineCoolTransition transition) {
    lazyLineCool();
    return LineCool_instance->get_energy_diff(element, transition);
  }

  double get_statistical_weight_(LineCool5LvElem element,
                                 uint8_t level) {
    lazyLineCool();
    return LineCool_instance->get_statistical_weight(element, level);
  }

  double get_linecooling_5lv_(LineCool5LvElem element, double temperature,
                              double electron_density, double abundance) {
    lazyLineCool();
    return LineCool_instance->get_linecooling_5lv(element, temperature,
                                                  electron_density, abundance);
  }

#ifdef __cplusplus
}
#endif
