#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "linecool_c_wrapper.hpp"

int main() {
  
  printf("%e\n",get_EinsteinA_(CII, (char)1));
  return 0;
  
}
