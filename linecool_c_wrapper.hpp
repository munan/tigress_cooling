#ifndef LineCool_C_WRAPPER_H 
#define LineCool_C_WRAPPER_H 

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LINECOOL_ENUM
#define LINECOOL_ENUM
  
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

  enum LineCool2LvElem {
    NIII = LINECOOL_5LV_NELEM,
    NeII,
    SIV,
    LINECOOL_NELEM
  };

#define LINECOOL_2LV_NELEM (LINECOOL_NELEM - LINECOOL_5LV_NELEM)

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

#endif

  double get_A_(enum LineCool5LvElem element,
                enum LineCoolTransition transition);
  double get_energy_diff_(enum LineCool5LvElem element,
                          enum LineCoolTransition transition);
  double get_statistical_weight_(enum LineCool5LvElem element,
                                 uint8_t level);
  double get_linecooling_5lv_(enum LineCool5LvElem element, double temperature,
                              double electron_density, double abundance);

#ifdef __cplusplus
}
#endif


#endif
