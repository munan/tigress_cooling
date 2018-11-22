Heating and cooling functions for TIGRESS simulation

Implementation is in tigress_cool.h and tigress_cool.c
Please check the TODO comments in these two files for integrating it with
Athena-TGRESS code

An example is included for solar neighborhood conditions with equilibrium
temperature at a range of densities in "main.c" file (the equilibrium temperature 
is from equilibrium chemistry calculations of Gong, Ostriker and Wolfire 2017). 
You can run the example by 

$python run.py

Then you can use the ipython notebook

tigress_cooling.ipynb

in the "script/" folder to plot the results compared to the original TIGRESS
heating and cooling rates (Koyama and Inutsuka 2002).

A detailed documentation is included in the "doc/" folder. Please pay special
attention to Section 6: notes for implementation in TIGRESS.

Enjoy!

Munan
