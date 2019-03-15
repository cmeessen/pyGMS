# pyGMS

**G**eo**M**odelling**S**ystem is a modeling toolkit created at the GFZ Potsdam.
pyGMS is a Python 3 module that provides some basic functions to plot and handle
ASCII GMS files. It was originally written for the purpose of plotting yield
strenght envelope cross sections. Therefore, if executed in the terminal, it
expects a GMS file and a file specifying data on the yield strength envelope.

```bash
$ python pyGMS.py 
Error: not enough arguments!
Usage: GMS.py GMS_FEM FILE1 [FILE2] [...]

Plot one or multiple YSE-profiles from YSE_profile

       Parameters
       ----------
       GMS_FEM    The GMS *.fem file in ASCII format
       FILE1 ..   Output of YSE_profile with this internal structure

                  Column Property                Unit
                  ------ ----------------------  ----
                  0      Distance along profile  m
                  1      Depth                   m
                  2      DSIGMA compression      MPa
                  3      X coordinate            m
                  4      Y coordinate            m

```
