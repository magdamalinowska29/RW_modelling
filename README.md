# RW_modelling

This repository contains all the files needed to perform behavioural anallysis using Rescorla-Wagner models as featured in my Master's thesis.

### PREPROCESSING ###

 script **getting_data_pilot** from the **getting_data** folder is used to change the raw data from the format received in pavlovia to the format needed for the analysis 

 ### MODELLING ###

 folder **models** contains model functions for 4 RW models  as well as th math.py script with functions used by all four models

 scripts in **simulation_scripts** can simulate data using each of the RW models

 scripts in **fitting_scripts** allow to fit the model parameters to the simulated or experimental data 

  folder **modelling_checks** contains scripts needed to check the performance of the models (parameter recovery, model recovery)

  ### ADDITIONAL SCRIPTS ###

  scripts in **accuraccy_plotting** allows to plot the experimental/simulated data in terms of of accuracy or frequency of different choices

  **data_processing_not_modelling** contains the a script that extracts from raw data information about the amount of rewards gathered in experiment per participant
