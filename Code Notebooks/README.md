# Maximizing Cost Savings of Residential Solar-Plus-Storage Systems Through Predictive Optimization

Dan Drusch  
Department of Data Science, University of Wisconsin - La Crosse  
DS785: Capstone  
April 28, 2024

## Requirements
Required python libraries can be installed using `pip install -r requirements.txt`

You'll also need to install `glpk` for the Pyomo solver (https://www.gnu.org/software/glpk/)

## Structure
### Project_data
The `Project_data` directory contains the raw data gathered from the various sources: 
 - `Weather Data`: Historical weather from NOAA's [Hourly/Sub-Hourly Observational Data](https://www.ncei.noaa.gov/maps/hourly/)
 - `Use Load`: Home electrical use-load data from NREL/DOE's [End-Use Load Profiles for the U.S. Building Stock.](https://dx.doi.org/10.25984/1876417)
 - `Solar Data`: Input solar radiation/weather files from NREL's [National Solar Radiation Database](https://nsrdb.nrel.gov/data-sets/us-data) and configuration+output from NREL's [System Advisor Model](https://https://sam.nrel.gov)
 - `Energy_Rates`: Rate schedule information collected from [PG&E](https://www.pge.com) and [WE Energies](https://www.we-energies.com)

 It also contains output of various notebooks in this repo to use later

### Code Notebooks
The `Code Notebooks` directory contains the actual code used for this project. The notebooks are exectuted in several steps.

#### Data Cleanup
- `clear_sky_averaging.ipynb`: Averages multiple years' worth of clear-sky solar radiation to create a "typical" profile to input into SAM
- `noaa_weather_parse.ipynb`: Parses the NOAA weather files according to the ISD standard
- `load_cleanup.ipynb`: Resamples the use-load data to match time indexes used across the project

#### Data Preparation/Forecaster Evaluation and Tuning
- `generation_forecaster.ipynb`: Tests different methods and tunes hyperparameters for Solar Generation. Also creates additional predictor variables and output to `Project_data/Predictor Data`
- `load_forecasteripynb`: Tests different methods and tunes hyperparamters for Use Load. Also creates additional predictor variables and output to `Project_data/Predictor Data`
- `rate_config_creation.ipynb`: Builds rate schedule configuration files based on the data gathered from the utilities

#### Simulations and Comparisons
- `optimizer.py`: Contains the optimizer function, predictor functions based on the best methods+hyperparameters found for the forecasters, and convenience functions for running simulations
- `method_comparisons.ipynb`: Runs different configurations/rate plans through the forecaster/optimizer and other comparison methods.

#### Other
- `visualizations.ipynb`: Playground for developing visualizations