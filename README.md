# Urban Heat Island intensity prediction using ensemble regression methods

This machine learning project, part of the CS-433 Machine Learning course, aims to address the challenges posed by urban heat islands (UHIs) through predictive modeling. Urban heat islands refer to areas within cities that experience significantly higher temperatures than their surrounding rural areas, primarily due to human activities and the built environment.

## Overview

In recent years, the effects of UHIs on both the environment and public health have become increasingly apparent. Rising temperatures, intensified energy consumption, and compromised air quality contribute to a multitude of issues, affecting the well-being of urban residents and exacerbating climate change. Our project leverages the power of machine learning to predict and understand the spatial distribution of urban heat islands, enabling measures for urban planning and climate resilience.

## Key objectives
1. Predict the temperature delta between urban and rural areas
2. Predict accurately the population based on the delta of temperature

## Getting started

The two objectives are separated in two notebooks (task1.ipynb and task2.ipynb), and the main functions used are in the functions.py file. In order to train the model, we gathered data from several sources : 
- Temperature, relative humidity, wind speed, rural urban mask, land sea mask come from the [Copernicus research program](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-urban-climate-cities%3Ftab=overview?tab=doc)
- NDVI and land cover type come from NASA [EARTHDATA program](https://appeears.earthdatacloud.nasa.gov/task/area)
- The population come from a dataset made by the [Joint Research Center](https://data.jrc.ec.europa.eu/dataset/be02937c-5a08-4732-a24a-03e0a48bdcda#dataaccess) of the European Union

In order to train the model, one need either to download the data for at least one European city, or, on demand, a csv file can be sent with around 2 millions samples for 5 European cities (Lyon, Madrid, Amsterdam, Vienna and Stockholm). Trained models (finalized_model.sav and gb_model.pk1) can be found in the repo. It is the model with the best performances we obtained.

### Librairies to install

This project uses several additional python librairies, with python 3.8+
- Scikit-learn, Numpy, Pandas, Matplotlib, Tqdm, rasterio, Xarray, Pickle, mpl-scatter-density

### Data files

The data files need to be separated in the following way :
- data_population_day (containing all the files with the day population)
- data_population_night
- data_cities -> x_data -> files (where x is the name of the city)
- NDVI (containing the NDVI files)
- elevation (containing elevation files)
- Land cover files with no folder
