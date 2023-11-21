import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
from tqdm import tqdm



def resample_image(image_to_resample, new_dimensions):
    '''Given a 2D array, increase its resolution to new dimensions, with no interpolation'''
    new_image = np.zeros(new_dimensions)
    for i in range(new_dimensions[0]):
        for j in range(new_dimensions[1]):
            new_image[i,j] = image_to_resample[int(i*image_to_resample.shape[0]/new_dimensions[0]), int(j*image_to_resample.shape[1]/new_dimensions[1])]
    return new_image

def crop_and_downgrade(pop_day_tiff, pop_night_tiff, temp_city):

    min_lon = temp_city.longitude.min().values.item()
    max_lon = temp_city.longitude.max().values.item()
    min_lat = temp_city.latitude.min().values.item()
    max_lat = temp_city.latitude.max().values.item()

    # crop the tiff with the city bounds
    cropped_pop_day = pop_day_tiff.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=pop_day_tiff.transform))
    cropped_pop_night = pop_night_tiff.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=pop_night_tiff.transform))


    #convert the latitude in meters
    min_lat = 110574 * min_lat
    max_lat = 110574 * max_lat

    #convert the longitude in meters
    min_lon = 111320 * min_lon
    max_lon = 111320 * max_lon

    return cropped_pop_day,cropped_pop_night

def crop_image(image_to_crop, temp_city):
    min_lon = temp_city.longitude.min().values.item()
    max_lon = temp_city.longitude.max().values.item()
    min_lat = temp_city.latitude.min().values.item()
    max_lat = temp_city.latitude.max().values.item()

    # crop the tiff with the city bounds
    cropped_image = image_to_crop.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=image_to_crop.transform))

    return cropped_image


def compute_deltaT_urban(temperature_ds, urban_mask_ds):
    deltaT_ds = []
    for i in range(temperature_ds.time.shape[0]):
        rural_avg = temperature_ds.tas[i,:,:].where(urban_mask_ds.ruralurbanmask[:,:] == 1).mean(dim=['x','y']).values.item()
        deltaT_ds.append(temperature_ds.tas[i,:,:].values - rural_avg)
    deltaT_ds = np.array(deltaT_ds).flatten()
    return deltaT_ds

def process_data(folder_path, pop_day, pop_night, elevation, number_of_sample_per_city):
    '''Create a dataframe with, for each city, the temperature, the population, the wind speed, the humidity and compute the delta of
    temperature between urban and rural areas and add it to the dataframe'''
    city_df = pd.DataFrame(columns=['temp', 'pop', 'wind', 'hum', 'deltaT', 'hour', 'city'])
    for city in tqdm(['Basel', 'Cologne', 'Glasgow', 'Hamburg', 'Nantes','Turin']):
        temp_file_path = folder_path+'/tas_'+city+'_UrbClim_2011_07_v1.0.nc'
        wind_file_path = folder_path+'/sfcWind_'+city +'_UrbClim_2011_07_v1.0.nc'
        hum_file_path = folder_path+'/russ_'+city +'_UrbClim_2011_07_v1.0.nc'
        rural_mask_file_path = folder_path+'/ruralurbanmask_'+city +'_UrbClim_v1.0.nc'
        temp_file = xr.open_dataset(temp_file_path)
        wind_file = xr.open_dataset(wind_file_path)
        hum_file = xr.open_dataset(hum_file_path)
        rural_mask_file = xr.open_dataset(rural_mask_file_path)

        
        cropped_pop_day, cropped_pop_night = crop_and_downgrade(pop_day, pop_night, temp_file)
        elevation_city = crop_image(elevation, temp_file)

        pop_day_city = resample_image(cropped_pop_day, temp_file.tas[0,:,:].shape)
        pop_night_city = resample_image(cropped_pop_night, temp_file.tas[0,:,:].shape)

        elevation_city = resample_image(elevation_city, temp_file.tas[0,:,:].shape)
        elevation_flatten = np.tile(elevation_city.flatten(), temp_file.tas.shape[0])

        populations = np.concatenate([np.tile(pop_night_city.flatten(),8), np.tile(pop_day_city.flatten(), 12), np.tile(pop_night_city.flatten(), 4)])
        populations = np.tile(populations, 31)

        day_hours = np.tile(np.arange(0,24), temp_file.x.shape[0]*temp_file.y.shape[0])
        hours = np.tile(day_hours.reshape(temp_file.x.shape[0]*temp_file.y.shape[0], 24).flatten(order='F'), 31)

        deltaT = compute_deltaT_urban(temp_file, rural_mask_file)
        rural = np.tile(rural_mask_file.ruralurbanmask.values.flatten(), temp_file.tas.shape[0])
        print(rural.shape, deltaT.shape)
        city = np.tile(np.array([city]), number_of_sample_per_city)

        #generate random indexes to sample the data
        indexes = np.random.randint(0, temp_file.tas.shape[0]*temp_file.tas.shape[1]*temp_file.tas.shape[2], number_of_sample_per_city)

        city_df = pd.concat([city_df, pd.DataFrame({'temp': temp_file.tas.values.flatten()[indexes],
                                                    'pop':populations[indexes], 
                                                    'wind': wind_file.sfcWind.values.flatten()[indexes], 
                                                    'hum': hum_file.russ.values.flatten()[indexes],
                                                    'deltaT': deltaT[indexes],
                                                    'hour': hours[indexes],
                                                    'elevation' : elevation_flatten[indexes],
                                                    'isrural' : rural[indexes],
                                                    'city' : city})], ignore_index=True)

    return city_df

def plot_avg_deltaT(folder_path):
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i,city in tqdm(enumerate(['Alicante', 'Basel', 'Cologne', 'Glasgow', 'Hamburg', 'Nantes', 'Oslo', 'Porto', 'Turin', 'Zagreb'])):
        temp_file_path = xr.open_dataset(folder_path+'/tas_'+city+'_UrbClim_2011_07_v1.0.nc')
        rural_mask_file_path = xr.open_dataset(folder_path+'/ruralurbanmask_'+city +'_UrbClim_v1.0.nc')
        pixel = np.tile(np.arange(0, temp_file_path.tas.shape[1]*temp_file_path.tas.shape[2]), temp_file_path.tas.shape[0])
        deltaT = compute_deltaT_urban(temp_file_path, rural_mask_file_path)
        df = pd.DataFrame({'deltaT': deltaT, 'pixel': pixel})
        df_mean_dt = df.groupby('pixel').mean()
        df_mean_dt_values = df_mean_dt.deltaT.values.reshape(temp_file_path.tas.shape[1],temp_file_path.tas.shape[2])
        df_mean_dt_values = np.flipud(df_mean_dt_values)
        axs[int(i/5), i%5].imshow(df_mean_dt_values, cmap='Spectral_r') 
        axs[int(i/5), i%5].set_title(city)

    #share the colorbar for all the subplots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axs[0,0].imshow(df_mean_dt_values, cmap='Spectral_r'), cax=cbar_ax)
    plt.show()


def process_data_city(folder_path, pop_day, pop_night, elevation, number_of_sample_per_city, city):
    '''Create a dataframe with, for each city, the temperature, the population, the wind speed, the humidity and compute the delta of
    temperature between urban and rural areas and add it to the dataframe'''
    city_df = pd.DataFrame(columns=['temp', 'pop', 'wind', 'hum', 'deltaT', 'hour', 'city'])

    temp_file_path = folder_path+'/tas_'+city+'_UrbClim_2011_07_v1.0.nc'
    wind_file_path = folder_path+'/sfcWind_'+city +'_UrbClim_2011_07_v1.0.nc'
    hum_file_path = folder_path+'/russ_'+city +'_UrbClim_2011_07_v1.0.nc'
    rural_mask_file_path = folder_path+'/ruralurbanmask_'+city +'_UrbClim_v1.0.nc'
    temp_file = xr.open_dataset(temp_file_path)
    wind_file = xr.open_dataset(wind_file_path)
    hum_file = xr.open_dataset(hum_file_path)
    rural_mask_file = xr.open_dataset(rural_mask_file_path)


    cropped_pop_day, cropped_pop_night = crop_and_downgrade(pop_day, pop_night, temp_file)
    elevation_city = crop_image(elevation, temp_file)

    pop_day_city = resample_image(cropped_pop_day, temp_file.tas[0,:,:].shape)
    pop_night_city = resample_image(cropped_pop_night, temp_file.tas[0,:,:].shape)

    elevation_city = resample_image(elevation_city, temp_file.tas[0,:,:].shape)
    elevation_flatten = np.tile(elevation_city.flatten(), temp_file.tas.shape[0])

    populations = np.concatenate([np.tile(pop_night_city.flatten(),8), np.tile(pop_day_city.flatten(), 12), np.tile(pop_night_city.flatten(), 4)])
    populations = np.tile(populations, 31)

    day_hours = np.tile(np.arange(0,24), temp_file.x.shape[0]*temp_file.y.shape[0])
    hours = np.tile(day_hours.reshape(temp_file.x.shape[0]*temp_file.y.shape[0], 24).flatten(order='F'), 31)

    deltaT = compute_deltaT_urban(temp_file, rural_mask_file)
    rural = np.tile(rural_mask_file.ruralurbanmask.values.flatten(), temp_file.tas.shape[0])
    print(rural.shape, deltaT.shape)
    city = np.tile(np.array([city]), number_of_sample_per_city)

    #generate random indexes to sample the data
    indexes = np.random.randint(0, temp_file.tas.shape[0]*temp_file.tas.shape[1]*temp_file.tas.shape[2], number_of_sample_per_city)

    city_df = pd.DataFrame({'temp': temp_file.tas.values.flatten()[indexes],
                                                'pop':populations[indexes], 
                                                'wind': wind_file.sfcWind.values.flatten()[indexes], 
                                                'hum': hum_file.russ.values.flatten()[indexes],
                                                'deltaT': deltaT[indexes],
                                                'hour': hours[indexes],
                                                'city' : city,
                                                'elevation' : elevation_flatten[indexes],
                                                'isrural' : rural[indexes]
                                                })

    return city_df
