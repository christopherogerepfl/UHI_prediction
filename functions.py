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
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap, LogNorm



def resample_image(image_to_resample, new_dimensions):
    '''Given a 2D array, increase its resolution to new dimensions, with no interpolation'''
    new_image = np.zeros(new_dimensions)
    for i in range(new_dimensions[0]):
        for j in range(new_dimensions[1]):
            new_image[i,j] = image_to_resample[int(i*image_to_resample.shape[0]/new_dimensions[0]), int(j*image_to_resample.shape[1]/new_dimensions[1])]
    return new_image

def crop_and_downgrade(pop_day_tiff=None, pop_night_tiff=None, temp_city=None):

    min_lon = temp_city.longitude.min().values.item()
    max_lon = temp_city.longitude.max().values.item()
    min_lat = temp_city.latitude.min().values.item()
    max_lat = temp_city.latitude.max().values.item()

    # crop the tiff with the city bounds
    cropped_pop_day = pop_day_tiff.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=pop_day_tiff.transform))
    cropped_pop_night = pop_night_tiff.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=pop_night_tiff.transform))
    #cropped_pop_night = pop_night_tiff.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=pop_night_tiff.transform))

    #downgrade the resolution of the temp_city to the resolution of the population tiff
    dim1 = cropped_pop_day.shape[0]
    dim2 = cropped_pop_day.shape[1]

    temp_city_down = temp_city.coarsen(x=int(temp_city.x.shape[0]/dim2), y=int(temp_city.y.shape[0]/dim1), boundary='trim').mean()


    #convert the latitude in meters
    min_lat = 110574 * min_lat
    max_lat = 110574 * max_lat

    #convert the longitude in meters
    min_lon = 111320 * min_lon
    max_lon = 111320 * max_lon

    return cropped_pop_day, cropped_pop_night

def crop_image(image_to_crop, temp_city):
    min_lon = temp_city.longitude.min().values.item()
    max_lon = temp_city.longitude.max().values.item()
    min_lat = temp_city.latitude.min().values.item()
    max_lat = temp_city.latitude.max().values.item()

    # crop the tiff with the city bounds
    cropped_image = image_to_crop.read(1, window=rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, transform=image_to_crop.transform))
    return cropped_image


def compute_deltaT_urban(temperature_ds, urban_mask_ds, hour=None):
    deltaT_ds = []
    if hour is not None:
        rural_avg = temperature_ds.tas[hour,:,:].where(urban_mask_ds.ruralurbanmask[:,:] == 1).mean(dim=['x','y']).values.item()
        deltaT_ds.append(temperature_ds.tas[hour,:,:].values - rural_avg)
        deltaT_ds = np.array(deltaT_ds).flatten()
    else:
        for i in range(temperature_ds.time.shape[0]):
            rural_avg = temperature_ds.tas[i,:,:].where(urban_mask_ds.ruralurbanmask[:,:] == 1).mean(dim=['x','y']).values.item()
            deltaT_ds.append(temperature_ds.tas[i,:,:].values - rural_avg)
        deltaT_ds = np.array(deltaT_ds).flatten()

    return deltaT_ds

def process_data(elevation, land_cover, NDVI,number_of_sample_per_hour=10000, cities=[]):
    '''Create a dataframe with, for each city, the temperature, the population, the wind speed, the humidity and compute the delta of
    temperature between urban and rural areas and add it to the dataframe'''
    city_df = pd.DataFrame(columns=['temp', 'pop', 'wind', 'hum', 'deltaT', 'hour', 'month', 'elevation', 'city', 'land cover type', 'NDVI', 'isrural'])
    for city in tqdm(cities):
        #crop the tif images to the city bounds
        temp_file_path = 'data_cities/'+str.lower(city)+'_data'+'/tas_'+city+'_UrbClim_2011_'+'0'+'1'+'_v1.0.nc'   
        temp_file = xr.open_dataset(temp_file_path)    

        elevation_city = crop_image(elevation, temp_file)
        lc_city = crop_image(land_cover, temp_file)

        min_lon = temp_file.longitude.min().values.item()
        max_lon = temp_file.longitude.max().values.item()
        min_lat = temp_file.latitude.min().values.item()
        max_lat = temp_file.latitude.max().values.item()

        NDVI_masked = NDVI.sel(lon=slice(min_lon,max_lon),lat=slice(max_lat,min_lat))
        NDVI_clean = NDVI_masked.NDVI.values[0,:,:]


        NDVI_resampled = resample_image(NDVI_clean, temp_file.tas[0,:,:].shape)
        NDVI_flatten = np.tile(NDVI_resampled.flatten(), temp_file.tas.shape[0])

        

        elevation_city = resample_image(elevation_city, temp_file.tas[0,:,:].shape)
        elevation_flatten = np.tile(elevation_city.flatten(), temp_file.tas.shape[0])

        lc_city = resample_image(lc_city, temp_file.tas[0,:,:].shape)
        lc_flatten = np.tile(lc_city.flatten(), temp_file.tas.shape[0]) 

        for month in range(1,13):
            #Open the datasets
            temp_file_path = 'data_cities/'+str.lower(city)+'_data'+'/tas_'+city+'_UrbClim_2011_'+str(month).zfill(2)+'_v1.0.nc'
            wind_file_path = 'data_cities/'+str.lower(city)+'_data'+'/sfcWind_'+city +'_UrbClim_2011_'+str(month).zfill(2)+'_v1.0.nc'
            hum_file_path = 'data_cities/'+str.lower(city)+'_data'+'/russ_'+city +'_UrbClim_2011_'+str(month).zfill(2)+'_v1.0.nc'
            rural_mask_file_path = 'data_cities/'+str.lower(city)+'_data'+'/ruralurbanmask_'+city +'_UrbClim_v1.0.nc'
            landseamask_file_path = 'data_cities/'+str.lower(city)+'_data'+'/landseamask_'+city+'_UrbClim_v1.0.nc'

            rural_mask_file = xr.open_dataset(rural_mask_file_path, )
            landseamask_file = xr.open_dataset(landseamask_file_path)
            temp_file = xr.open_dataset(temp_file_path).where(landseamask_file.landseamask == 1, drop=True)
            wind_file = xr.open_dataset(wind_file_path).where(landseamask_file.landseamask == 1, drop=True)
            hum_file = xr.open_dataset(hum_file_path).where(landseamask_file.landseamask == 1, drop=True)

            datasets = [temp_file, wind_file, hum_file, rural_mask_file, landseamask_file]

            pop_day = rio.open('data_population_day\ENACT_POP_D'+str(month).zfill(2)+'2011_EU28_R2020A_4326_30ss_V1_0.tif')
            pop_night = rio.open('data_population_night\ENACT_POP_N'+str(month).zfill(2)+'2011_EU28_R2020A_4326_30ss_V1_0.tif')
            cropped_pop_day, cropped_pop_night = crop_and_downgrade(pop_day, pop_night, temp_file)

            pop_day_city = resample_image(cropped_pop_day, temp_file.tas[0,:,:].shape)
            pop_night_city = resample_image(cropped_pop_night, temp_file.tas[0,:,:].shape)

            populations = np.concatenate([np.tile(pop_night_city.flatten(),8), np.tile(pop_day_city.flatten(), 12), np.tile(pop_night_city.flatten(), 4)])
            populations = np.tile(populations, 31)

            day_hours = np.tile(np.arange(0,24), temp_file.x.shape[0]*temp_file.y.shape[0])
            hours = np.tile(day_hours.reshape(temp_file.x.shape[0]*temp_file.y.shape[0], 24).flatten(order='F'), 31)

            deltaT = compute_deltaT_urban(temp_file, rural_mask_file)
            rural = np.tile(rural_mask_file.ruralurbanmask.values.flatten(), temp_file.tas.shape[0])
            city_list = np.tile(np.array([city]), len(hours))

            #generate random indexes to sample the data
            indexes = np.random.randint(0, temp_file.tas.shape[0]*temp_file.tas.shape[1]*temp_file.tas.shape[2], number_of_sample_per_hour)
            indexes_2 = indexes[temp_file.tas.values.flatten()[indexes] > 0]
            months = np.tile(np.array([month]), len(indexes_2))

            city_df = pd.concat([city_df, pd.DataFrame({'temp': temp_file.tas.values.flatten()[indexes_2],
                                                        'pop':populations[indexes_2], 
                                                        'wind': wind_file.sfcWind.values.flatten()[indexes_2], 
                                                        'hum': hum_file.russ.values.flatten()[indexes_2],
                                                        'deltaT': deltaT[indexes_2],
                                                        'hour': hours[indexes_2],
                                                        'month': months,
                                                        'elevation' : elevation_flatten[indexes_2],
                                                        'land cover type':lc_flatten[indexes_2],
                                                        'NDVI': NDVI_flatten[indexes_2],
                                                        'isrural': rural[indexes_2],
                                                        'city' : city_list[:len(indexes_2)]})])
            
            for dataset in datasets:
                dataset.close()
            
            print('processing '+city+' month '+str(month)+' done')

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


def process_data_city(folder_path, pop_day, pop_night, elevation, lc, number_of_sample_per_city, city):
    '''Create a dataframe with, for each city, the temperature, the population, the wind speed, the humidity and compute the delta of
    temperature between urban and rural areas and add it to the dataframe'''
    city_df = pd.DataFrame(columns=['temp', 'pop', 'wind', 'hum', 'deltaT', 'hour','city'])

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
    lc_city = crop_image(lc, temp_file)

    pop_day_city = resample_image(cropped_pop_day, temp_file.tas[0,:,:].shape)
    pop_night_city = resample_image(cropped_pop_night, temp_file.tas[0,:,:].shape)

    elevation_city = resample_image(elevation_city, temp_file.tas[0,:,:].shape)
    elevation_flatten = np.tile(elevation_city.flatten(), temp_file.tas.shape[0])

    lc_city = resample_image(lc_city, temp_file.tas[0,:,:].shape)
    lc_flatten = np.tile(lc_city.flatten(), temp_file.tas.shape[0])

    populations = np.concatenate([np.tile(pop_night_city.flatten(),8), np.tile(pop_day_city.flatten(), 12), np.tile(pop_night_city.flatten(), 4)])
    populations = np.tile(populations, 31)

    day_hours = np.tile(np.arange(0,24), temp_file.x.shape[0]*temp_file.y.shape[0])
    hours = np.tile(day_hours.reshape(temp_file.x.shape[0]*temp_file.y.shape[0], 24).flatten(order='F'), 31)

    deltaT = compute_deltaT_urban(temp_file, rural_mask_file)
    rural = np.tile(rural_mask_file.ruralurbanmask.values.flatten(), temp_file.tas.shape[0])
    print(rural.shape, deltaT.shape)
    city = np.tile(np.array([city]), number_of_sample_per_city)

    latitude = np.tile(temp_file.latitude.values.flatten(), temp_file.tas.shape[0])
    longitude = np.tile(temp_file.longitude.values.flatten(), temp_file.tas.shape[0])

    #generate random indexes to sample the data
    indexes = np.random.randint(0, temp_file.tas.shape[0]*temp_file.tas.shape[1]*temp_file.tas.shape[2], number_of_sample_per_city)

    city_df = pd.DataFrame({'temp': temp_file.tas.values.flatten()[indexes],
                            'pop':populations[indexes], 
                            'wind': wind_file.sfcWind.values.flatten()[indexes], 
                            'hum': hum_file.russ.values.flatten()[indexes],
                            'deltaT': deltaT[indexes],
                            'hour': hours[indexes],
                            'elevation' : elevation_flatten[indexes],
                            'isrural' : rural[indexes],
                            'land cover type':lc_flatten[indexes],
                            'city' : city,
                            'latitude' : latitude[indexes],
                            'longitude' : longitude[indexes]
                            })

    return city_df



# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
