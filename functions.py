import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


def bound_pop(temp_ds, pop_tif):
    x_min = temp_ds.latitude.min().values.item()
    x_max = temp_ds.latitude.max().values.item()
    y_min = temp_ds.longitude.min().values.item()
    y_max = temp_ds.longitude.max().values.item()

    bounded_pop = pop_tif.read(1, window=rio.windows.from_bounds(y_min, x_min, y_max, x_max, transform=pop_tif.transform))
    return bounded_pop

def moving_average(a, n=3):
    ''' Moving average of a 1D array '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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

    # downsample the temp_city to match the cropped_pop resolution
    temp_city = temp_city.coarsen(y=int(temp_city.y.shape[0]/cropped_pop_day.shape[0]), x=int(temp_city.x.shape[0]/cropped_pop_day.shape[1]), boundary='trim').mean()

    #reverse the y axis of the temp_city
    temp_city = temp_city.reindex(y=temp_city.y[::-1])

    return cropped_pop_day,cropped_pop_night,  temp_city

def dataviz_UHI(temp_xarray, pop_night, pop_day):
    temp_xarray.tas[0,:,:].plot(cmap = 'Spectral_r')
    plt.title('Temperature')
    plt.show()

    cropped_pop_day,cropped_pop_night, temp_city = crop_and_downgrade(pop_day, pop_night, temp_xarray)
    sub1, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].imshow(cropped_pop_day, cmap='Spectral_r')
    ax[0].set_title('Population during the day')
    ax[1].imshow(cropped_pop_night, cmap='Spectral_r')
    ax[1].set_title('Population during the night')
    ax[2].imshow(temp_city[0,:,:], cmap='Spectral_r')
    ax[2].set_title('Temperature')

def compute_deltaT(temperature_ds, urban_mask_ds):
    deltaT_ds = temperature_ds.copy()
 
    for i in range(temperature_ds.time.shape[0]):
        avg_temp_rural = temperature_ds.tas[i,:,:].where(urban_mask_ds.ruralurbanmask == 1).mean(dim=['x','y'])
        deltaT_ds.tas[i,:,:] = deltaT_ds.tas[i,:,:] - avg_temp_rural
    return deltaT_ds

def mplReg(X_train, X_test, y_train, y_test, hidden_layer_sizes=(10,10,10), max_iter=100, verbose=True, learning_rate='constant', solver = 'adam'):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, verbose=verbose, learning_rate=learning_rate, solver = solver)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    print('MSE: ', mean_squared_error(y_test, predictions))
    print('R2: ', r2_score(y_test, predictions))

