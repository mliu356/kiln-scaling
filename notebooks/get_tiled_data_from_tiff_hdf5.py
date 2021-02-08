#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import json
import sys
import requests
import time
from requests.auth import HTTPBasicAuth
import pandas as pd
import datetime
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from os import listdir, path
from os.path import isfile, join
import h5py


# In[8]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth)


# In[6]:


# set params
tile_height, tile_length = (64, 64)
bands_to_read = None # ['B4', 'B3', 'B2'] # set to None to read all bands
examples_per_save_file = 1000
composite_file_name = 'bangladesh_all_bands_final'
download_all_first = False

save_path = '/atlas/u/{}/data/kiln-scaling/tiles/'.format(sys.argv[0])
composite_save_path = '/atlas/u/{}/data/kiln-scaling/composites/'.format(sys.argv[0])

# save_path = '../data/tiles_hdf5/'
# composite_save_path = '../data/composites/'

# resources
kilns = pd.read_csv("../data/bangladesh_kilns.csv")
all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B10', 'B11', 'B12']
band_dict = dict(zip(all_bands, range(1, len(all_bands) + 1)))

print(kilns.head())


# In[7]:


file_list = drive.ListFile({'q': "title contains '" + composite_file_name + "'"}).GetList()
print("Found " + str(len(file_list)) + " files")
file_list = sorted(file_list, key=lambda file: file['title'])
for file in file_list[:5]:
  print('title: %s, id: %s' % (file['title'], file['id']))


# In[8]:


# calculate image grid
first_x_coord = file_list[0]['title'].split(".")[0].split("-")[1]
first_y_coord = file_list[0]['title'].split(".")[0].split("-")[2]
num_image_cols = len([x for x in file_list if x['title'].split(".")[0].split("-")[1] == first_x_coord])
num_image_rows = len([x for x in file_list if x['title'].split(".")[0].split("-")[2] == first_y_coord])
print("Number of image grid columns:", num_image_cols)
print("Number of image grid rows:", num_image_rows)


# In[9]:


coords = []
with open("../data/countries.geojson", "r") as countries_geojson:
    country_dict = json.load(countries_geojson)["features"]
for obj in country_dict:
    name = obj['properties']['ADMIN']
    if name == "Bangladesh":
        coords = obj['geometry']["coordinates"]
flat_coords = []
for sublist in coords:
    for coord in sublist:
        for c in coord:
            flat_coords.append(c)

bangladesh_geo = Polygon(flat_coords)


# In[10]:


# testing variables
num_tiles_dropped = 0

# optional pre-download all files
if download_all_first:
    for file in file_list:
        start_time = time.time()
        composite_file_path = composite_save_path + file['title']
        if path.exists(composite_file_path):
            print("File already downloaded.")
        else:
            print("Downloading file...")
            # download the file
            download_file = drive.CreateFile({'id': file['id']})
            file.GetContentFile(composite_file_path)
            print("Finished file in " + str(time.time() - start_time))
    print("Done downloading all files.")


# In[ ]:


def get_bands_and_bounds_from_file(file, band_names=None):
    print("Starting file " + file['title'])
    composite_file_path = composite_save_path + file['title']
    if path.exists(composite_file_path):
        print("File already downloaded.")
    else:
        print("Downloading file...")
        # download the file
        download_file = drive.CreateFile({'id': file['id']})
        file.GetContentFile(composite_file_path)
    
    # open file with rasterio
    print("Reading file...")
    dataset = rasterio.open(composite_file_path)
    if band_names is None: # get all bands
        bands = dataset.read()
    else:
        # TODO refactor using np.index_select
        bands = []
        for idx, band_name in enumerate(band_names):
            band_index = band_dict[band_name]
            bands += [dataset.read(band_index)]
        bands = np.array(bands)
    print("Done processing file")
    return bands, dataset.bounds

def get_data_and_bounds_given_pixels(ds_bounds, bands, start_row, start_col, contains_kiln):
    global num_tiles_dropped
    
    num_bands, ds_height, ds_length = bands.shape
    tile_top = ds_bounds.top - (start_row / ds_height) * (ds_bounds.top - ds_bounds.bottom)
    tile_bottom = ds_bounds.top - ((start_row + tile_height) / ds_height) * (ds_bounds.top - ds_bounds.bottom)
    tile_left = ds_bounds.left + (start_col / ds_length) * (ds_bounds.right - ds_bounds.left)
    tile_right = ds_bounds.left + ((start_col + tile_length) / ds_length) * (ds_bounds.right - ds_bounds.left)
    bounds = np.array([tile_bottom, tile_left, tile_top, tile_right])
    tile_coordinates = [[tile_left, tile_top], [tile_right, tile_top], [tile_right, tile_bottom], [tile_left, tile_bottom]]
#     print("top", tile_top, "bottom", tile_bottom, "left", tile_left, "right", tile_right)
    if not contains_kiln:
        for point in [Point([c[0], c[1]]) for c in tile_coordinates]:
            if not bangladesh_geo.contains(point):
                num_tiles_dropped += 1
                return None, None
    return bands[:, start_row : start_row + tile_height, start_col : start_col + tile_length], bounds

def add_example(ex_data, ex_bounds, save_index, counter, is_positive):
    tile_bounds[counter] = ex_bounds
    examples[counter] = ex_data
    labels[counter] = 1 if is_positive else 0
    new_counter = counter + 1
    
    # save files if needed
    if new_counter == examples_per_save_file:
        filename = save_path + "examples_" + str(save_index) + ".hdf5"
        print("Saving file", filename)
        f = h5py.File(filename, 'w')
        bounds_dset = f.create_dataset("bounds", data=tile_bounds)
        examples_dset = f.create_dataset("images", data=examples)
        labels_dset = f.create_dataset("labels", data=labels)
        f.close()
        
        save_index += 1
        new_counter = 0
    return new_counter, save_index

def get_kiln_tiles(bounds, num_rows, num_cols):
    kilns_in_image = kilns.loc[(kilns['lat'] >= bounds['bottom']) & (kilns['lat'] <= bounds['top']) 
        & (kilns['lon'] >= bounds['left']) & (kilns['lon'] <= bounds['right'])]
    
    tiles = set() # set of tuples of (tile_row, tile_col)
    for index, kiln in kilns_in_image.iterrows():
        lon_percent = 1 - (kiln['lon'] - bounds['left']) / (bounds['right'] - bounds['left'])
        row_index = int(num_rows * lon_percent)
        
        lat_percent = 1 - (kiln['lat'] - bounds['bottom']) / (bounds['top'] - bounds['bottom'])
        col_index = int(num_cols * lat_percent)
        tiles.add((row_index, col_index))
    return tiles


# In[ ]:


save_index, counter = 0, 0

tile_bounds = np.zeros([examples_per_save_file, 4])
examples = np.zeros([examples_per_save_file, len(all_bands) if bands_to_read is None else len(bands_to_read), tile_height, tile_length])
labels = np.zeros([examples_per_save_file, 1])


# In[ ]:


# file_list = file_list[:1] # testing purposes

total_start_time = time.time()
for index, file in enumerate(file_list):
    file_start_time = time.time()
    bands, ds_bounds = get_bands_and_bounds_from_file(file, band_names=bands_to_read)
    
    file_in_country = False
    file_coordinates = [[ds_bounds.left, ds_bounds.top], [ds_bounds.right, ds_bounds.top], [ds_bounds.right, ds_bounds.bottom], [ds_bounds.left, ds_bounds.bottom]]
    for point in [Point([c[0], c[1]]) for c in file_coordinates]:
        if bangladesh_geo.contains(point):
            file_in_country = True
    
    if file_in_country:
        num_bands, ds_height, ds_length = bands.shape
        num_rows = ds_height // tile_height
        num_cols = ds_length // tile_length

        bounds = {
            "bottom": ds_bounds.bottom,
            "top": ds_bounds.top,
            "left": ds_bounds.left,
            "right": ds_bounds.right
        }

        row_px_excess, col_px_excess = None, None
        percent_row_excess, percent_col_excess = None, None

        if index % num_image_cols == num_image_cols - 1: # rightmost column
            # calculate excess col pixels
            col_px_excess = ds_length % tile_length
            bounds["right"] -= col_px_excess / ds_length * (ds_bounds.right - ds_bounds.left)

        if index // num_image_cols == num_image_rows - 1: # last row
            # calculate excess row pixels
            row_px_excess = ds_height % tile_height
            bounds["bottom"] += row_px_excess / ds_height * (ds_bounds.top - ds_bounds.bottom)

        kiln_tiles = get_kiln_tiles(bounds, num_rows, num_cols)
        print("Num tiles with kilns:", len(kiln_tiles))
        print(kiln_tiles)

        print("Tiling dataset...")
        for tile_idx_row in range(0, num_rows):
            px_row = tile_idx_row * tile_height
            for tile_idx_col in range(0, num_cols):
                px_col = tile_idx_col * tile_length
                tile_has_kiln = (tile_idx_row, tile_idx_col) in kiln_tiles
                data, data_bounds = get_data_and_bounds_given_pixels(ds_bounds, bands, px_row, px_col, tile_has_kiln)
                if data is not None:
                    counter, save_index = add_example(data, data_bounds, save_index, counter, tile_has_kiln)
                
        # TODO: handle leftovers

        print("Total tiles dropped (outside country):", num_tiles_dropped)
        print("Total tiles kept:", str(num_rows * num_cols - num_tiles_dropped))
        num_tiles_dropped = 0
        print("Finished file in", time.time() - file_start_time, "\n")
print("Finished " + str(len(file_list)) + " files in: " + str(time.time() - total_start_time))


# In[ ]:


# f = h5py.File('../data/tiles_hdf5/examples_0.hdf5','r')
# print("all keys", f.keys())
# print(f['images'].shape)
# print(f['bounds'].shape)
# print(f['labels'].shape)


# In[ ]:


print(tile_bounds.shape)


# In[ ]:




