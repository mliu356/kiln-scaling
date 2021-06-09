#!/usr/bin/env python
# coding: utf-8

# ## To-Do
# * read composite images directly from drive
# 
# ## To-Done
# * only get negative examples completely in bangladesh
# * only need to snip the cols in the last row and the rows in the last col
# * create github
# * save data to .data
# * don't upload .data to github

# In[1]:


import os
import json
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


# In[5]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# gauth = GoogleAuth()
# option 1
# gauth.LocalWebserverAuth()

# option 2
# gauth.CommandLineAuth()
# drive = GoogleDrive(gauth)

# option 3
gauth = GoogleAuth()
auth_url = gauth.GetAuthUrl() # Create authentication url user needs to visit
print(auth_url)


# In[7]:





# In[22]:


code = input("Type google drive code: ")
# code = "4/1AY0e-g4WrCj-ofnl0l2LdbZbIZBL1H_Bw47ueqMkLHIIRJ-HYlChJs35avs" 
gauth.Auth(code) # Authorize and build service from the code
drive = GoogleDrive(gauth)


# In[6]:


# set params
tile_height, tile_length = (64, 64)
bands_to_read = None # ['B4', 'B3', 'B2'] # set to None to read all bands
examples_per_save_file = 1000
save_path = '/atlas/u/mhelabd/data/kiln-scaling/tiles/'
composite_file_name = 'bangladesh_all_bands_final'
composite_save_path = '/atlas/u/mhelabd/data/kiln-scaling/composites/'

# resources
kilns = pd.read_csv("../data/bangladesh_kilns.csv")
all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B10', 'B11', 'B12']
band_dict = dict(zip(all_bands, range(1, len(all_bands) + 1)))

print(kilns.head())


# In[23]:


file_list = drive.ListFile({'q': "title contains '" + composite_file_name + "'"}).GetList()
print("Found " + str(len(file_list)) + " files")
file_list = sorted(file_list, key=lambda file: file['title'])
for file in file_list[:5]:
  print('title: %s, id: %s' % (file['title'], file['id']))


# In[24]:


# calculate image grid
first_x_coord = file_list[0]['title'].split(".")[0].split("-")[1]
first_y_coord = file_list[0]['title'].split(".")[0].split("-")[2]
num_image_cols = len([x for x in file_list if x['title'].split(".")[0].split("-")[1] == first_x_coord])
num_image_rows = len([x for x in file_list if x['title'].split(".")[0].split("-")[2] == first_y_coord])
print("Number of image grid columns:", num_image_cols)
print("Number of image grid rows:", num_image_rows)


# In[25]:


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


# In[43]:


# print(datasets[0])
# print(datasets[0].bounds)


# In[26]:


# testing variables
num_tiles_dropped = 0


# In[30]:


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

def get_data_given_pixels(ds_bounds, bands, start_row, start_col, contains_kiln):
    global num_tiles_dropped
    
    num_bands, ds_height, ds_length = bands.shape
    tile_top = ds_bounds.top - (start_row / ds_height) * (ds_bounds.top - ds_bounds.bottom)
    tile_bottom = ds_bounds.top - ((start_row + tile_height) / ds_height) * (ds_bounds.top - ds_bounds.bottom)
    tile_left = ds_bounds.left + (start_col / ds_length) * (ds_bounds.right - ds_bounds.left)
    tile_right = ds_bounds.left + ((start_col + tile_length) / ds_length) * (ds_bounds.right - ds_bounds.left)
    tile_coordinates = [[tile_left, tile_top], [tile_right, tile_top], [tile_right, tile_bottom], [tile_left, tile_bottom]]
#     print("top", tile_top, "bottom", tile_bottom, "left", tile_left, "right", tile_right)
    if not contains_kiln:
        for point in [Point([c[0], c[1]]) for c in tile_coordinates]:
            if not bangladesh_geo.contains(point):
                num_tiles_dropped += 1
                return None
    return bands[:, start_row : start_row + tile_height, start_col : start_col + tile_length]

# def add_example(ex_data, save_index, list_to_add, is_positive):
def add_example(ex_data, save_index, counter, examples, is_positive):
    examples[counter] = ex_data
    new_counter = counter + 1
    # save files if needed
    if new_counter == examples_per_save_file:
        filename = save_path + ("pos" if is_positive else "neg") + "_examples_" + str(save_index) + ".csv"
        print("Saving file", filename)
        np.savetxt(filename, examples.flatten(), delimiter=",")
        print("Finished saving file")
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
        


# In[13]:


# data_pixels = []
# for ds in datasets:
#     data_pixels += [get_bands_from_dataset(ds, band_names=bands_to_read)]
    
# print("Data shape:", data_pixels[0].shape)


# In[28]:


pos_save_index, neg_save_index = 0, 0
pos_counter, neg_counter = 0, 0

pos_x_examples = np.zeros([examples_per_save_file, len(all_bands) if bands_to_read is None else len(bands_to_read), tile_height, tile_length])
neg_x_examples = np.zeros([examples_per_save_file, len(all_bands) if bands_to_read is None else len(bands_to_read), tile_height, tile_length])
print("Examples shape:", pos_x_examples.shape)


# In[1]:


file_list = file_list[:1] # testing purposes

total_start_time = time.time()
for index, file in enumerate(file_list):
    file_start_time = time.time()
    bands, ds_bounds = get_bands_and_bounds_from_file(file, band_names=bands_to_read)
    
    file_in_country = False
    file_coordinates = [[ds_bounds.left, ds_bounds.top], [ds_bounds.right, ds_bounds.top], [ds_bounds.right, ds_bounds.bottom], [ds_bounds.left, ds_bounds.bottom]]
    for point in [Point([c[0], c[1]]) for c in tile_coordinates]:
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
            bounds["right"] -= col_px_excess / ds_length * (ds.bounds.right - ds.bounds.left)

        if index // num_image_cols == num_image_rows - 1: # last row
            # calculate excess row pixels
            row_px_excess = ds_height % tile_height
            bounds["bottom"] += row_px_excess / ds_height * (ds.bounds.top - ds.bounds.bottom)

        kiln_tiles = get_kiln_tiles(bounds, num_rows, num_cols)
        print("Num tiles with kilns:", len(kiln_tiles))
        print(kiln_tiles)

        print("Tiling dataset...")
        for tile_idx_row in range(0, num_rows):
            px_row = tile_idx_row * tile_height
            for tile_idx_col in range(0, num_cols):
                px_col = tile_idx_col * tile_length
                if (tile_idx_row, tile_idx_col) in kiln_tiles:
                    data = get_data_given_pixels(ds_bounds, bands, px_row, px_col, True)
                    pos_counter, pos_save_index = add_example(data, pos_save_index, pos_counter, pos_x_examples, True)
                else:
                    data = get_data_given_pixels(ds_bounds, bands, px_row, px_col, False)
                    if data is not None:
                        neg_counter, neg_save_index = add_example(data, neg_save_index, neg_counter, neg_x_examples, False)
        print("Total tiles dropped (outside country):", num_tiles_dropped)
        print("Total tiles kept:", str(num_rows * num_cols - num_tiles_dropped))
        num_tiles_dropped = 0
        print("Finished file in", time.time() - file_start_time, "\n")
print("Finished " + str(len(file_list)) + " files in: " + str(time.time() - total_start_time))


# In[ ]:


print(pos_x_examples[12])
print(tile_idx_row, tile_idx_col)


# In[ ]:





# In[ ]:




