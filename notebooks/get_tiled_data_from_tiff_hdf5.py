#!/usr/bin/env python
# coding: utf-8

# In[4]:


# !pip install pandas
# jupyter nbconvert --to script get_tiled_data_from_tiff_hdf5.ipynb


# In[6]:


import os
import json
import time
import pandas as pd
import datetime
import rasterio
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from os import path
import h5py
import geopy.distance


# In[2]:


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


# In[250]:


# set params
tile_height, tile_length = (64, 64)
examples_per_save_file = 1000
composite_file_name = 'bangladesh_all_bands_final'
download_all_first = True

# save_path = '/atlas/u/mhelabd/data/kiln-scaling/tiles/'
# composite_save_path = '/atlas/u/mhelabd/data/kiln-scaling/composites/'

save_path = '/atlas/u/mliu356/data/kiln-scaling/tiles_drop_neighbors/'
composite_save_path = '/atlas/u/mliu356/data/kiln-scaling/composites/'

# save_path = '../data/tiles_hdf5/'
# composite_save_path = '../data/composites/'

# resources
kilns = pd.read_csv("../data/bangladesh_kilns.csv")
all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B10', 'B11', 'B12']

print(kilns.head())


# In[ ]:


def mkdirs(names):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)
mkdirs([save_path, composite_save_path])


# In[5]:


file_list = drive.ListFile({'q': "title contains '" + composite_file_name + "'"}).GetList()
print("Found " + str(len(file_list)) + " files")
file_list = sorted(file_list, key=lambda file: file['title'])
for file in file_list[:5]:
  print('title: %s, id: %s' % (file['title'], file['id']))


# In[6]:


# calculate image grid
first_x_coord = file_list[0]['title'].split(".")[0].split("-")[1]
first_y_coord = file_list[0]['title'].split(".")[0].split("-")[2]
num_image_cols = len([x for x in file_list if x['title'].split(".")[0].split("-")[1] == first_x_coord])
num_image_rows = len([x for x in file_list if x['title'].split(".")[0].split("-")[2] == first_y_coord])
print("Number of image grid columns:", num_image_cols)
print("Number of image grid rows:", num_image_rows)


# In[92]:


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
            
flat_coords += [flat_coords[0]]
bangladesh_geo = Polygon(flat_coords)


# In[93]:


# optional pre-download all files
if download_all_first:
    for file in file_list:
        start_time = time.time()
        composite_file_path = composite_save_path + file['title']
        if path.exists(composite_file_path):
            print("File already downloaded.", composite_file_path)
        else:
            print("Downloading file...")
            # download the file
            download_file = drive.CreateFile({'id': file['id']})
            file.GetContentFile(composite_file_path)
            print("Finished file in " + str(time.time() - start_time))
    print("Done downloading all files.")


# In[228]:


def get_bands_and_bounds_from_file(file):
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
    bands = dataset.read()
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
    tile_geo = Polygon([[tile_left, tile_top], [tile_right, tile_top], [tile_right, tile_bottom], [tile_left, tile_bottom], [tile_left, tile_top]])
    if not contains_kiln and not bangladesh_geo.intersects(tile_geo):
        num_tiles_dropped += 1
        return None, None
    return bands[:, start_row : start_row + tile_height, start_col : start_col + tile_length], bounds

def save_current_file(save_index, counter):
    filename = save_path + "examples_" + str(save_index) + ".hdf5"
    print("Saving file", filename)
    f = h5py.File(filename, 'w')
    bounds_dset = f.create_dataset("bounds", data=tile_bounds[:counter])
    examples_dset = f.create_dataset("images", data=examples[:counter])
    labels_dset = f.create_dataset("labels", data=labels[:counter])
    f.close()
    return save_index + 1, 0

def add_example(ex_data, ex_bounds, save_index, counter, is_positive):
    tile_bounds[counter] = ex_bounds
    examples[counter] = ex_data
    labels[counter] = 1 if is_positive else 0
    new_counter = counter + 1
    
    if new_counter == examples_per_save_file:
        return save_current_file(save_index, counter)
    return save_index, new_counter

def get_kilns_and_drop_tiles(bounds, num_rows, num_cols):
    kilns_in_image = kilns.loc[(kilns['lat'] >= bounds['bottom']) & (kilns['lat'] <= bounds['top']) 
        & (kilns['lon'] >= bounds['left']) & (kilns['lon'] <= bounds['right'])]
    
    drop = set()
    tiles = set()
    for index, kiln in kilns_in_image.iterrows():
        kiln_pos = (kiln['lat'], kiln['lon'])
        kiln_to_top = geopy.distance.geodesic(kiln_pos, (bounds['top'], kiln_pos[1])).km
        kiln_to_bottom = geopy.distance.geodesic(kiln_pos, (bounds['bottom'], kiln_pos[1])).km
        row_index = int(kiln_to_top / (kiln_to_top + kiln_to_bottom) * num_rows)
        
        kiln_to_left = geopy.distance.geodesic(kiln_pos, (kiln_pos[0], bounds['left'])).km
        kiln_to_right = geopy.distance.geodesic(kiln_pos, (kiln_pos[0], bounds['right'])).km
        col_index = int(kiln_to_left / (kiln_to_left + kiln_to_right) * num_cols)
        new_tile = (row_index, col_index)
        
        tiles.add(new_tile)
        drop.discard(new_tile)
        
        neighbors = [(row_index - 1, col_index - 1), (row_index, col_index - 1), (row_index + 1, col_index - 1), 
                     (row_index - 1, col_index), (row_index + 1, col_index), 
                     (row_index - 1, col_index + 1), (row_index, col_index + 1), (row_index + 1, col_index + 1)]
        for n in neighbors:
            if n not in tiles and n[0] >= 0 and n[0] < num_rows and n[1] >= 0 and n[1] < num_cols:
                drop.add(n)
        
    return tiles, drop


# In[126]:


## testing & visualization methods

# image is a single example of shape (13, 64, 64)
def visualize_tile(image, indices=[3, 2, 1]):
    row_idx = np.array(indices)
    X = np.transpose(image[row_idx], (1, 2, 0))
    X *= 1 / np.max(X)
    print(X.shape)
    plt.imshow(X)
    
def pretty_bounds(bounds):
    return [[bounds[1], bounds[0]], [bounds[1], bounds[2]], [bounds[3], bounds[2]], [bounds[3], bounds[0]], [bounds[1], bounds[0]]]


# In[309]:


## testing variables
num_tiles_dropped = 0
# pos_examples = []

save_index, counter = 0, 0

tile_bounds = np.zeros([examples_per_save_file, 4])
examples = np.zeros([examples_per_save_file, len(all_bands), tile_height, tile_length])
labels = np.zeros([examples_per_save_file, 1])


# In[233]:


# file_list = file_list[:1] # testing purposes

total_start_time = time.time()
for index, file in enumerate(file_list):
    file_start_time = time.time()
    bands, ds_bounds = get_bands_and_bounds_from_file(file)
    
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

    kiln_tiles, drop_tiles = get_kilns_and_drop_tiles(bounds, num_rows, num_cols)
    print("Num tiles with kilns:", len(kiln_tiles))
    print(kiln_tiles)

    print("Tiling dataset...")
    for tile_idx_row in range(0, num_rows):
        px_row = tile_idx_row * tile_height
        for tile_idx_col in range(0, num_cols):
            px_col = tile_idx_col * tile_length
            drop_tile = (tile_idx_row, tile_idx_col) in drop_tiles
            if not drop_tile:
                tile_has_kiln = (tile_idx_row, tile_idx_col) in kiln_tiles
                data, data_bounds = get_data_and_bounds_given_pixels(ds_bounds, bands, px_row, px_col, tile_has_kiln)
                if data is not None:
                    save_index, counter = add_example(data, data_bounds, save_index, counter, tile_has_kiln)

    # handle leftovers in a final file
    if index == len(file_list) - 1:
        save_current_file(save_index, counter)

    print("Total tiles dropped (outside country):", num_tiles_dropped)
    print("Total tiles kept:", str(num_rows * num_cols - num_tiles_dropped))
    num_tiles_dropped = 0
    print("Finished file in", time.time() - file_start_time, "\n")
print("Finished " + str(len(file_list)) + " files in: " + str(time.time() - total_start_time))


# ## Test hdf5 file data format & visualizations

# In[244]:


# visualize_tile(pos_examples[10])
# visualize_tile(examples[1])
# visualize_tile(examples[2])


# In[304]:


# WANTED_BANDS = [3, 2, 1]

# for i in range(5):
#     with h5py.File(save_path + "examples_" + str(i) + ".hdf5", "r") as f:
#         if i == 0:
#             X = np.array(f["images"][()])\
#                 .reshape((-1, len(all_bands), 64, 64))
#             X = np.moveaxis(X, 1, -1)[:, :, :, WANTED_BANDS]
#             y = np.array(f["labels"][()])
#         else:
#             x_i = np.array(f["images"][()])\
#                 .reshape((-1, len(all_bands), 64, 64))
#             x_i = np.moveaxis(x_i, 1, -1)[:, :, :, WANTED_BANDS]
#             X = np.concatenate((X, x_i))
#             y_i = np.array(f["labels"][()])
#             y = np.concatenate((y, y_i))


# In[315]:


# print("Examples w/ kilns:", np.where(y==1.0)[0])
# index = 4321
# print("label:", y[index])
# vis_X = X[index]
# vis_X *= 1 / np.max(vis_X)
# print(vis_X.shape)
# plt.imshow(vis_X)

