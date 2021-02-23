#!/usr/bin/env python
# coding: utf-8

# In[4]:


# !pip install pandas
# jupyter nbconvert --to script get_tiled_data_from_tiff_hdf5.ipynb


# In[93]:


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
from rasterio.windows import Window, WindowMethodsMixin, bounds as r_bounds


# In[6]:


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


# In[124]:


local_testing_mode = True

# set params
tile_height, tile_length = (64, 64)
examples_per_save_file = 1000
composite_file_name = 'bangladesh_all_bands_final'
download_all_first = not local_testing_mode
offset_px = 20
offset_configs = [(0, 0), (offset_px, 0), (0, offset_px), (offset_px, offset_px)]

save_path = '/atlas/u/mliu356/data/kiln-scaling/tiles_with_offsets_coords/'
composite_save_path = '/atlas/u/mliu356/data/kiln-scaling/composites/'

if local_testing_mode:
    save_path = '../data/tiles_testing1/'
    composite_save_path = '../data/composites/'

# resources
kilns = pd.read_csv("../data/bangladesh_kilns.csv")
all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B10', 'B11', 'B12']

print(kilns.head())


# In[8]:


def mkdirs(names):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)
mkdirs([save_path, composite_save_path])


# In[9]:


file_list = drive.ListFile({'q': "title contains '" + composite_file_name + "'"}).GetList()
print("Found " + str(len(file_list)) + " files")
file_list = sorted(file_list, key=lambda file: file['title'])
for file in file_list[:5]:
  print('title: %s, id: %s' % (file['title'], file['id']))


# In[10]:


# calculate image grid
first_x_coord = file_list[0]['title'].split(".")[0].split("-")[1]
first_y_coord = file_list[0]['title'].split(".")[0].split("-")[2]
num_image_cols = len([x for x in file_list if x['title'].split(".")[0].split("-")[1] == first_x_coord])
num_image_rows = len([x for x in file_list if x['title'].split(".")[0].split("-")[2] == first_y_coord])
print("Number of image grid columns:", num_image_cols)
print("Number of image grid rows:", num_image_rows)


# In[18]:


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


# In[11]:


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


# In[113]:


def get_tile_info_from_px(dataset, px_row, px_col):
    global num_tiles_dropped
    global pos_examples_data, pos_examples_bounds
    global test_ex_data, test_ex_bounds
    
    window = Window(px_col, px_row, tile_length, tile_height)
    transform = dataset.window_transform(window)
    profile = dataset.profile
    profile.update({
        'height': tile_length,
        'width': tile_height,
        'transform': transform})
    
    bands = dataset.read(window=window)
    bounds = list(r_bounds(window, dataset.transform))
    tile_geo = Polygon([[bounds[0], bounds[2]], [bounds[0], bounds[3]], [bounds[1], bounds[3]], [bounds[1], bounds[2]], [bounds[0], bounds[2]]])
    
    kilns_in_image = kilns.loc[(kilns['lat'] >= bounds[1]) & (kilns['lat'] <= bounds[3]) 
        & (kilns['lon'] >= bounds[0]) & (kilns['lon'] <= bounds[2])]
    if len(kilns_in_image) >= 1:
        print(kilns_in_image)
        pos_examples_data += [bands]
        pos_examples_bounds += [bounds]
        return bands, bounds, True
    elif not bangladesh_geo.intersects(tile_geo):
        num_tiles_dropped += 1
        return None, None, None
    else:
        return bands, bounds, False


# In[109]:


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


# In[55]:


## testing & visualization methods

# image is a single example of shape (13, 64, 64)
def visualize_tile(image, indices=[3, 2, 1]):
    row_idx = np.array(indices)
    X = np.transpose(image[row_idx], (1, 2, 0))
    X *= 1 / np.max(X)
    print(X.shape)
    plt.imshow(X)
    
def pretty_bounds(bounds):
    return [[bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]], [bounds[0], bounds[3]], [bounds[0], bounds[1]]]


# In[51]:


## testing variables
num_tiles_dropped = 0
pos_examples_data = []
pos_examples_bounds = []
test_ex_data = []
test_ex_bounds = []

save_index, counter = 0, 0

tile_bounds = np.zeros([examples_per_save_file, 4])
examples = np.zeros([examples_per_save_file, len(all_bands), tile_height, tile_length])
labels = np.zeros([examples_per_save_file, 1])


# In[114]:


if local_testing_mode:
    file_list = file_list[:1]

total_start_time = time.time()
for index, file in enumerate(file_list):
    file_start_time = time.time()
    print("Starting file " + file['title'])
    composite_file_path = composite_save_path + file['title']
    if not path.exists(composite_file_path):
        print("Downloading file...")
        # download the file
        download_file = drive.CreateFile({'id': file['id']})
        file.GetContentFile(composite_file_path)
        
    dataset = rasterio.open(composite_file_path)

    for offset_config in offset_configs:
        num_rows = int((dataset.height - offset_config[0]) / tile_height)
        num_cols = int((dataset.width - offset_config[1]) / tile_length)
        
        for tile_idx_row in range(0, num_rows):
            px_row = tile_idx_row * tile_height + offset_config[0]
            for tile_idx_col in range(0, num_cols):
                px_col = tile_idx_col * tile_length + offset_config[1]
                t_data, t_bounds, t_has_kiln = get_tile_info_from_px(dataset, px_row, px_col)
                if t_data is not None:
                    save_index, counter = add_example(t_data, t_bounds, save_index, counter, t_has_kiln)
                    
    # handle leftovers in a final file
    if index == len(file_list) - 1:
        save_current_file(save_index, counter)

    print("Total tiles dropped (outside country):", num_tiles_dropped)
    print("Total tiles kept:", str(num_rows * num_cols * len(offset_configs) - num_tiles_dropped))
    num_tiles_dropped = 0
    print("Finished file in", time.time() - file_start_time, "\n")
print("Finished " + str(len(file_list)) + " files in: " + str(time.time() - total_start_time))


# ## Test hdf5 file data format & visualizations

# In[83]:


print(pretty_bounds([dataset.bounds.left, dataset.bounds.bottom, dataset.bounds.right, dataset.bounds.top]))


# In[119]:


# print(len(pos_examples_bounds))
# vis_index = 2

# visualize_tile(pos_examples_data[vis_index])
# pretty_bounds(pos_examples_bounds[vis_index])


# In[27]:


# print("Examples w/ kilns:", np.where(labels==1.0)[0])
# index = 4321
# print("label:", y[index])
# vis_X = X[index]
# vis_X *= 1 / np.max(vis_X)
# print(vis_X.shape)
# plt.imshow(vis_X)

