import tkinter as tk
import pandas as pd
import pandas as pd
import numpy as np
import cv2

import pygetwindow as gw
import threading
import json

from PIL import Image

from overlay import *
from image_manipulation import *
from tarkov import *
from image_manipulation import *



def update_json_variables(file=None):
    global slot_locations_min_x,slot_locations_min_y,slot_locations_max_x,slot_locations_max_y,slot_locations_threshold,slot_locations_min_distance,tax_t_i,tax_t_r,tax_quantity,tax_quantity_factor,fast_nr_corners,fast_threshold_step,fast_empty_slot_threshold_factor,fast_fir_factor,fast_all_item_nr_corners,fast_all_item_nr_selected_corners,fast_all_item_auto_threshold,fast_predict_improved_nr_corners,fast_predict_improved_nr_selected_corners,fast_predict_nr_corners,fast_predict_nr_selected_corners,predict_max_nr_slots,predict_min_nr_matched_features,slot_size_around_mouse_factor,items_flea_market_update_interval_mins,thread_prediction_sleep_interval_secs,overlay_update_interval_msecs,label_y_offset,key_manual_predict,max_items_to_predict,predictions_threshold,window_title_tarkov,font_label_item_size,font_label_manual_item_size,overlay_border_size,overlay_transparent_color
    global overlay_nr_smallest_prices,overlay_label_cheap_items_color,currency_dollar_to_rubles_factor,overlay_label_show_trader
    global key_program_quit

    if file is None:
        file = config_file

    config = open(file)
    data = json.load(config)
    data_list = list(data.items())

    slot_locations_min_x = data_list[0][1]
    slot_locations_min_y = data_list[1][1]
    slot_locations_max_x = data_list[2][1]
    slot_locations_max_y = data_list[3][1]
    slot_locations_threshold = data_list[4][1]
    slot_locations_min_distance = data_list[5][1]
    tax_t_i = data_list[6][1]
    tax_t_r = data_list[7][1]
    tax_quantity = data_list[8][1]
    tax_quantity_factor = data_list[9][1]
    fast_nr_corners = data_list[10][1]
    fast_threshold_step = data_list[11][1]
    fast_empty_slot_threshold_factor = data_list[12][1]
    fast_fir_factor = data_list[13][1]
    fast_all_item_nr_corners = data_list[14][1]
    fast_all_item_nr_selected_corners = data_list[15][1]
    fast_all_item_auto_threshold = data_list[16][1]
    fast_predict_improved_nr_corners = data_list[17][1]
    fast_predict_improved_nr_selected_corners = data_list[18][1]
    fast_predict_nr_corners = data_list[19][1]
    fast_predict_nr_selected_corners = data_list[20][1]
    predict_max_nr_slots = data_list[21][1]
    predict_min_nr_matched_features = data_list[22][1]
    slot_size_around_mouse_factor = data_list[23][1]
    items_flea_market_update_interval_mins = data_list[24][1]
    thread_prediction_sleep_interval_secs = data_list[25][1]
    overlay_update_interval_msecs = data_list[26][1]
    label_y_offset = data_list[27][1]
    key_manual_predict = data_list[28][1]
    max_items_to_predict = data_list[29][1]
    predictions_threshold = data_list[30][1]
    window_title_tarkov = data_list[31][1]
    font_label_item_size = data_list[32][1]
    font_label_manual_item_size = data_list[33][1]
    overlay_border_size = data_list[34][1]
    overlay_transparent_color = data_list[35][1]
    overlay_nr_smallest_prices = data_list[36][1]
    overlay_label_cheap_items_color = data_list[37][1]
    overlay_label_show_trader = data_list[38][1]
    currency_dollar_to_rubles_factor = data_list[39][1]
    key_program_quit = data_list[40][1]

    config.close()



############################################################
# MAIN PROGRAM                                             #
############################################################

# if __name__ == "__main__":
#     main()

# def main():
#     return


print("Program Starts -- Be patient")

# paths
path_icons = './icons/'
path_images = './images/'
path_grid_icons = './grid_icons/'
path_data = './data/'
filename_ending_grid_icon = '-grid-image.jpg'
config_file = './config.jsonc'


icons = []
descriptors = []
descriptors_values = []
item_images = []
predictions = []
distances = []
slot_locations = []
price_labels = []
all_items_df = None
predictions_df = None

img_slot_gray = None
root = None

# variables used in JSON config file
slot_locations_min_x = 0
slot_locations_min_y = 0
slot_locations_max_x = 0
slot_locations_max_y = 0
slot_locations_threshold = 0
slot_locations_min_distance = 0
tax_t_i = 0
tax_t_r = 0
tax_quantity = 0
tax_quantity_factor = 0
fast_nr_corners = 0
fast_threshold_step = 0
fast_empty_slot_threshold_factor = 0
fast_fir_factor = 0
fast_all_item_nr_corners = 0
fast_all_item_nr_selected_corners = 0
fast_all_item_auto_threshold = False
fast_predict_improved_nr_corners = 0
fast_predict_improved_nr_selected_corners = 0
fast_predict_nr_corners = 0
fast_predict_nr_selected_corners = 0
predict_max_nr_slots = 0
predict_min_nr_matched_features = 0
slot_size_around_mouse_factor = 0
items_flea_market_update_interval_mins = 0
thread_prediction_sleep_interval_secs = 0
overlay_update_interval_msecs = 0
label_y_offset = 0
key_manual_predict = ""
max_items_to_predict = 0
predictions_threshold = 0
window_title_tarkov = ""
font_label_item_size = 0
font_label_manual_item_size = 0
overlay_border_size = 0
overlay_transparent_color = ""
overlay_nr_smallest_prices = 0
overlay_label_cheap_items_color = ""
overlay_label_show_trader = False
currency_dollar_to_rubles_factor = 0
key_program_quit = ""

update_json_variables(config_file)

# initializing other variables
labels_visible = False
item_images_updated = False
predictions_updated = False
STOP_THREADS = False
thread_predict = threading.Thread(target=threaded_prediction, args=(item_images,))
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
window_scale_factor = 1
last_api_update = 0
nr_valid_predictions = 0

# get tarkov window
print("Looking for the EFT window...")
window_tarkov = gw.getWindowsWithTitle(window_title_tarkov)[0]
window_tarkov_position = window_tarkov.topleft
window_tarkov_size = window_tarkov.size

window_scale_factor = window_tarkov_size[1] / 1080.0
slot_size = (int) (64 * window_scale_factor) # in pixels, for FHD resolution

font_label_item = ("helvetica", font_label_item_size)
font_label_manual_item = ("helvetica", font_label_manual_item_size)

# get needed data
print("Load item information...")
all_items_df = items_dict_to_df(getAllItemsPrices())
print("Load item icons from disk...")
icons = load_icons_from_disk()
print("Create image descriptors for items...")
descriptors,descriptors_values = create_all_descriptors()

# load and filter slot reference image
img_slot = Image.open(path_images + 'slot_empty.png')
img_slot = np.asarray(img_slot)
img_slot = scale_image(img_slot, scale_factor=window_scale_factor)
img_slot_gray = cv2.cvtColor(img_slot, cv2.COLOR_BGR2GRAY)

# initialize arrays
predictions_df = pd.DataFrame({'slot_x': [0] * max_items_to_predict, 'slot_y': [0] * max_items_to_predict
        , 'predicted_item': [-1] * max_items_to_predict, 'distance': [0] * max_items_to_predict})
price_labels = [None] * max_items_to_predict
item_images = [None] * max_items_to_predict
predictions = [-1] * max_items_to_predict
distances = [0] * max_items_to_predict
slot_locations = [(0,0)] * max_items_to_predict

# start prediction thread
print("Start the prediction thread...")
thread_predict.start()

# get items from screenshot
print("Predict items on current screen...")
predict_current_inventory(predictions_df, True)

# find overlay position
x_window = window_tarkov_position[0]
y_window = window_tarkov_position[1]
window_width = window_tarkov_size[0]
window_height = window_tarkov_size[1]

# create the window
print("Create the overlay...")
root = create_overlay()

# create a transparent frame to make a border
my_frame = tk.Frame(root, width=window_width-2*overlay_border_size, height=window_height-2*overlay_border_size, bg=overlay_transparent_color)
my_frame.place(x=overlay_border_size,y=overlay_border_size)

# add a button
button1 = tk.Button(root, text='Update predictions', fg='blue', command=update_predictions)
button1.grid(row=0,column=0)
button2_text = ''
if labels_visible:
    button2_text = 'hide labels'
else:
    button2_text = 'show labels'
button2 = tk.Button(root, text=button2_text, fg='blue', command=show_hide_labels)
button2.grid(row=0,column=1)
button3 = tk.Button(root, text='read JSON file', fg='blue', command=update_json_variables)
button3.grid(row=0,column=2)
button4 = tk.Button(root, text='Update predictions and slots', fg='blue', command=update_predictions_and_slots)
button4.grid(row=0, column=3)

# run the update process
print("Start updating the overlay... -- Use ESCAPE to kill the program")
print(f"Use {key_manual_predict} to predict slot under mouse.")
update()

# show the window and take focus
root.focus_force()
root.bind(key_program_quit, lambda e: root.destroy())
root.mainloop()

# stop the update thread
STOP_THREADS = True
thread_predict.join()
print("Update thread stopped -- End of program")
