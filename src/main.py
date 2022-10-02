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

### Global variables
config_file = './config.jsonc'



def update_json_variables(file=None):

    if file is None:
        global config_file
        file = config_file

    config = open(file)
    data = json.load(config)

    config.close()
    return data



############################################################
# MAIN PROGRAM                                             #
############################################################

def main():

    print("Program Starts -- Be patient")

    # paths
    path_icons = './icons/'
    path_images = './images/'
    path_grid_icons = './grid_icons/'
    path_data = './data/'
    filename_ending_grid_icon = '-grid-image.jpg'
    


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
    json_data = None

    global config_file
    json_data = update_json_variables(config_file)

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
    window_tarkov = gw.getWindowsWithTitle(json_data.get("window_title_tarkov"))[0]
    window_tarkov_position = window_tarkov.topleft
    window_tarkov_size = window_tarkov.size

    window_scale_factor = window_tarkov_size[1] / 1080.0
    slot_size = (int) (64 * window_scale_factor) # in pixels, for FHD resolution

    font_label_item = ("helvetica", json_data.get("font_label_item_size"))
    font_label_manual_item = ("helvetica", json_data.get("font_label_manual_item_size"))

    # get needed data
    print("Load item information...")
    all_items = getAllItemsPrices()
    all_items_df = pd.DataFrame(columns=['name', 'id', 'width', 'height', 'icon_width', 'icon_height', 'features', 'fleaMarket', 'flea_avg48', 'flea_ch48percent', 'prapor', 'therapist', 'fence', 'skier', 'peacekeeper', 'mechanic', 'ragman', 'jaeger', 'basePrice'], index=range(len(all_items)))
    update_items_df_from_dict(all_items_df, all_items)

    print("Load item icons from disk...")
    icons = load_icons_from_disk(all_items_df, path_grid_icons, filename_ending_grid_icon, slot_size)
    
    print("Create image descriptors for items...")
    descriptors,descriptors_values = update_all_descriptors(all_items_df, descriptors, descriptors_values, icons, slot_size, json_data)

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
    my_frame = tk.Frame(root, width=window_width-2*json_data.get("overlay_border_size"), height=window_height-2*json_data.get("overlay_border_size"), bg=overlay_transparent_color)
    my_frame.place(x=json_data.get("overlay_border_size"),y=json_data.get("overlay_border_size"))

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
    root.bind(json_data.get("key_program_quit"), lambda e: root.destroy())
    root.mainloop()

    # stop the update thread
    STOP_THREADS = True
    thread_predict.join()
    print("Update thread stopped -- End of program")

if __name__ == "__main__":
    main()