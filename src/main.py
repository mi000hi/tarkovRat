import tkinter as tk
import pandas as pd
import pandas as pd
import numpy as np
import cv2

import pygetwindow as gw
import threading
import json

from PIL import Image

# from image_recognition import Item_predictor
# from overlay import Overlay, create_overlay, show_hide_labels, update, update_predictions, update_predictions_and_slots
# from tarkov import Tarkov_inventory, threaded_prediction

import image_recognition as ir
import overlay as ov
import tarkov as tr
import image_manipulation as im

# from overlay import *
# from image_manipulation import *
# from tarkov import *
# from image_manipulation import *

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
    window_tarkov = (window_tarkov_position[0],window_tarkov_position[1], window_tarkov_size[0], window_tarkov_size[1])

    window_scale_factor = window_tarkov_size[1] / 1080.0
    slot_size = (int) (64 * window_scale_factor) # in pixels, for FHD resolution

    
    # get needed data
    tarkov_inventory = tr.Tarkov_inventory(path_grid_icons=path_grid_icons, path_images=path_images, slot_size=slot_size, window_scale_factor=window_scale_factor, json_data=json_data)
    item_predictor = ir.Item_predictor(tarkov_inventory=tarkov_inventory, json_data=json_data)

    tarkov_inventory.STOP_THREADS = False


    # start prediction thread
    print("Start the prediction thread...")
    thread_predict = threading.Thread(target=tr.threaded_prediction, args=(tarkov_inventory, item_predictor,))
    thread_predict.start()

    # # get items from screenshot
    # print("Predict items on current screen...")
    # predict_current_inventory(predictions_df, True, window_tarkov)

    
    overlay = ov.Overlay(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, window_tarkov_position=window_tarkov_position, window_tarkov_size=window_tarkov_size)



    # run the update process
    print("Start updating the overlay... -- Use ESCAPE to kill the program")
    key = json_data.get("key_manual_predict")
    print(f"Use {key} to predict slot under mouse.")
    ov.update(tarkov_inventory=tarkov_inventory, overlay=overlay, item_predictor=item_predictor)

    # show the window and take focus
    overlay.root.focus_force()
    overlay.root.bind(json_data.get("key_program_quit"), lambda e: overlay.root.destroy())
    overlay.root.mainloop()

    # stop the update thread
    tarkov_inventory.STOP_THREADS = True
    thread_predict.join()
    print("Update thread stopped -- End of program")

if __name__ == "__main__":
    main()