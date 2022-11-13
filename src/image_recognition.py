import numpy as np
import pandas as pd
import cv2

import pyautogui
import random

from PIL import Image

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from image_manipulation import scale_image
#     from overlay import place_label
#     from tarkov import format_price_for_label, get_price_per_slot

import image_manipulation as im
import overlay as ov
import tarkov as tr

class Item_predictor():
    'this class predicts items'

    path_images = ''
    filename_empty_slot_img = 'slot_empty.png'

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # parameters
    item_images_updated = False
    predictions_updated = False
    nr_valid_predictions = 0

    # variables
    descriptors = []
    descriptors_values = []
    img_slot_gray = None
    predictions_df = None
    price_labels = None
    item_images = None
    predictions = None
    distances = None
    slot_locations = None

    json_data = None

    # classes
    tarkov_inventory = None

    def __init__(self, tarkov_inventory, json_data):

        self.tarkov_inventory = tarkov_inventory
        self.json_data = json_data
        self.path_images = tarkov_inventory.path_images

         # create item image descriptors
        print("Create image descriptors for items...")
        update_all_descriptors(image_predictor=self, tarkov_inventory=tarkov_inventory)

        # load and filter slot reference image
        img_slot = Image.open(self.path_images + self.filename_empty_slot_img)
        img_slot = np.asarray(img_slot)
        img_slot = im.scale_image(img_slot, scale_factor=tarkov_inventory.window_scale_factor)
        self.img_slot_gray = cv2.cvtColor(img_slot, cv2.COLOR_BGR2GRAY)

        # initialize prediction arrays
        self.predictions_df = pd.DataFrame({'slot_x': [0] * json_data.get("max_items_to_predict"), 'slot_y': [0] * json_data.get("max_items_to_predict")
                , 'predicted_item': [-1] * json_data.get("max_items_to_predict"), 'distance': [0] * json_data.get("max_items_to_predict")})
        self.price_labels = [None] * json_data.get("max_items_to_predict")
        self.item_images = [None] * json_data.get("max_items_to_predict")
        self.predictions = [-1] * json_data.get("max_items_to_predict")
        self.distances = [0] * json_data.get("max_items_to_predict")
        self.slot_locations = [(0,0)] * json_data.get("max_items_to_predict")




def run_sift(image_predictor, img, nr_corners=100, nr_selected_corners=999999, auto_threshold=True, verbose=False):
    nr_selected_corners = min(nr_selected_corners, nr_corners)
    threshold_step = 2
    threshold = -threshold_step

    # adjust threshold to get consistent number of corners
    kp = []
    while len(kp) > nr_corners or len(kp) == 0:
        threshold += 2
        fast = cv2.FastFeatureDetector_create(threshold=threshold)
        kp = fast.detect(img, None)

        if len(kp) == 0:
            return [],None
        if not auto_threshold:
            break

    # histogram for empty slot detection
    hist_x = [0] * (img.shape[1])
    hist_y = [0] * (img.shape[0])
    empty_slot_threshold = image_predictor.json_data.get("fast_empty_slot_threshold_factor")*len(kp)
    for k in kp:
        hist_x[(int) (k.pt[0])] += 1
        hist_y[(int) (k.pt[1])] += 1
    if max(hist_x) > empty_slot_threshold or max(hist_y) > empty_slot_threshold:
        if verbose:
            print("I think this is an empty slot...")
        return [],None

    # randomly select kps to get them uniformly distributed
    kp_selected = kp
    fir_factor = image_predictor.json_data.get("fast_fir_factor")
    if nr_selected_corners < nr_corners:
        kp_selected = []
        for i in range(nr_selected_corners):
            rand_kp = kp[random.randint(0,len(kp)-1)]

            # filter kp from fir symbol
            if rand_kp.pt[0] > fir_factor*image_predictor.tarkov_inventory.slot_size and rand_kp.pt[1] > fir_factor*image_predictor.tarkov_inventory.slot_size:
                continue
            kp_selected.append(rand_kp)

    # compute descriptors
    kp,des = image_predictor.sift.compute(img, kp_selected)
    return kp,des

def update_all_descriptors(image_predictor, tarkov_inventory):
    image_predictor.descriptors.clear()
    image_predictor.descriptors_values.clear()
    descriptors_values_length = 0
    for i,icon in enumerate(tarkov_inventory.icons):
        # no item icon
        if len(icon) == 0:
            image_predictor.descriptors.append(None)
            continue
        image_predictor.descriptors.append([])

        # descriptors for each item slot
        w = tarkov_inventory.all_items_df.loc[i,'width']
        h = tarkov_inventory.all_items_df.loc[i,'height']
        for x in range(w):
            for y in range(h):
                icon_slot = icon[y*tarkov_inventory.slot_size:(y+1)*tarkov_inventory.slot_size, x*tarkov_inventory.slot_size:(x+1)*tarkov_inventory.slot_size].copy()
                # TODO: may be bad idea to not use autothreshold
                kp,des = run_sift(image_predictor, icon_slot, nr_corners=tarkov_inventory.json_data.get("fast_all_item_nr_corners"), nr_selected_corners=tarkov_inventory.json_data.get("fast_all_item_nr_selected_corners"), auto_threshold=tarkov_inventory.json_data.get("fast_all_item_auto_threshold"))

                image_predictor.descriptors[i].append(descriptors_values_length)
                image_predictor.descriptors_values.append(des)
                descriptors_values_length += 1

def predict_icon(tarkov_inventory, item_predictor, img, improved=False, matching_item=None, verbose=False):
    # global icons, bf, descriptors, descriptors_values
    # global fast_predict_improved_nr_corners,fast_predict_improved_nr_selected_corners
    # global fast_predict_nr_corners,fast_predict_nr_selected_corners
    # global predict_max_nr_slots,predict_min_nr_matched_features

    distances = []
    distances_local = []
    kp,des = None,None
    if improved:
        # predict using template matching
        matches = []
        for icon in tarkov_inventory.icons:
            if icon is None or len(icon) == 0:
                matches.append(0)
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(img, icon[0:tarkov_inventory.slot_size,0:tarkov_inventory.slot_size], cv2.TM_CCOEFF_NORMED))
            matches.append(max_val)
        return np.argmax(matches),1/max(matches)

        # kp,des = run_sift(img, nr_corners=fast_predict_improved_nr_corners, nr_selected_corners=fast_predict_improved_nr_selected_corners)
    else:
        kp,des = run_sift(item_predictor, img, nr_corners=tarkov_inventory.json_data.get("fast_predict_nr_corners"), nr_selected_corners=tarkov_inventory.json_data.get("fast_predict_nr_selected_corners"))

    if verbose:
        print(f"nr kp: {len(kp)}")

    if not matching_item is None:
        distances_local = []
        max_slots = min(tarkov_inventory.json_data.get("predict_max_nr_slots"), len(item_predictor.descriptors[matching_item])) # max slot size of considered items
        for j in range(max_slots):
            # no kp / des found for specific slot
            if item_predictor.descriptors_values[item_predictor.descriptors[matching_item][j]] is None:
                distances_local.append(999999)
                continue

            matches = item_predictor.bf.match(des,item_predictor.descriptors_values[item_predictor.descriptors[matching_item][j]])
            matches = sorted(matches, key = lambda x:x.distance)

            # if too few features, abort
            if len(matches) < tarkov_inventory.json_data.get("predict_min_nr_matched_features"):
                distances_local.append(999999)
                continue

            # sum up match distances
            distance = 0.0
            for match in matches:
                distance += match.distance
            distance = distance/len(matches)

            distances_local.append(distance)

        # return distance of best matching slot
        min_distance = min(distances_local)
        return matching_item,min_distance

    for i in range(len(tarkov_inventory.icons)):
        # no kp / des found
        if item_predictor.descriptors[i] is None:
            distances.append(999999)
            continue

        distances_local = []
        max_slots = min(tarkov_inventory.json_data.get("predict_max_nr_slots"), len(item_predictor.descriptors[i])) # max slot size of considered items
        for j in range(max_slots):
            # no kp / des found for specific slot
            if item_predictor.descriptors_values[item_predictor.descriptors[i][j]] is None:
                distances_local.append(999999)
                continue

            matches = item_predictor.bf.match(des,item_predictor.descriptors_values[item_predictor.descriptors[i][j]])
            matches = sorted(matches, key = lambda x:x.distance)

            # if too few features, abort
            if len(matches) < tarkov_inventory.json_data.get("predict_min_nr_matched_features"):
                distances_local.append(999999)
                continue

            # sum up match distances
            distance = 0.0
            for match in matches:
                distance += match.distance
            distance = distance/len(matches)

            distances_local.append(distance)

        # return distance of best matching slot
        min_distance = min(distances_local)
        distances.append(min_distance)

    # item with lowest distance is prediction
    prediction = np.argmin(distances)
    min_distance = min(distances)

    if verbose:
        print(f"distance of prediction item: {min_distance}")

    return prediction,min_distance

def predict_all_icons(tarkov_inventory, item_predictor, images, predictions, distances, verbose=False):
    for i,item in enumerate(images):
        if verbose and i%10 == 0:
            print(f"predicting item nr {i}")
        p,d = predict_icon(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, img=item)
        predictions.append(p)
        distances.append(d)

def get_image_around_mouse(tarkov_inventory, position):
    # global slot_size
    # global slot_size_around_mouse_factor

    size =  tarkov_inventory.slot_size * tarkov_inventory.json_data.get("slot_size_around_mouse_factor")
    x = (int) (position[0] - 0.5*size)
    y = (int) (position[1] - 0.5*size)
    screenshot = pyautogui.screenshot(region=(x,y,size,size))
    screenshot = np.array(screenshot)
    return screenshot

def predict_item_under_mouse(tarkov_inventory, item_predictor, overlay):
    # global item_images, max_items_to_predict, item_images_updated
    # global slot_locations, nr_valid_predictions, slot_size
    # global font_label_manual_item

    mouse_pos = pyautogui.position()
    item = get_image_around_mouse(tarkov_inventory, mouse_pos)
    prediction,distance = predict_icon(tarkov_inventory, item_predictor, item, improved=True)

    # find prices for predicted item
    price_max,trader = tr.get_price_per_slot(tarkov_inventory, prediction)

    # format price to string
    text = tr.format_price_for_label(tarkov_inventory, prediction, price_max, trader)

    if overlay.labels_visible:
        ov.place_label(overlay, item_predictor, text, mouse_pos[0], mouse_pos[1], -1, font_color='red', font=overlay.font_label_manual_item, anchor='center')