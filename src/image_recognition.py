import numpy as np
import cv2

import pyautogui
import random



def run_sift(img, nr_corners=100, nr_selected_corners=999999, auto_threshold=True, verbose=False):
    global sift
    global fast_empty_slot_threshold_factor,fast_fir_factor

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
    empty_slot_threshold = fast_empty_slot_threshold_factor*len(kp)
    for k in kp:
        hist_x[(int) (k.pt[0])] += 1
        hist_y[(int) (k.pt[1])] += 1
    if max(hist_x) > empty_slot_threshold or max(hist_y) > empty_slot_threshold:
        if verbose:
            print("I think this is an empty slot...")
        return [],None

    # randomly select kps to get them uniformly distributed
    kp_selected = kp
    fir_factor = fast_fir_factor
    if nr_selected_corners < nr_corners:
        kp_selected = []
        for i in range(nr_selected_corners):
            rand_kp = kp[random.randint(0,len(kp)-1)]

            # filter kp from fir symbol
            if rand_kp.pt[0] > fir_factor*slot_size and rand_kp.pt[1] > fir_factor*slot_size:
                continue
            kp_selected.append(rand_kp)

    # compute descriptors
    kp,des = sift.compute(img, kp_selected)
    return kp,des

def create_all_descriptors():
    global icons, all_items_df, slot_size
    global fast_all_item_nr_corners,fast_all_item_nr_selected_corners,fast_all_item_auto_threshold

    descriptors = []
    descriptors_values = []
    descriptors_values_length = 0
    for i,icon in enumerate(icons):
        # no item icon
        if len(icon) == 0:
            descriptors.append(None)
            continue
        descriptors.append([])

        # descriptors for each item slot
        w = all_items_df.loc[i,'width']
        h = all_items_df.loc[i,'height']
        for x in range(w):
            for y in range(h):
                icon_slot = icon[y*slot_size:(y+1)*slot_size, x*slot_size:(x+1)*slot_size].copy()
                # TODO: may be bad idea to not use autothreshold
                kp,des = run_sift(icon_slot, nr_corners=fast_all_item_nr_corners, nr_selected_corners=fast_all_item_nr_selected_corners, auto_threshold=fast_all_item_auto_threshold)

                descriptors[i].append(descriptors_values_length)
                descriptors_values.append(des)
                descriptors_values_length += 1

    return descriptors, descriptors_values

def predict_icon(img, improved=False, matching_item=None, verbose=False):
    global icons, bf, descriptors, descriptors_values
    global fast_predict_improved_nr_corners,fast_predict_improved_nr_selected_corners
    global fast_predict_nr_corners,fast_predict_nr_selected_corners
    global predict_max_nr_slots,predict_min_nr_matched_features

    distances = []
    distances_local = []
    kp,des = None,None
    if improved:
        # predict using template matching
        matches = []
        for icon in icons:
            if icon is None or len(icon) == 0:
                matches.append(0)
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(img, icon[0:slot_size,0:slot_size], cv2.TM_CCOEFF_NORMED))
            matches.append(max_val)
        return np.argmax(matches),1/max(matches)

        # kp,des = run_sift(img, nr_corners=fast_predict_improved_nr_corners, nr_selected_corners=fast_predict_improved_nr_selected_corners)
    else:
        kp,des = run_sift(img, nr_corners=fast_predict_nr_corners, nr_selected_corners=fast_predict_nr_selected_corners)

    if verbose:
        print(f"nr kp: {len(kp)}")

    if not matching_item is None:
        distances_local = []
        max_slots = min(predict_max_nr_slots, len(descriptors[matching_item])) # max slot size of considered items
        for j in range(max_slots):
            # no kp / des found for specific slot
            if descriptors_values[descriptors[matching_item][j]] is None:
                distances_local.append(999999)
                continue

            matches = bf.match(des,descriptors_values[descriptors[matching_item][j]])
            matches = sorted(matches, key = lambda x:x.distance)

            # if too few features, abort
            if len(matches) < predict_min_nr_matched_features:
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

    for i in range(len(icons)):
        # no kp / des found
        if descriptors[i] is None:
            distances.append(999999)
            continue

        distances_local = []
        max_slots = min(predict_max_nr_slots, len(descriptors[i])) # max slot size of considered items
        for j in range(max_slots):
            # no kp / des found for specific slot
            if descriptors_values[descriptors[i][j]] is None:
                distances_local.append(999999)
                continue

            matches = bf.match(des,descriptors_values[descriptors[i][j]])
            matches = sorted(matches, key = lambda x:x.distance)

            # if too few features, abort
            if len(matches) < predict_min_nr_matched_features:
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

def predict_all_icons(images, predictions, distances, verbose=False):
    for i,item in enumerate(images):
        if verbose and i%10 == 0:
            print(f"predicting item nr {i}")
        p,d = predict_icon(item)
        predictions.append(p)
        distances.append(d)

def get_image_around_mouse(position):
    global slot_size
    global slot_size_around_mouse_factor

    size =  slot_size * slot_size_around_mouse_factor
    x = (int) (position[0] - 0.5*size)
    y = (int) (position[1] - 0.5*size)
    screenshot = pyautogui.screenshot(region=(x,y,size,size))
    screenshot = np.array(screenshot)
    return screenshot

def predict_item_under_mouse():
    global item_images, max_items_to_predict, item_images_updated
    global slot_locations, nr_valid_predictions, slot_size
    global font_label_manual_item

    mouse_pos = pyautogui.position()
    item = get_image_around_mouse(mouse_pos)
    prediction,distance = predict_icon(item, improved=True)

    # find prices for predicted item
    price_max,trader = get_price_per_slot(prediction)

    # format price to string
    text = format_price_for_label(prediction, price_max, trader)

    if labels_visible:
        place_label(text, mouse_pos[0], mouse_pos[1], -1, font_color='red', font=font_label_manual_item, anchor='center')