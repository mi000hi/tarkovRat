import tkinter as tk
import pandas as pd
import math
import pandas as pd
import numpy as np
import cv2

import pygetwindow as gw
import pyautogui
import threading
import time
import requests
import random

from PIL import Image, ImageStat, ImageTk
from matplotlib import pyplot as plt
from os.path import exists




### INVENTORY ITEM DETECTION ###

def edge_detection(img_np):
    img = img_np.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.Canny(img, threshold1=50, threshold2=200, apertureSize=5)
    img = cv2.blur(img, (3,3))
    return img

def inventory_line_detection(img):
    global window_scale_factor
    img_inventory_edge = edge_detection(img)
    
    # horizontal and vertical line detection
    line_length = (int) (40 * window_scale_factor)
    img_horizontal = img_inventory_edge.copy()
    img_vertical = img_inventory_edge.copy()

    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length,1))
    vertical_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1,line_length))
    cv2.erode(img_horizontal, horizontal_struct, img_horizontal)
    cv2.erode(img_vertical, vertical_struct, img_vertical)
    cv2.dilate(img_horizontal, horizontal_struct, img_horizontal)
    cv2.dilate(img_vertical, vertical_struct, img_vertical)

    result = cv2.add(img_horizontal, img_vertical)
    return result

def find_slot_locations(inventory_filtered, slot_gray, min_x = 1000):
    global slot_locations, nr_valid_predictions
    global window_scale_factor

    slot_x_offset = min_x
    slot_y_offset = 100
    slot_x_max = 2900
    slot_y_max = 1300
    inv = inventory_filtered[slot_y_offset:slot_y_max, slot_x_offset:slot_x_max].copy()
    slot = slot_gray.copy()

    matched_slots = cv2.matchTemplate(inv, slot, cv2.TM_CCORR_NORMED)
    
    threshold = 0.6
    min_distance = 60 * window_scale_factor
    
    # find slots in filtered inventory image
    slots = []
    for y in range(matched_slots.shape[0]):
        for x in range(matched_slots.shape[1]):
            if matched_slots[y][x] > threshold:
                slots.append((x,y))
    
    # filter number of locations
    valid_slots = 0
    for i,new_slot in enumerate(slots):
        if valid_slots >= len(slot_locations):
            print(f"Reached maximum amount of valid slots: {valid_slots}")
            break
        add_slot = True
        x,y = new_slot
        for j in range(valid_slots):
            s = slot_locations[j]
            if abs(s[0]-x) < min_distance and abs(s[1]-y) < min_distance:
                add_slot = False
                break
        if add_slot == True:
            slot_locations[valid_slots] = (x,y)
            valid_slots += 1

    # add offset to slot locations
    for i in range(len(slot_locations)):
        x,y = slot_locations[i]
        slot_locations[i] = (x+slot_x_offset, y+slot_y_offset)
    
    nr_valid_predictions = valid_slots

def draw_slots_on_image(img, slots):
    img_with_slots = img.copy()
    for x,y in slots:
        img_with_slots = cv2.circle(img_with_slots, (x, y), radius=5, color=(255,0,0), thickness=-1)
    return img_with_slots

def get_item_images_from_inventory(inventory):
    global item_images, max_items_to_predict, item_images_updated
    global slot_locations, nr_valid_predictions, slot_size
    
    for i in range(nr_valid_predictions):
        s = slot_locations[i]
        item_images[i] = inventory[s[1]:s[1]+slot_size, s[0]:s[0]+slot_size]
        
    item_images_updated = True

def scale_image(image, scale_factor=2, width=-1, height=-1):
    if width == -1 or height == -1:
        width = (int) (image.shape[1]*scale_factor)
        height = (int) (image.shape[0]*scale_factor)
    resized = cv2.resize(image, (width,height))
    return resized




### ITEM PREDICTION ###

def tax(itemindex, price_rubels):
    global all_items_df

    if price_rubels == 0:
        return 0

    t_i = 0.05
    t_r = 0.1
    baseprice = all_items_df.loc[itemindex, 'basePrice']
    quantity = 1
    quantity_factor = 1
    v_o = baseprice * quantity / quantity_factor
    if v_o == 0:
        return 0
    v_r = price_rubels # value of requirements
    p_o = np.log10(v_o / v_r)
    p_r = np.log10(v_r / v_o)

    tax = v_o * t_i * (4**p_o) * quantity_factor + v_r * t_r * (4**p_r) * quantity_factor
    return tax

def load_icons_from_disk(verbose=False):
    global all_items_df
    global path_grid_icons,filename_ending_grid_icon,window_scale_factor
    
    icons = []
    for index,item in all_items_df.iterrows():
        filename = path_grid_icons + item['id'] + filename_ending_grid_icon
        if not exists(filename):
            icons.append([])
            if verbose:
                print(f"File {filename} does not exist.")
        else:
            width_in_slots = item.loc['width']
            height_in_slots = item.loc['height']
            icon = np.asarray(Image.open(filename))
            icon = scale_image(icon, width=(int) (width_in_slots*slot_size*window_scale_factor)
                    , height=(int) (height_in_slots*slot_size*window_scale_factor))
            icons.append(icon)

    return icons

def run_sift(img, nr_corners=80, nr_selected_corners=999999, auto_threshold=True, verbose=False):
    global sift
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
    empty_slot_threshold = 0.2*len(kp)
    for k in kp:
        hist_x[(int) (k.pt[0])] += 1
        hist_y[(int) (k.pt[1])] += 1
    if max(hist_x) > empty_slot_threshold or max(hist_y) > empty_slot_threshold:
        if verbose:
            print("I think this is an empty slot...")
        return [],None
    
    # randomly select kps to get them uniformly distributed
    kp_selected = kp
    if nr_selected_corners < nr_corners:
        kp_selected = []      
        for i in range(nr_selected_corners):
            rand_kp = kp[random.randint(0,len(kp)-1)]

            # filter kp from fir symbol
            # TODO: dynamic fir symbol size
            if rand_kp.pt[0] > 50 and rand_kp.pt[1] > 50:
                continue
            kp_selected.append(rand_kp)
    
    # compute descriptors
    kp,des = sift.compute(img, kp_selected)
    return kp,des

def create_all_descriptors():
    global icons, all_items_df, slot_size

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
                kp,des = run_sift(icon_slot, nr_corners=50)

                descriptors[i].append(descriptors_values_length)
                descriptors_values.append(des)
                descriptors_values_length += 1

    return descriptors, descriptors_values

def predict_icon(img, verbose=False):
    global icons, bf, descriptors, descriptors_values

    distances = []
    distances_local = []
    kp,des = run_sift(img, nr_corners=50, nr_selected_corners=10)

    if verbose:
        print(f"nr kp: {len(kp)}")

    for i in range(len(icons)):
        # no kp / des found
        if descriptors[i] is None:
            distances.append(999999)
            continue

        distances_local = []
        max_slots = min(6, len(descriptors[i])) # max slot size of considered items
        for j in range(max_slots):
            # no kp / des found for specific slot
            if descriptors_values[descriptors[i][j]] is None:
                distances_local.append(999999)
                continue

            matches = bf.match(des,descriptors_values[descriptors[i][j]])
            matches = sorted(matches, key = lambda x:x.distance)
            
            # if too few features, abort
            if len(matches) < 5:
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





def predict_current_inventory(predictions_df):
    global root
    
    # hide all tkinter
    if not root is None:
        root.withdraw()
    
    # get inventory
    screenshot = pyautogui.screenshot(region=(window_tarkov_position[0],window_tarkov_position[1]
                                              , window_tarkov_size[0], window_tarkov_size[1]))
    screenshot = np.array(screenshot)
    
    # show all tkinter again
    if not root is None:
        root.deiconify()
    
    # get items from screenshot
    get_predictions_from_inventory(screenshot)

def threaded_prediction(items, verbose=False):
    global item_images_updated, nr_valid_predictions, predictions_updated
    global predictions_df, slot_locations
    global predictions, distances, prediction_threshold
    global last_api_update, all_items_df
    
    while True:
        global STOP_THREADS
        if STOP_THREADS:
            break
        if not item_images_updated:
            # update item data every 10 mins
            now = time.time()
            if now - last_api_update < 10 * 60:
                print("Updating the API data.")
                all_items_df = update_items_df(getAllItemsPrices())
                last_api_update = now
            time.sleep(1)
            if verbose:
                print("Thread running...")

        else:
            print(f"we have {nr_valid_predictions} new predictions")
            for i in range(nr_valid_predictions):
                item = item_images[i]
                
                # update prediction information
                p,d = predict_icon(item)
                predictions[i] = p
                distances[i] = d

                # update prediction dataframe
                predictions_df.loc[i,'slot_x'] = slot_locations[i][0]
                predictions_df.loc[i,'slot_y'] = slot_locations[i][1]

                predictions_df.loc[i,'predicted_item'] = predictions[i]
                predictions_df.loc[i,'distance'] = distances[i]
                
                predictions_updated = True

            item_images_updated = False

def get_predictions_from_inventory(inventory):
    global img_slot_gray
    global thread_predict
    
    # get item images
    t0 = time.time()
    inventory_filtered = inventory_line_detection(inventory)
    find_slot_locations(inventory_filtered, img_slot_gray)
    get_item_images_from_inventory(inventory)
    t1 = time.time()
    print(f'inventory slots and images took {t1-t0} s')
    
    # predict each item from inventory
    ## thread is already doing this

    # TODO: can return be removed?
    # return predictions_df

def place_label(text, x, y, index):
    global root,font_label_item

    label = tk.Label(root, text=text, font=font_label_item, fg='white', bg='black')
    label.place(x=x, y=y+20, anchor = 'nw')
    
    # remove old label
    if index < len(price_labels):
        if not price_labels[index] is None:
            price_labels[index].destroy()
        price_labels[index] = label
    else:
        price_labels.append(label)
    
def create_overlay():
    root = tk.Tk()
    root.title("this is not a virus. be a chad")
    root.geometry("%dx%d+%d+%d" % (window_width,window_height,x_window,y_window))
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.configure(bg='red')

    # make color black transparent
    root.wm_attributes('-transparentcolor', 'black')
    return root

def update_price_labels():
    remove_price_labels()
    add_price_labels()

def add_price_labels():
    global nr_valid_predictions, predictions_df, price_labels, all_items_df
    global predictions_threshold
    
    for i in range(nr_valid_predictions):
        prediction = predictions_df.loc[i]
        prediction_index = prediction[2]
        prediction_distance = prediction[3]
        if prediction_index == -1 or prediction_distance > predictions_threshold:
            if not price_labels[i] is None:
                price_labels[i].destroy()
                price_labels[i] = None
            continue
    
        # label location
        x = prediction[0]
        y = prediction[1]

        # find prices for predicted item
        traders = {0:'prapor', 1:'therapist', 2:'fence', 3:'skier', 4:'peacekeeper', 5:'mechanic', 6:'ragman', 7:'jaeger'}
        price_flea = all_items_df.loc[prediction_index, 'fleaMarket']
        price_traders = all_items_df.loc[prediction_index, 'prapor':'jaeger']
        
        # find best trader
        price_traders_max = np.nanmax(price_traders.values.tolist() + [0])
        best_trader = 0
        if price_traders_max == 0:
            best_trader = '-1'
        else:
            best_trader = traders.get(np.nanargmax(price_traders))
        
        # subtract tax from flea price
        if math.isnan(price_flea):
            price_flea = 0
        else:
            flea_tax = tax(prediction_index, price_flea)
            price_flea -= flea_tax

        # check trader or flea
        price_max = max(price_traders_max, price_flea)
        flea_best = np.argmax([price_traders_max, price_flea])

        trader = best_trader
        if flea_best == 1:
            trader = 'flea'

        # no price available
        if math.isnan(price_max):
            continue

        # price per slot
        nr_slots = all_items_df.loc[prediction_index, 'width'] * all_items_df.loc[prediction_index, 'height']
        price_max = price_max/nr_slots
        # format price to string
        price = (int) (price_max/1000)
        price_string = str(price) + 'k'
        if price == 0:
            price = price_max
            price_string = str(price)
        name = all_items_df.loc[prediction_index][0]
        
        # create the new label
        text = name + '\n' + price_string + '\n' + trader
        place_label(text, x, y, i)
    
    # remove other old labels
    for i in range(nr_valid_predictions, len(price_labels)):
        if price_labels[i] is None:
            break
        price_labels[i].destroy()

def remove_price_labels():
    for label in price_labels:
        if not label is None:
            label.destroy()
        
def update():
    global root,canvas
    
    if labels_visible:
        update_price_labels()
    
    # run itself again after 100 ms
    root.after(200, update)
    
def update_predictions():
    global predictions_df
    predict_current_inventory(predictions_df)
    
def show_hide_labels():
    global price_labels, labels_visible
    
    if labels_visible:
        for i in range(len(price_labels)):
            if not price_labels[i] is None:
                price_labels[i].destroy()
                price_labels[i] = None
        labels_visible = False
    else:
        add_price_labels()
        labels_visible = True

def items_dict_to_df(all_items):
    all_items_df = pd.DataFrame(columns=['name', 'id', 'width', 'height', 'icon_width', 'icon_height', 'features', 'fleaMarket', 'flea_avg48', 'flea_ch48percent', 'prapor', 'therapist', 'fence', 'skier', 'peacekeeper', 'mechanic', 'ragman', 'jaeger', 'basePrice'], index=range(len(all_items)))

    # iterate over all items
    for index in range(len(all_items)):
        all_items_df.iloc[index]['name'] = all_items[index].get('shortName')
        all_items_df.iloc[index]['id'] = all_items[index].get('id')
        all_items_df.iloc[index]['width'] = all_items[index].get('width')
        all_items_df.iloc[index]['height'] = all_items[index].get('height')
        all_items_df.iloc[index]['flea_avg48'] = all_items[index].get('avg24hPrice')
        all_items_df.iloc[index]['flea_ch48percent'] = all_items[index].get('changeLast48hPercent')
        all_items_df.iloc[index]['basePrice'] = all_items[index].get('basePrice')

        # iterate over all traders that can buy the item
        for offer in all_items[index].get('sellFor'):
            trader = offer.get('source')
            price  = offer.get('price')
            all_items_df.iloc[index][trader] = price
    
    return all_items_df

def update_items_df(all_items_dict):
    global all_items_df

    # iterate over all items
    for index in range(len(all_items_dict)):
        if all_items_df.iloc[index]['id'] != all_items_dict[index].get('id'):
            print(f"The item {all_items_df.iloc[index]['name']} does not match the new updated information!")
            continue
        all_items_df.iloc[index]['flea_avg48'] = all_items_dict[index].get('avg24hPrice')
        all_items_df.iloc[index]['flea_ch48percent'] = all_items_dict[index].get('changeLast48hPercent')

        # iterate over all traders that can buy the item
        for offer in all_items_dict[index].get('sellFor'):
            trader = offer.get('source')
            price  = offer.get('price')
            all_items_df.iloc[index][trader] = price

def run_query(query):
    response = requests.post('https://api.tarkov.dev/graphql', json={'query': query})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(response.status_code, query))

def getAllItemsPrices():
    query_all_items_prices = """
    {
        items {
            shortName
            id
            width
            height
            avg24hPrice
            changeLast48hPercent
            basePrice
            sellFor {
                price
                source
            }
        }
    }
    """
    result = run_query(query_all_items_prices)
    return result['data']['items']





############################################################
# MAIN PROGRAM                                             #
############################################################


# paths
path_icons = './icons/'
path_images = './images/'
path_grid_icons = './grid_icons/'
path_data = './data/'

filename_ending_grid_icon = '-grid-image.jpg'
window_title_tarkov = 'EscapeFromTarkov'

icons = []
descriptors = []
descriptors_values = []
all_items_df = None
max_items_to_predict = 300
nr_valid_predictions = 0
predictions_threshold = 300
predictions_df = None
item_images = []
predictions = []
distances = []
slot_locations = []
img_slot_gray = None
root = None
price_labels = []
labels_visible = True
item_images_updated = False
predictions_updated = False
STOP_THREADS = False
thread_predict = threading.Thread(target=threaded_prediction, args=(item_images,))
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
window_scale_factor = 1
last_api_update = 0

# get tarkov window
window_tarkov = gw.getWindowsWithTitle(window_title_tarkov)[0]
window_tarkov_position = window_tarkov.topleft
window_tarkov_size = window_tarkov.size

window_scale_factor = window_tarkov_size[1] / 1080.0
slot_size = (int) (64 * window_scale_factor) # in pixels, for FHD resolution

font_label_item = ("helvetica", 10)
overlay_border_size = 5

# get needed data
all_items_df = items_dict_to_df(getAllItemsPrices())
icons = load_icons_from_disk()
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
thread_predict.start()

# get items from screenshot
predict_current_inventory(predictions_df)

# find overlay position
x_window = window_tarkov_position[0]
y_window = window_tarkov_position[1]
window_width = window_tarkov_size[0]
window_height = window_tarkov_size[1]

# create the window
root = create_overlay()

# create a transparent frame to make a border
my_frame = tk.Frame(root, width=window_width-2*overlay_border_size, height=window_height-2*overlay_border_size, bg='black')
my_frame.place(x=overlay_border_size,y=overlay_border_size)

# add a button
button1 = tk.Button(root, text='Update predictions', fg='blue', command=update_predictions)
button1.pack()
button2 = tk.Button(root, text='show/hide labels', fg='blue', command=show_hide_labels)
button2.pack()
labels_visible = True

# run the update process
update()

# show the window and take focus
root.focus_force()
root.bind('<Escape>', lambda e: root.destroy())
root.mainloop()

# stop the update thread
STOP_THREADS = True
thread_predict.join()
print("Update thread stopped -- End of program")
