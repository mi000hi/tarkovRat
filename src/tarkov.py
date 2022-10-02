import pandas as pd
import math
import pandas as pd
import numpy as np

import pyautogui
import time
import requests

from PIL import Image
from os.path import exists

from image_manipulation import *
from image_recognition import *


### ITEM PREDICTION ###

def tax(itemindex, price_rubels):
    global all_items_df
    global tax_t_i,tax_t_r,tax_quantity,tax_quantity_factor

    if price_rubels == 0:
        return 0

    t_i = tax_t_i
    t_r = tax_t_r
    p_o_exponent = 1
    p_r_exponent = 1
    baseprice = all_items_df.loc[itemindex, 'basePrice']
    quantity = tax_quantity
    quantity_factor = tax_quantity_factor
    v_o = baseprice * quantity / quantity_factor
    if v_o == 0:
        return 0
    v_r = price_rubels # value of requirements
    if v_r < v_o:
        p_o_exponent = 1.08
    p_o = np.log10(v_o / v_r)**p_o_exponent
    if v_r >= v_o:
        p_r_exponent = 1.08
    p_r = np.log10(v_r / v_o)**p_r_exponent

    tax = v_o * t_i * (4**p_o) * quantity_factor + v_r * t_r * (4**p_r) * quantity_factor
    return tax

def load_icons_from_disk(all_items_df, path_grid_icons, filename_ending_grid_icon, slot_size, verbose=False):
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
            icon = scale_image(icon, width=(int) (width_in_slots*slot_size)
                    , height=(int) (height_in_slots*slot_size))
            icons.append(icon)

    return icons

def predict_current_inventory(predictions_df, update_slot_locations):
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
    get_predictions_from_inventory(screenshot, update_slot_locations)

def threaded_prediction(items, verbose=False):
    global item_images_updated, nr_valid_predictions, predictions_updated
    global predictions_df, slot_locations
    global predictions, distances, prediction_threshold
    global last_api_update, all_items_df
    global items_flea_market_update_interval_mins,thread_prediction_sleep_interval_secs
    global predictions_threshold

    while True:
        global STOP_THREADS
        if STOP_THREADS:
            break
        if not item_images_updated:
            # update item data every 10 mins
            now = time.time()
            if now - last_api_update > items_flea_market_update_interval_mins * 60:
                print("Updating the API data.")
                update_items_df(getAllItemsPrices())
                last_api_update = now
            time.sleep(thread_prediction_sleep_interval_secs)
            if verbose:
                print("Thread running...")

        else:
            print(f"we have {nr_valid_predictions} new predictions")
            for i in range(nr_valid_predictions):
                item = item_images[i]

                # check if item has already been predicted
                if distances[i] < predictions_threshold:
                    # check if previous prediction is the same item
                    p,d = predict_icon(item, matching_item=predictions[i])
                    if d < predictions_threshold:
                        continue

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

def get_predictions_from_inventory(inventory, update_slot_locations):
    global img_slot_gray
    global thread_predict

    # get item images
    t0 = time.time()
    if update_slot_locations:
        inventory_filtered = inventory_line_detection(inventory)
        find_slot_locations(inventory_filtered, img_slot_gray)
    get_item_images_from_inventory(inventory)
    t1 = time.time()
    print(f'inventory slots and images took {t1-t0} s')

    # predict each item from inventory
    ## thread is already doing this

    # TODO: can return be removed?
    # return predictions_df

def get_price_per_slot(item_index):
    global all_items_df, currency_dollar_to_rubles_factor

    traders = {0:'prapor', 1:'therapist', 2:'fence', 3:'skier', 4:'peacekeeper', 5:'mechanic', 6:'ragman', 7:'jaeger'}
    price_flea = all_items_df.loc[item_index, 'fleaMarket']
    price_traders = all_items_df.loc[item_index, 'prapor':'jaeger'].copy()

    #change all currency to rubels
    price_peacekeeper = price_traders['peacekeeper']
    price_traders['peacekeeper'] = price_peacekeeper * currency_dollar_to_rubles_factor

    # find best trader
    price_traders_max = np.nanmax(price_traders.values.tolist() + [0])
    best_trader = 0
    if price_traders_max == 0:
        best_trader = '-1'
    else:
        best_trader = traders.get(np.nanargmax(price_traders))

    # subtract tax from flea price
    flea_tax = 0
    if math.isnan(price_flea):
        price_flea = 0
    else:
        flea_tax = tax(item_index, price_flea)

    # check trader or flea
    price_max = max(price_traders_max, price_flea-flea_tax)
    flea_best = np.argmax([price_traders_max, price_flea-flea_tax])

    trader = best_trader
    if flea_best == 1:
        trader = 'flea'

    # no price available
    if math.isnan(price_max):
        return 0

    # price per slot
    nr_slots = all_items_df.loc[item_index, 'width'] * all_items_df.loc[item_index, 'height']
    price_max = price_max/nr_slots

    return price_max, trader

def format_price_for_label(prediction_index, price_max, trader):
    global all_items_df,overlay_label_show_trader

    price = (int) (price_max/1000)
    price_string = str(price) + 'k'
    if price == 0:
        price = (int) (price_max)
        price_string = str(price)
    name = all_items_df.loc[prediction_index][0]

    # create the new label
    text = name + '\n' + price_string
    if overlay_label_show_trader:
        text = text + '\n' + trader

    return text

def update_items_df_from_dict(df, all_items):
    # iterate over all items
    for index in range(len(all_items)):
        df.iloc[index]['name'] = all_items[index].get('shortName')
        df.iloc[index]['id'] = all_items[index].get('id')
        df.iloc[index]['width'] = all_items[index].get('width')
        df.iloc[index]['height'] = all_items[index].get('height')
        df.iloc[index]['flea_avg48'] = all_items[index].get('avg24hPrice')
        df.iloc[index]['flea_ch48percent'] = all_items[index].get('changeLast48hPercent')
        df.iloc[index]['basePrice'] = all_items[index].get('basePrice')

        # iterate over all traders that can buy the item
        for offer in all_items[index].get('sellFor'):
            trader = offer.get('source')
            price  = offer.get('price')
            df.iloc[index][trader] = price

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