import pandas as pd
import math
import pandas as pd
import numpy as np

import pyautogui
import time
import requests

from PIL import Image
from os.path import exists

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from image_manipulation import find_slot_locations, get_item_images_from_inventory, inventory_line_detection, scale_image
#     from image_recognition import predict_icon

import image_manipulation as im
import image_recognition as ir

class Tarkov_inventory:
    'class for the tarkov inventory'

    # file paths
    path_grid_icons = ''
    path_images = ''

    filename_ending_grid_icon = '-grid-image.jpg'
    filename_empty_slot_img = 'slot_empty.png'

    # parameters
    slot_size = 0
    window_scale_factor = 0
    last_api_update = 0

    # variables
    all_items_df = None
    icons = None
    json_data = None
    slot_locations = None

    STOP_THREADS = False

    def __init__(self, path_grid_icons, path_images, slot_size, window_scale_factor, json_data):

        self.path_grid_icons = path_grid_icons
        self.path_images = path_images
        self.slot_size = slot_size
        self.window_scale_factor = window_scale_factor
        self.json_data = json_data

        # load all items
        print("Load item information from web API...")
        all_items = getAllItemsPrices()
        self.all_items_df = pd.DataFrame(columns=['name', 'id', 'width', 'height', 'icon_width', 'icon_height', 'features', 'fleaMarket', 'flea_avg48', 'flea_ch48percent', 'prapor', 'therapist', 'fence', 'skier', 'peacekeeper', 'mechanic', 'ragman', 'jaeger', 'basePrice'], index=range(len(all_items)))
        update_items_df_from_dict(tarkov_inventory=self, all_items=all_items)

        # load item icons from disk
        print("Load item icons from disk...")
        self.icons = load_icons_from_disk(tarkov_inventory=self)

        



### ITEM PREDICTION ###

def tax(tarkov_inventory, itemindex, price_rubels):
    # global all_items_df
    # global tax_t_i,tax_t_r,tax_quantity,tax_quantity_factor

    if price_rubels == 0:
        return 0

    t_i = tarkov_inventory.json_data.get("tax_t_i")
    t_r = tarkov_inventory.json_data.get("tax_t_r")
    p_o_exponent = 1
    p_r_exponent = 1
    baseprice = tarkov_inventory.all_items_df.loc[itemindex, 'basePrice']
    quantity = tarkov_inventory.json_data.get("tax_quantity")
    quantity_factor = tarkov_inventory.json_data.get("tax_quantity_factor")
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

def load_icons_from_disk(tarkov_inventory, verbose=False):
    icons = []
    for index,item in tarkov_inventory.all_items_df.iterrows():
        filename = tarkov_inventory.path_grid_icons + item['id'] + tarkov_inventory.filename_ending_grid_icon

        if not exists(filename):
            icons.append([])
            if verbose:
                print(f"File {filename} does not exist.")
        else:
            width_in_slots = item.loc['width']
            height_in_slots = item.loc['height']
            icon = np.asarray(Image.open(filename))
            icon = im.scale_image(icon, width=(int) (width_in_slots*tarkov_inventory.slot_size)
                    , height=(int) (height_in_slots*tarkov_inventory.slot_size))
            icons.append(icon)

    return icons

def predict_current_inventory(overlay, tarkov_inventory, item_predictor, update_slot_locations):
    # global root

    # hide all tkinter
    if not overlay.root is None:
        overlay.root.withdraw()

    # get inventory
    screenshot = pyautogui.screenshot(region=overlay.window_tarkov)
    screenshot = np.array(screenshot)

    # show all tkinter again
    if not overlay.root is None:
        overlay.root.deiconify()

    # get items from screenshot
    get_predictions_from_inventory(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, inventory=screenshot, update_slot_locations=update_slot_locations)

def threaded_prediction(tarkov_inventory, item_predictor, verbose=False):

    while True:
        # global STOP_THREADS
        if tarkov_inventory.STOP_THREADS:
            break
        if not item_predictor.item_images_updated:
            # update item data every 10 mins
            now = time.time()
            if now - tarkov_inventory.last_api_update > tarkov_inventory.json_data.get("items_flea_market_update_interval_mins") * 60:
                print("Updating the API data.")
                update_items_df(tarkov_inventory=tarkov_inventory, all_items_dict=getAllItemsPrices())
                tarkov_inventory.last_api_update = now
            time.sleep(tarkov_inventory.json_data.get("thread_prediction_sleep_interval_secs"))
            if verbose:
                print("Thread running...")

        else:
            print(f"we have {item_predictor.nr_valid_predictions} new predictions")
            for i in range(item_predictor.nr_valid_predictions):
                item = item_predictor.item_images[i]

                # check if item has already been predicted
                # if item_predictor.distances[i] < tarkov_inventory.json_data.get("predictions_threshold"):
                #     # check if previous prediction is the same item
                #     p,d = ir.predict_icon(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, img=item, matching_item=item_predictor.predictions[i])
                #     if d < tarkov_inventory.json_data.get("predictions_threshold"):
                #         continue

                # update prediction information
                p,d = ir.predict_icon(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, img=item)
                item_predictor.predictions[i] = p
                item_predictor.distances[i] = d

                # update prediction dataframe
                item_predictor.predictions_df.loc[i,'slot_x'] = item_predictor.slot_locations[i][0]
                item_predictor.predictions_df.loc[i,'slot_y'] = item_predictor.slot_locations[i][1]

                item_predictor.predictions_df.loc[i,'predicted_item'] = item_predictor.predictions[i]
                item_predictor.predictions_df.loc[i,'distance'] = item_predictor.distances[i]

                item_predictor.predictions_updated = True

            item_predictor.item_images_updated = False

def get_predictions_from_inventory(tarkov_inventory, item_predictor, inventory, update_slot_locations):

    # get item images
    t0 = time.time()
    if update_slot_locations:
        inventory_filtered = im.inventory_line_detection(tarkov_inventory=tarkov_inventory, img=inventory)
        im.find_slot_locations(tarkov_inventory, item_predictor, inventory_filtered, item_predictor.img_slot_gray)
    im.get_item_images_from_inventory(tarkov_inventory, item_predictor, inventory)
    t1 = time.time()
    print(f'inventory slots and images took {t1-t0} s')

    # predict each item from inventory
    ## thread is already doing this

    # TODO: can return be removed?
    # return predictions_df

def get_price_per_slot(tarkov_inventory, item_index):
    # global all_items_df, currency_dollar_to_rubles_factor

    traders = {0:'prapor', 1:'therapist', 2:'fence', 3:'skier', 4:'peacekeeper', 5:'mechanic', 6:'ragman', 7:'jaeger'}
    price_flea = tarkov_inventory.all_items_df.loc[item_index, 'fleaMarket']
    price_traders = tarkov_inventory.all_items_df.loc[item_index, 'prapor':'jaeger'].copy()

    #change all currency to rubels
    price_peacekeeper = price_traders['peacekeeper']
    price_traders['peacekeeper'] = price_peacekeeper * tarkov_inventory.json_data.get("currency_dollar_to_rubles_factor")

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
        flea_tax = tax(tarkov_inventory, item_index, price_flea)

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
    nr_slots = tarkov_inventory.all_items_df.loc[item_index, 'width'] * tarkov_inventory.all_items_df.loc[item_index, 'height']
    price_max = price_max/nr_slots

    return price_max, trader

def format_price_for_label(tarkov_inventory, prediction_index, price_max, trader):
    # global all_items_df,overlay_label_show_trader

    price = (int) (price_max/1000)
    price_string = str(price) + 'k'
    if price == 0:
        price = (int) (price_max)
        price_string = str(price)
    name = tarkov_inventory.all_items_df.loc[prediction_index][0]

    # create the new label
    text = name + '\n' + price_string
    if tarkov_inventory.json_data.get("overlay_label_show_trader"):
        text = text + '\n' + trader

    return text

def update_items_df_from_dict(tarkov_inventory, all_items):
    # iterate over all items
    for index in range(len(all_items)):
        tarkov_inventory.all_items_df.iloc[index]['name'] = all_items[index].get('shortName')
        tarkov_inventory.all_items_df.iloc[index]['id'] = all_items[index].get('id')
        tarkov_inventory.all_items_df.iloc[index]['width'] = all_items[index].get('width')
        tarkov_inventory.all_items_df.iloc[index]['height'] = all_items[index].get('height')
        tarkov_inventory.all_items_df.iloc[index]['flea_avg48'] = all_items[index].get('avg24hPrice')
        tarkov_inventory.all_items_df.iloc[index]['flea_ch48percent'] = all_items[index].get('changeLast48hPercent')
        tarkov_inventory.all_items_df.iloc[index]['basePrice'] = all_items[index].get('basePrice')

        # iterate over all traders that can buy the item
        for offer in all_items[index].get('sellFor'):
            trader = offer.get('source')
            price  = offer.get('price')
            tarkov_inventory.all_items_df.iloc[index][trader] = price

def update_items_df(tarkov_inventory, all_items_dict):

    # iterate over all items
    for index in range(len(all_items_dict)):
        if tarkov_inventory.all_items_df.iloc[index]['id'] != all_items_dict[index].get('id'):
            print(f"The item {tarkov_inventory.all_items_df.iloc[index]['name']} does not match the new updated information!")
            continue
        tarkov_inventory.all_items_df.iloc[index]['flea_avg48'] = all_items_dict[index].get('avg24hPrice')
        tarkov_inventory.all_items_df.iloc[index]['flea_ch48percent'] = all_items_dict[index].get('changeLast48hPercent')

        # iterate over all traders that can buy the item
        for offer in all_items_dict[index].get('sellFor'):
            trader = offer.get('source')
            price  = offer.get('price')
            tarkov_inventory.all_items_df.iloc[index][trader] = price

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