import tkinter as tk
import pandas as pd
import pandas as pd
import keyboard

from tarkov import *
from image_recognition import *

def place_label(text, x, y, index, font_color='white', font=None, anchor='nw', color=None, verbose=False):
    global root,font_label_item
    global label_y_offset
    global overlay_transparent_color,price_labels

    if color is None:
        color = font_color

    if font is None:
        font = font_label_item
    label = tk.Label(root, text=text, font=font, fg=color, bg=overlay_transparent_color)
    label.place(x=x, y=y+label_y_offset, anchor=anchor)

    # remove old label
    if verbose and index == -1:
        print(f"label index is -1 => here only when predict item under mouse -- correct??")
    if index < len(price_labels):
        if not price_labels[index] is None:
            price_labels[index].destroy()
        price_labels[index] = label
    else:
        # TODO: is this needed???
        price_labels.append(label)

def create_overlay():
    global overlay_transparent_color

    root = tk.Tk()
    root.title("this is not a virus. be a chad")
    root.geometry("%dx%d+%d+%d" % (window_width,window_height,x_window,y_window))
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.configure(bg='red')

    # make color black transparent
    root.wm_attributes('-transparentcolor', overlay_transparent_color)
    return root

def update_price_labels():
    remove_price_labels(remove_manual_label_too=False)
    add_price_labels()

def add_price_labels():
    global nr_valid_predictions, predictions_df, price_labels, all_items_df
    global predictions_threshold,overlay_nr_smallest_prices,overlay_label_cheap_items_color

    prices = []
    prices_over_1k = []
    traders = []
    predictions = []

    if nr_valid_predictions > 0:
        for i in range(nr_valid_predictions):
            prediction = predictions_df.loc[i]
            prediction_index = prediction[2]
            prediction_distance = prediction[3]
            if prediction_index == -1 or prediction_distance > predictions_threshold: #    if not price_labels[i] is None:
            #        price_labels[i].destroy()
            #        price_labels[i] = None
                predictions.append(-1)
                prices.append(-1)
                traders.append(-1)
                continue

            # find prices for predicted item
            price_max,trader = get_price_per_slot(prediction_index)

            predictions.append(prediction_index)
            prices.append(price_max)
            traders.append(trader)

            if price_max > 1000:
                prices_over_1k.append(price_max)

        # find lowest priced items
        prices_df = pd.DataFrame(prices_over_1k)
        lowest_prices_threshold = max(prices_df[0].nsmallest(overlay_nr_smallest_prices))

        for i in range(nr_valid_predictions):
            if predictions[i] == -1:
                continue

            prediction = predictions_df.loc[i]

            # label location
            x = prediction[0]
            y = prediction[1]

            text = format_price_for_label(predictions[i], prices[i], traders[i])

            if prices[i] <= lowest_prices_threshold:
                place_label(text, x, y, i, color=overlay_label_cheap_items_color)
            else:
                place_label(text, x, y, i)

    # remove other old labels except manual label
    for i in range(len(predictions), len(price_labels)-1):
        if price_labels[i] is None:
            continue
        price_labels[i].destroy()

def remove_price_labels(remove_manual_label_too=True):
    global price_labels

    for i in range(len(price_labels)-1-1):
        label = price_labels[i]
        if not label is None:
            label.destroy()

    if remove_manual_label_too:
        label = price_labels[-1]
        if not label is None:
            label.destroy()

def update():
    global root,key_manual_predict
    global overlay_update_interval_msecs
    global predictions_updated

    if keyboard.is_pressed(key_manual_predict):
        predict_item_under_mouse()

    if labels_visible and predictions_updated:
        update_price_labels()

    # run itself again after 100 ms
    root.after(overlay_update_interval_msecs, update)

def update_predictions():
    global predictions_df
    predict_current_inventory(predictions_df, update_slot_locations=False)

def update_predictions_and_slots():
    global predictions_df
    predict_current_inventory(predictions_df, update_slot_locations=True)

def show_hide_labels():
    global price_labels, labels_visible
    global button2

    if labels_visible:
        for i in range(len(price_labels)):
            if not price_labels[i] is None:
                price_labels[i].destroy()
                price_labels[i] = None
        labels_visible = False
        button2['text'] = "show labels"
    else:
        add_price_labels()
        labels_visible = True
        button2['text'] = "hide labels"