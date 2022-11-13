import tkinter as tk
import pandas as pd
import pandas as pd
import keyboard

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from image_recognition import predict_item_under_mouse
#     from tarkov import format_price_for_label, get_price_per_slot, predict_current_inventory
#     from main import update_json_variables

import image_recognition as ir
import tarkov as tr
import main

class Overlay:

    # variables
    window_tarkov_position = (0,0)
    window_tarkov_size = (0,0)
    window_tarkov = (0,0,0,0)
    root = None
    button2 = None
    json_data = None

    labels_visible = True

    font_label_item = None
    font_label_manual_item = None


    def __init__(self, tarkov_inventory, item_predictor, window_tarkov_position, window_tarkov_size):

        self.window_tarkov_position = window_tarkov_position
        self.window_tarkov_size = window_tarkov_size
        self.window_tarkov = (window_tarkov_position[0], window_tarkov_position[1], window_tarkov_size[0], window_tarkov_size[1])

        self.json_data = tarkov_inventory.json_data
        self.font_label_item = ("helvetica", tarkov_inventory.json_data.get("font_label_item_size"))
        self.font_label_manual_item = ("helvetica", tarkov_inventory.json_data.get("font_label_manual_item_size"))


        # create the window
        print("Create the overlay...")
        self.root = create_overlay(self.json_data, self.window_tarkov_position, self.window_tarkov_size)

        # create a transparent frame to make a border
        my_frame = tk.Frame(self.root, width=window_tarkov_size[0]-2*tarkov_inventory.json_data.get("overlay_border_size"), height=window_tarkov_size[1]-2*tarkov_inventory.json_data.get("overlay_border_size"), bg=tarkov_inventory.json_data.get("overlay_transparent_color"))
        my_frame.place(x=tarkov_inventory.json_data.get("overlay_border_size"),y=tarkov_inventory.json_data.get("overlay_border_size"))

        # add a button
        button1 = tk.Button(self.root, text='Update predictions', fg='blue', command=lambda: update_predictions(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, overlay=self))
        button1.grid(row=0,column=0)
        button2_text = ''
        if self.labels_visible:
            button2_text = 'hide labels'
        else:
            button2_text = 'show labels'
        self.button2 = tk.Button(self.root, text=button2_text, fg='blue', command=lambda: show_hide_labels(overlay=self, item_predictor=item_predictor, tarkov_inventory=tarkov_inventory))
        self.button2.grid(row=0,column=1)
        button3 = tk.Button(self.root, text='read JSON file', fg='blue', command=main.update_json_variables)
        button3.grid(row=0,column=2)
        button4 = tk.Button(self.root, text='Update predictions and slots', fg='blue', command=lambda: update_predictions_and_slots(tarkov_inventory=tarkov_inventory, item_predictor=item_predictor, overlay=self))
        button4.grid(row=0, column=3)


def place_label(overlay, item_predictor, text, x, y, index, font_color='white', font=None, anchor='nw', color=None, verbose=False):
    # global root,font_label_item
    # global label_y_offset
    # global overlay_transparent_color,price_labels

    if color is None:
        color = font_color

    if font is None:
        font = overlay.font_label_item
    label = tk.Label(overlay.root, text=text, font=font, fg=color, bg=overlay.json_data.get("overlay_transparent_color"))
    label.place(x=x, y=y+overlay.json_data.get("label_y_offset"), anchor=anchor)

    # remove old label
    if verbose and index == -1:
        print(f"label index is -1 => here only when predict item under mouse -- correct??")
    if index < len(item_predictor.price_labels):
        if not item_predictor.price_labels[index] is None:
            item_predictor.price_labels[index].destroy()
        item_predictor.price_labels[index] = label
    else:
        # TODO: is this needed???
        item_predictor.price_labels.append(label)

def create_overlay(json_data, window_tarkov_pos, window_tarkov_size):
    # global overlay_transparent_color

    root = tk.Tk()
    root.title("this is not a virus. be a chad")
    root.geometry("%dx%d+%d+%d" % (window_tarkov_size[0],window_tarkov_size[1],window_tarkov_pos[0],window_tarkov_pos[1]))
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.configure(bg='red')

    # make color black transparent
    root.wm_attributes('-transparentcolor', json_data.get("overlay_transparent_color"))
    return root

def update_price_labels(overlay, item_predictor, tarkov_inventory):
    remove_price_labels(item_predictor, remove_manual_label_too=False)
    add_price_labels(overlay, item_predictor, tarkov_inventory)

def add_price_labels(overlay, item_predictor, tarkov_inventory):
    # global nr_valid_predictions, predictions_df, price_labels, all_items_df
    # global predictions_threshold,overlay_nr_smallest_prices,overlay_label_cheap_items_color

    prices = []
    prices_over_1k = []
    traders = []
    predictions = []

    if item_predictor.nr_valid_predictions > 0:
        for i in range(item_predictor.nr_valid_predictions):
            prediction = item_predictor.predictions_df.loc[i]
            prediction_index = prediction[2]
            prediction_distance = prediction[3]
            if prediction_index == -1 or prediction_distance > item_predictor.json_data.get("predictions_threshold"): #    if not price_labels[i] is None:
            #        price_labels[i].destroy()
            #        price_labels[i] = None
                predictions.append(-1)
                prices.append(-1)
                traders.append(-1)
                continue

            # find prices for predicted item
            price_max,trader = tr.get_price_per_slot(tarkov_inventory, prediction_index)

            predictions.append(prediction_index)
            prices.append(price_max)
            traders.append(trader)

            if price_max > 1000:
                prices_over_1k.append(price_max)

        # find lowest priced items
        prices_df = pd.DataFrame(prices_over_1k)
        lowest_prices_threshold = max(prices_df[0].nsmallest(item_predictor.json_data.get("overlay_nr_smallest_prices")))

        for i in range(item_predictor.nr_valid_predictions):
            if predictions[i] == -1:
                continue

            prediction = item_predictor.predictions_df.loc[i]

            # label location
            x = prediction[0]
            y = prediction[1]

            text = tr.format_price_for_label(tarkov_inventory, predictions[i], prices[i], traders[i])

            if prices[i] <= lowest_prices_threshold:
                place_label(overlay, item_predictor, text, x, y, i, color=item_predictor.json_data.get("overlay_label_cheap_items_color"))
            else:
                place_label(overlay, item_predictor, text, x, y, i)

    # remove other old labels except manual label
    for i in range(len(predictions), len(item_predictor.price_labels)-1):
        if item_predictor.price_labels[i] is None:
            continue
        item_predictor.price_labels[i].destroy()

def remove_price_labels(item_predictor, remove_manual_label_too=True):
    # global price_labels

    for i in range(len(item_predictor.price_labels)-1-1):
        label = item_predictor.price_labels[i]
        if not label is None:
            label.destroy()

    if remove_manual_label_too:
        label = item_predictor.price_labels[-1]
        if not label is None:
            label.destroy()

def update(tarkov_inventory, overlay, item_predictor):
    # global root,key_manual_predict
    # global overlay_update_interval_msecs
    # global predictions_updated

    if keyboard.is_pressed(tarkov_inventory.json_data.get("key_manual_predict")):
        ir.predict_item_under_mouse(tarkov_inventory, item_predictor, overlay)

    if overlay.labels_visible and item_predictor.predictions_updated:
        update_price_labels(overlay, item_predictor, tarkov_inventory)

    # run itself again after 100 ms
    # TODO give arguments to update function
    overlay.root.after(tarkov_inventory.json_data.get("overlay_update_interval_msecs"), lambda : update(tarkov_inventory, overlay, item_predictor))

def update_predictions(tarkov_inventory, item_predictor, overlay):
    # global predictions_df
    tr.predict_current_inventory(overlay, tarkov_inventory, item_predictor, update_slot_locations=False)

def update_predictions_and_slots(tarkov_inventory, item_predictor, overlay):
    # global predictions_df
    tr.predict_current_inventory(overlay, tarkov_inventory, item_predictor, update_slot_locations=True)

def show_hide_labels(overlay, item_predictor, tarkov_inventory):
    # global price_labels, labels_visible
    # global button2

    if overlay.labels_visible:
        for i in range(len(item_predictor.price_labels)):
            if not item_predictor.price_labels[i] is None:
                item_predictor.price_labels[i].destroy()
                item_predictor.price_labels[i] = None
        overlay.labels_visible = False
        overlay.button2['text'] = "show labels"
    else:
        add_price_labels(overlay=overlay, item_predictor=item_predictor, tarkov_inventory=tarkov_inventory)
        overlay.labels_visible = True
        overlay.button2['text'] = "hide labels"