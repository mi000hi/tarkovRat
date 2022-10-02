### INVENTORY ITEM DETECTION ###

def edge_detection(img_np):
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#     img = cv2.blur(img, (3,3))
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.Canny(img, threshold1=50, threshold2=200, apertureSize=5)
#     img = cv2.blur(img, (2,2))
    img = cv2.blur(img, (3,3))
    return img

def inventory_line_detection(img):
    img_inventory_edge = edge_detection(img)
    
    img_horizontal = img_inventory_edge.copy()
    img_vertical = img_inventory_edge.copy()

    line_length = 50 #(int) (50 * window_scale_factor)
    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length,1))
    vertical_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1,line_length))

    cv2.erode(img_horizontal, horizontal_struct, img_horizontal)
    cv2.dilate(img_horizontal, horizontal_struct, img_horizontal)

    cv2.erode(img_vertical, vertical_struct, img_vertical)
    cv2.dilate(img_vertical, vertical_struct, img_vertical)

    result = cv2.add(img_horizontal, img_vertical)
    return result

def find_slot_locations(inventory_filtered, slot_gray):
    global slot_locations, nr_valid_predictions
    
    matched_slots = cv2.matchTemplate(inventory_filtered, slot_gray, cv2.TM_CCORR_NORMED)
    # matched_slots = inventory_filtered
    
    threshold = 0.7
    # threshold = 150
    slots_min_x = (int) (1920/3)
    min_distance = 50
    slots = []
    
    for y in range(matched_slots.shape[0]):
        for x in range(matched_slots.shape[1]):
            if matched_slots[y][x] > threshold:
                if x > slots_min_x:
                    slots.append((x,y))
    
    # filter number of locations
    valid_slots = 0
    for i,new_slot in enumerate(slots):
        if valid_slots >= len(slot_locations):
            break
        add_slot = True
        x = new_slot[0]
        y = new_slot[1]
        for j in range(valid_slots):
            s = slot_locations[j]
            if abs(s[0]-x) < min_distance and abs(s[1]-y) < min_distance:
                add_slot = False
                break
        if add_slot == True:
            slot_locations[valid_slots] = (x,y)
            valid_slots += 1
    
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



### ITEM PREDICTION ###

def load_icons_from_disk(verbose=False):
    global all_items_df
    
    icons = []
    for index,item in all_items_df.iterrows():
        filename = path_grid_icons + item['id'] + filename_ending_grid_icon
        if exists(filename):
            img = np.asarray(Image.open(filename))
            img = cv2.resize(img, ((int) (img.shape[0]*window_scale_factor)
                                   , (int) (img.shape[1]*window_scale_factor)))
            icons.append(img)
        else:
            icons.append([])
            if verbose:
                print(f"File {filename} does not exist.")
    return icons

def run_sift(img):
    global sift
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp,des = sift.detectAndCompute(gray,None)
    return kp,des

def create_all_descriptors():
    global icons
    descriptors = []
    for icon in icons:
        if len(icon) == 0:
            descriptors.append(None)
            continue
        icon = icon[0:slot_size, 0:slot_size]
        kp,des = run_sift(icon)
        descriptors.append(des)
    return descriptors

def predict_icon(img):
    global icons, bf, descriptors
    distances = []

    kp,des = run_sift(img)

    for i in range(len(icons)):
        if descriptors[i] is None:
            distances.append(999999)
            continue

        matches = bf.match(des,descriptors[i])
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:30]
        
        if len(matches) == 0:
            distances.append(999999)
            continue

        distance = 0.0
        for match in matches:
            distance += match.distance/len(matches)
            
        distances.append(distance)

    prediction = np.argmin(distances)
    min_distance = min(distances)
    return prediction,min_distance

def predict_all_icons(images, predictions, distances):
    for item in images:
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

def threaded_prediction(items):
    global item_images_updated, nr_valid_predictions, predictions_updated
    global predictions_df, slot_locations
    global predictions, distances, prediction_threshold
    
    while True:
        if not item_images_updated:
            time.sleep(1)
#             print("Thread running...")
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
    print(f'inventory took {t1-t0} s')
    
    # predict each item from inventory
    ## thread is already doing this

    t2 = time.time()
    print(f'item predictions took {t2-t1} s')
    return predictions_df

def place_label(text, x, y, index):
    global root

    label = tk.Label(root, text=text, font=("helvetica", 10), fg='white', bg='black')
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
    root.title("window title")
    root.geometry("%dx%d+%d+%d" % (window_width,window_height,x_window,y_window))
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.configure(bg='red')

    # make color red transparent
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
        index = prediction[2]
        distance = prediction[3]
        if index == -1 or distance > predictions_threshold:
            if not price_labels[i] is None:
                price_labels[i].destroy()
                price_labels[i] = None
            continue
            
        x = prediction[0]
        y = prediction[1]
        price_flea = all_items_df.loc[index, 'fleaMarket']
        price_traders = all_items_df.loc[index, 'prapor':'jaeger']
        price_traders_max = max(price_traders)
        price_max = max(price_flea, price_traders_max)

        if math.isnan(price_max):
            continue
        price = (int) (price_max/1000)
        price_string = str(price) + 'k'
        name = all_items_df.loc[index][0]
        text = name + '\n' + price_string
        place_label(text, x, y, i)
    
    # remove other labels
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
    root.after(100, update)
    
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