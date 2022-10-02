import cv2



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

def find_slot_locations(inventory_filtered, slot_gray, min_x=1000):
    global slot_locations, nr_valid_predictions
    global window_scale_factor
    global slot_locations_min_x,slot_locations_min_y,slot_locations_max_x,slot_locations_max_y
    global slot_locations_threshold,slot_locations_min_distance

    slot_x_offset = min_x
    slot_y_offset = slot_locations_min_y
    slot_x_max = slot_locations_max_x
    slot_y_max = slot_locations_max_y
    inv = inventory_filtered[slot_y_offset:slot_y_max, slot_x_offset:slot_x_max].copy()
    slot = slot_gray.copy()

    matched_slots = cv2.matchTemplate(inv, slot, cv2.TM_CCORR_NORMED)

    threshold = slot_locations_threshold
    min_distance = slot_locations_min_distance * window_scale_factor

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

    # TODO: refactor nr_valid_predictions variable
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