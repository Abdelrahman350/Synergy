import numpy as np
import cv2
from numpy.random import rand, uniform, normal

def normalize_image(image):
    image = image.astype(float)
    image /= 127.5
    image -= 1.0
    return image

def denormalize_image(image):
    image += 1.0
    image *= 127.5
    return image

def resize_image(image, input_shape=(224, 224)):
    original_shape = image.shape
    image = cv2.resize(image, input_shape[0:2], interpolation = cv2.INTER_AREA)
    resized_shape = image.shape
    aspect_0 = resized_shape[0]/float(original_shape[0])
    aspect_1 = resized_shape[1]/float(original_shape[1])
    aspect_ratio = np.array([aspect_0, aspect_1, 1])
    return image, aspect_ratio

def colorjitter(image):
    cj_type = np.random.choice(['b', 's', 'c'])
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == 'b':
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image
    elif cj_type == 's':
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image
    elif cj_type == 'c':
        brightness = 10
        contrast = np.random.randint(40, 101)
        dummy = np.int16(image)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        image = np.uint8(dummy)
        return image
    else:
        return image

def noisy(image_ori):
    noise_type = np.random.choice(['gauss', 'sp'])
    '''
    ### Adding Noise ###
    image_ori: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == 'gauss':
        image = image_ori.copy() 
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image, gauss)
        return image    
    elif noise_type == 'sp':
        image = image_ori.copy() 
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image
    else:
        return image_ori

def filters(image_ori):
    f_type = np.random.choice(['blur', 'gaussian', 'median'])
    '''
    ### Filtering ###
    image_ori: image
    f_type: {blur: blur, gaussian: gaussian, median: median}
    '''
    if f_type == 'blur':
        image = image_ori.copy()
        fsize = 9
        return cv2.blur(image, (fsize, fsize))
    elif f_type == 'gaussian':
        image = image_ori.copy()
        fsize = 9
        return cv2.GaussianBlur(image, (fsize, fsize), 0)
    elif f_type == 'median':
        image = image_ori.copy()
        fsize = 9
        return cv2.medianBlur(image, fsize)
    else:
        return image_ori

def gray_img(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.merge([img_gray, img_gray, img_gray])
    return image

def crop(image, lmks):
    x_min = np.min(lmks[:, :, 0])
    y_min = np.min(lmks[:, :, 1])
    x_max = np.max(lmks[:, :, 0])
    y_max = np.max(lmks[:, :, 1])    
    rect = np.array([x_min, y_min, x_max, y_max])
    roi_box = rect.copy()

    # enlarge the bbox a little and do a square crop
    HCenter = (rect[1] + rect[3])/2
    WCenter = (rect[0] + rect[2])/2
    side_len = roi_box[3] - roi_box[1]
    scale = uniform(low=1.1, high=1.4)
    margin = side_len * scale // 2
    offset = rand() * normal(0, 0.15)
    roi_box[0] = WCenter * (1+offset) - margin
    roi_box[1] = HCenter * (1+offset) - margin
    roi_box[2] = WCenter * (1+offset) + margin
    roi_box[3] = HCenter * (1+offset) + margin
    image = crop_img(image, roi_box)
    return image, roi_box

def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh
    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res