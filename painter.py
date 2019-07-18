#!/usr/bin/python

import cv2
import numpy as np
import time
import IPython


def maskedWeighted(bg, alpha, fg, beta, mask):
    """
    Performs a weighted add only in the mask region. Outside of the mask, just uses bg
    """
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
    img_mix = cv2.addWeighted(bg, alpha, fg, beta, 0)
    img_fg = cv2.bitwise_and(img_mix, img_mix, mask=mask)
    return cv2.add(img_bg, img_fg)


def merge(bg, fg, alpha_mask):
    """
    alpha_mask must be a matrix with the same size (at least in 2D) of bg and fg.
    alpha_mask should range from 0.0 to 1.0
    """
    if len(alpha_mask.shape) == 2:
        alpha_mask = np.repeat(alpha_mask[:,:,np.newaxis], 3, axis=2)
    return (fg*alpha_mask + bg*(1-alpha_mask)).astype(np.uint8)

def overlay(base, overlay, mask):
    """
    Returns an image of overlay where mask, and background where not mask
    """
    mask_inv = cv2.bitwise_not(mask)
    img_bk = cv2.bitwise_and(base, base, mask=mask_inv)
    img_fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    return cv2.add(img_bk, img_fg)

def select_start_point(mask):
    r = np.random.random(mask.shape)
    return np.unravel_index(np.argmax(r * mask), r.shape)


def get_hsv_list():
    d_hue = 32
    d_value = 64
    d_sat = 128

    for value in range(0, 255, d_value):
        # if value < 64:
        #     yield [(0, 0, value), (255, 255, value)]
        # else:
            for hue in range(0, 255, d_hue):
                for saturation in range(0, 255, d_sat):

                    print (hue, saturation, value)
                    yield [(hue, saturation, value),
                           (hue+d_hue, saturation + d_sat, value + d_value)]

def radial_brush(radius, bristles, weight=0.5):
    brush = np.ones([radius*2 + 1, radius*2+1])

    for theta in range(bristles):
        r = radius * np.random.random()
        x = int(np.cos(theta)*r) + radius
        y = int(np.sin(theta)*r) + radius
        # IPython.embed()
        brush[x, y] *= 0.7

    return 1.0 - brush

def place_brush(brush, canvas, pos):
    mask = np.zeros(canvas.shape[0:2], np.double)
    x_0 = pos[0] - brush.shape[0]/2
    x_1 = pos[0] + brush.shape[0]/2 + 1
    y_0 = pos[1] - brush.shape[1]/2
    y_1 = pos[1] + brush.shape[1]/2 + 1

    if x_0 < 0 or x_1 >= canvas.shape[0]:
        return None
    if y_0 < 0 or y_1 >= canvas.shape[1]:
        return None
    
    mask[x_0:x_1, y_0:y_1] = brush
    return mask


def dab_fill(canvas, color, brush, pos):
    """
    canvas is the standard canvas
    color is a canvas-sized array of the paint color
    brush is the brush used
    mask is an array of 0s or 1s
    """
    # num_dabs = np.sum(mask)/200
    num_dabs = 1
    
    # IPython.embed()
    # for _ in range(int(num_dabs)):
    #     pos = select_start_point(mask)
    dab_mask = place_brush(brush, canvas, pos)
    if dab_mask is None:
        return canvas
    canvas = merge(canvas, color, dab_mask)
    return canvas

class Painter:
    def __init__(self, photo_filepath):
        self.photo= cv2.imread(photo_filepath)
        # print self.photo.shape
        self.photo = cv2.resize(self.photo, (1242, 932))
        self.blurred_photo = cv2.GaussianBlur(self.photo, (55,55), 0)
        self.photo_hsv = cv2.cvtColor(self.blurred_photo, cv2.COLOR_BGR2HSV)
        self.create_blank_canvas()
        self.painting_done = False
        self.region_countdown = 0
        self.active_region = None
        self.active_color = None
        self.hsv_bands = get_hsv_list()

    def create_blank_canvas(self):
        self.canvas = np.zeros(self.photo.shape, np.uint8)
        self.canvas[:,:,0] = 220
        self.canvas[:,:,1] = 245
        self.canvas[:,:,2] = 245

    def display(self):
        cv2.imshow("painting", self.canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    def filter(self, lower, upper):
        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)

        mask = cv2.inRange(self.photo_hsv, lower, upper)

        mask = cv2.GaussianBlur(mask, (17,17), 0)
        mask = cv2.inRange(mask, 50, 255)

        print "sum: ", np.sum(mask)/255
        if np.sum(mask)/255 < 200:
            return None, None
        
        # self.canvas = cv2.bitwise_and(self.photo, self.photo, mask=mask)
        # self.canvas[mask] = [0,0,255]
        fill_color = np.zeros(self.photo.shape, np.uint8)
        fill_color[:,:,0] = (lower[0]+upper[0])/2
        fill_color[:,:,1] = (lower[1]+upper[1])/2
        fill_color[:,:,2] = (lower[2]+upper[2])/2

        fill_color = cv2.cvtColor(fill_color, cv2.COLOR_HSV2BGR)
        
        # self.canvas = overlay(self.canvas, self.photo, mask)
        
        # self.canvas = overlay(self.canvas, fill_color, mask)

        mask = mask.astype(np.double)/255
        return mask, fill_color
        # mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        # self.canvas = maskedWeighted(self.canvas, 0.2, fill_color, 0.8, mask)
        # self.canvas = merge(self.canvas, fill_color, mask)

    def get_new_region_of_interest(self):
        self.active_region = None
        while self.active_region is None:
            try:
                lower, upper = self.hsv_bands.next()
            except:
                return False
            self.active_region, self.active_color = self.filter(lower, upper)
        return True

    def get_region_of_interest(self):
        if self.region_countdown <= 0:
            if not self.get_new_region_of_interest():
                self.painting_done = True
                return None, None
            self.region_countdown = np.sum(self.active_region)/200
            
        return self.active_region, self.active_color
        

    def paint(self):
        region, color = self.get_region_of_interest()
        if self.painting_done:
            return

        self.canvas = dab_fill(self.canvas, color, radial_brush(50, 3000), select_start_point(region))
        self.region_countdown -= 1
        # time.sleep(0.1)
        
    

    def run(self):
        # while(True):
        # for lower, upper in get_hsv_list():
        #     self.paint(lower, upper)
        #     self.display()
        while not self.painting_done:
            self.paint()
            self.display()


        print "Painting finished"
        while(True):
            self.display()

class WIP:
    def __init__(self, photo_filepath):

        self.photo = cv2.imread(photo_filepath)
        self.canvas = cv2.resize(self.photo, (1242, 932))

        # self.canvas = np.ones((1242, 932, 3))*0
        # self.canvas = np.ones((1242, 932, 3))*0
        # self.canvas[:,:,0] = 255

        self.circle = np.ones(self.canvas.shape, np.uint8)*255
        cv2.circle(self.circle, (500, 500), 150, (0,0,255), -1)
        # cv2.circle(self.canvas, (500, 500), 50, (0,0,255), -1)

        mask = cv2.inRange(self.circle, (0,0,1), (1, 1, 255))
        # IPython.embed()
        # self.canvas = overlay(self.canvas, self.circle, mask)
        # self.canvas = cv2.addWeighted(self.canvas, 0.5, self.circle, 0.5, 0)
        # self.canvas = maskedWeighted(self.canvas, 0.5, self.circle, 0.5, mask)


        a = mask.astype(np.double)/255 * 0.8

        a = place_brush(radial_brush(50, 3000), self.canvas, (450,450))
        self.canvas = merge(self.canvas, self.circle, a)
        

    def display(self):
        cv2.imshow("dummy", self.canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    def run(self):
        while True:
            self.display()
    
        
if __name__=="__main__":
    pic = Painter("Brad_with_victor.jpg")
    # pic = WIP("Brad_with_victor.jpg")
    pic.run()
