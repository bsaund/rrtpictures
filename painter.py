#!/usr/bin/python

import cv2
import numpy as np
import time


def overlay(base, overlay, mask):
    """
    Returns an image of overlay where mask, and background where not mask
    """
    mask_inv = cv2.bitwise_not(mask)
    img_bk = cv2.bitwise_and(base, base, mask=mask_inv)
    img_fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    return cv2.add(img_bk, img_fg)


def get_hsv_list():
    d_hue = 32
    d_value = 32
    d_sat = 256
    

    for value in range(0, 255, d_value):
        for hue in range(0, 255, d_hue):
            for saturation in range(0, 255, d_sat):


                yield [(hue, saturation, value),
                       (hue+d_hue, saturation + d_sat, value + d_value)]


class Painter:
    def __init__(self, photo_filepath):
        self.photo= cv2.imread(photo_filepath)
        print self.photo.shape
        self.photo = cv2.resize(self.photo, (1242, 932))
        self.blurred_photo = cv2.GaussianBlur(self.photo, (7,7), 0)
        self.photo_hsv = cv2.cvtColor(self.blurred_photo, cv2.COLOR_BGR2HSV)
        self.create_blank_canvas()

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
        # self.canvas = cv2.bitwise_and(self.photo, self.photo, mask=mask)
        # self.canvas[mask] = [0,0,255]
        self.canvas = overlay(self.canvas, self.photo, mask)

    

    def run(self):
        # while(True):
        for lower, upper in get_hsv_list():
            self.filter(lower, upper)
            self.display()
            time.sleep(0.1)

        while(True):
            self.display()
        
if __name__=="__main__":
    pic = Painter("Brad_with_victor.jpg")
    pic.run()
