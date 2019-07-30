#!/usr/bin/python

import cv2
import numpy as np
import IPython

"""
Utils for paining using opencv
"""


def merge(bg, fg, alpha_mask):
    """
    alpha_mask must be a matrix with the same size (at least in 2D) of bg and fg.
    alpha_mask should range from 0.0 to 1.0
    """
    if len(alpha_mask.shape) == 2:
        alpha_mask = np.repeat(alpha_mask[:,:,np.newaxis], 3, axis=2)
    return (fg*alpha_mask + bg*(1-alpha_mask)).astype(np.uint8)

def maskedWeighted(bg, alpha, fg, beta, mask):
    """
    Performs a weighted add only in the mask region. Outside of the mask, just uses bg
    """
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
    img_mix = cv2.addWeighted(bg, alpha, fg, beta, 0)
    img_fg = cv2.bitwise_and(img_mix, img_mix, mask=mask)
    return cv2.add(img_bg, img_fg)

def overlay(base, overlay, mask):
    """
    Returns an image of overlay where mask, and background where not mask
    """
    mask_inv = cv2.bitwise_not(mask)
    img_bk = cv2.bitwise_and(base, base, mask=mask_inv)
    img_fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    # IPython.embed()
    return cv2.add(img_bk, img_fg)


def radial_brush(radius, bristles, weight=0.1):
    """
    Returns a weighting matrix representing the bristles of a brush
    """
    brush = np.ones([radius*2 + 1, radius*2+1])

    for theta in range(bristles):
        r = radius * (np.random.random())
        x = int(np.cos(theta)*r+radius)
        y = int(np.sin(theta)*r+radius)
        brush[x, y] *= 1-weight
    
    brush = np.clip(brush, (1-weight)**2, 1.0)
    return 1.0 - brush


def paint_at(canvas, color, brush, pos):
    """
    More efficient that raw merge of giant canvases
    """
    x_0 = pos[0] - brush.shape[0]/2
    x_1 = pos[0] + brush.shape[0]/2 + 1
    y_0 = pos[1] - brush.shape[1]/2
    y_1 = pos[1] + brush.shape[1]/2 + 1

    if x_0 < 0 or x_1 >= canvas.shape[0]:
        return canvas
    if y_0 < 0 or y_1 >= canvas.shape[1]:
        return canvas

    color_window = np.zeros((brush.shape[0], brush.shape[1], 3), np.uint8)
    color_window[:,:,0] = color[0,0,0]
    color_window[:,:,1] = color[0,0,1]
    color_window[:,:,2] = color[0,0,2]

    canvas[x_0:x_1, y_0:y_1,:] = merge(canvas[x_0:x_1, y_0:y_1,:], color_window, brush)
    return canvas




def display(canvas, name="painting"):
    cv2.imshow(name, canvas)
    return not (cv2.waitKey(1) & 0xFF == ord('q'))

def display_until(canvas, name="tmp"):
    while display(canvas, name):
        pass


def disp_contour(img, con):
    di = cv2.dilate(con, np.ones((3,3), np.uint8), iterations=1)

    white = np.ones(img.shape, np.uint8)*255

    mix = overlay(img, white, di)
    mix = cv2.bitwise_and(mix, mix, mask=cv2.bitwise_not(con))
    
    display_until(mix)
    
    
def make_slic(img, region_size=int(10), ruler=10.0):
    """
    SLIC
    https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html
    """
        
    sp = cv2.ximgproc.createSuperpixelSLIC(img, region_size=region_size, ruler=ruler)
    for _ in range(5):
        sp.iterate()
    return sp

def disp_slic(img, region_size=10, ruler=10.0):
    sp = make_slic(img, region_size, ruler)
    con = sp.getLabelContourMask()

    disp_contour(img, con)

def make_lsc(img, region_size=10, ratio=0.075):
    """
    NOTE: This works better than slic it seems

    Linear Spectral Clustering
    https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html
    """
    sp = cv2.ximgproc.createSuperpixelLSC(img, region_size=region_size, ratio=ratio)
    for _ in range(5):
        sp.iterate()
    return sp

def disp_lsc(img, region_size=10, ratio=0.075):
    sp = make_lsc(img, region_size, ratio)
    disp_contour(img, sp.getLabelContourMask())
