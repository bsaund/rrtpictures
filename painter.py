#!/usr/bin/python

import cv2
import numpy as np
import time
import IPython
import tsp
import cProfile
import re
import os
import painting_utils as pu
from painting_utils import radial_brush, merge, paint_at







def select_fill_points(mask, num_points = 1):
    flat = mask.flatten().astype(np.double)
    inds = np.random.choice(range(len(flat)), num_points, p=flat/np.sum(flat),
                            replace=False)
    return np.array(np.unravel_index(inds, mask.shape)).transpose().tolist()
    # arr = np.array(np.where(mask > 0.5)).transpose().tolist()
    # return arr[::len(arr)/num_points]


def get_hsv_list(d_hue, d_value, d_sat):
    for value in range(0, 255, d_value):
        for hue in range(0, 255, d_hue):
            for saturation in range(0, 255, d_sat):
                # print (hue, saturation, value)
                yield [(hue, saturation, value),
                       (hue+d_hue, saturation + d_sat, value + d_value)]







def band_hsv(img, hue_band, sat_band, val_band):
    img[:,:,0] = img[:,:,0] / hue_band * hue_band
    img[:,:,1] = img[:,:,1] / sat_band * sat_band
    img[:,:,2] = img[:,:,2] / val_band * val_band
    return img



class Painter:
    def __init__(self, photo_filepath):
        self.photo_filename = photo_filepath
        self.photo= cv2.imread("pictures/" + photo_filepath)
        # print self.photo.shape
        self.photo = cv2.resize(self.photo, (1242, 932))
        # self.blurred_photo = cv2.GaussianBlur(self.photo, (55,55), 0)
        self.photo_hsv = cv2.cvtColor(self.photo, cv2.COLOR_BGR2HSV)
        self.create_blank_canvas()
        self.running = True
        self.painting_done = False
        
        self.region_countdown = 0
        self.active_region = None
        self.active_color = None
        self.region_dab_points = None
        self.hsv_bands = None
        self.brushes = None
        self.paint_iters = None
        self.pass_level = 1



    def create_blank_canvas(self):
        self.canvas = np.zeros(self.photo.shape, np.uint8)
        self.canvas[:,:,0] = 220
        self.canvas[:,:,1] = 245
        self.canvas[:,:,2] = 245


    def filter(self, lower, upper):
        mask = cv2.inRange(self.photo_hsv, lower, upper)
        # mask = cv2.GaussianBlur(mask, (17,17), 0)
        # mask = cv2.inRange(mask, 50, 255)

        # print "sum: ", np.sum(mask)/255
        if np.sum(mask)/255 < 200:
            return None, None
        
        fill_color = np.zeros((1,1,3), np.uint8)
        fill_color[0,0,0] = (lower[0]+upper[0])/2
        fill_color[0,0,1] = (lower[1]+upper[1])/2
        fill_color[0,0,2] = (lower[2]+upper[2])/2
        fill_color = cv2.cvtColor(fill_color, cv2.COLOR_HSV2BGR)


        mask = mask.astype(np.double)/255
        return mask, fill_color

    def set_new_pass_level(self, level):
        self.pass_level = level
        print("Setting pass level", level)
        if(level == 1):
            self.set_pass_level_1()
        if(level == 2):
            self.set_pass_level_2()
        if(level == 3):
            self.set_pass_level_3()
        if(level == 4):
            self.set_pass_level_4()
        if(level == 5):
            self.set_pass_level_5()
        if(level == 6):
            self.set_pass_level_6()

        return level <= 6
        # self.brushes = [radial_brush(30 + 2*i, 2000 + 100*i) for i in range(10)]

    def set_pass_level_1(self):
        self.brushes = [radial_brush(100, 10000, weight=0.05)]
        self.hsv_bands = get_hsv_list(d_hue=64, d_sat=128, d_value=128)
        self.paint_fraction = 1.0/400
        self.paint_iters = 1

    def set_pass_level_2(self):
        self.brushes = [radial_brush(50, 3000, weight=0.05)]
        self.hsv_bands = get_hsv_list(d_hue=32, d_sat=64, d_value=64)
        self.paint_fraction = 1.0/200
        self.paint_iters = 1

    def set_pass_level_3(self):
        self.brushes = [radial_brush(30, 400)]
        self.hsv_bands = get_hsv_list(d_hue=16, d_sat=64, d_value=64)
        self.paint_fraction = 1.0/30
        self.paint_iters = 1

    def set_pass_level_4(self):
        self.brushes = [radial_brush(10, 100, 0.05)]
        # self.hsv_bands = get_hsv_list(d_hue=16, d_sat=32, d_value=16)
        self.hsv_bands = get_hsv_list(d_hue=16, d_sat=64, d_value=64)
        self.paint_fraction = 1.0/20
        self.paint_iters = 10

    def set_pass_level_5(self):
        self.brushes = [radial_brush(5, 30, weight=0.05)]
        # self.hsv_bands = get_hsv_list(d_hue=16, d_sat=32, d_value=8)
        self.hsv_bands = get_hsv_list(d_hue=16, d_sat=64, d_value=64)
        self.paint_fraction = 1.0/2
        self.paint_iters = 100

    def set_pass_level_6(self):
        self.brushes = [radial_brush(1, 10, weight=0.2)]
        self.hsv_bands = get_hsv_list(d_hue=16, d_sat=64, d_value=64)
        self.paint_fraction = 1.0/2
        self.paint_iters = 100
        

    def get_new_region_of_interest(self):
        self.active_region = None
        while self.active_region is None:
            try:
                lower, upper = self.hsv_bands.next()
            except:
                if self.set_new_pass_level(self.pass_level+1):
                    continue
                return False
            self.active_region, self.active_color = self.filter(lower, upper)
        return True


    def sort_points(self):
        """
        Sorts points in an order that a human might choose to paint. 
        """
        self.region_dab_points.sort(key = lambda x: [x[1]/300, x[0]])
        return

        #More expensive TSP below
        
        d = self.region_dab_points
        mid = self.canvas.shape
        d.sort()
        # d.sort(key = lambda p: (p[0] - mid[0]/2)**2 + (p[1] - mid[1]/2)**2)
        # IPython.embed()
        size = 50
        chunks = [d[x:x+size] for x in range(0, len(d), size)]

        sorted_points = []
        

        for chunk in chunks:
            path = tsp.tsp(np.array(chunk))
            # path = path[1:]
            sorted_points += [chunk[p] for p in path]
            # IPython.embed()
        self.region_dab_points = sorted_points

    def get_region_of_interest(self):
        if self.region_countdown <= 0:
            if not self.get_new_region_of_interest():
                self.painting_done = True
                return None, None
            self.region_countdown = int(np.sum(self.active_region)*self.paint_fraction)
            self.region_dab_points = select_fill_points(self.active_region, self.region_countdown)
            self.region_countdown = len(self.region_dab_points)
            self.sort_points()
        return self.active_region, self.active_color
        

    def paint(self):
        region, color = self.get_region_of_interest()
        if self.painting_done:
            return

        # brush = np.random.choice(self.brushes)
        brush = self.brushes[0]

        
        for _ in range(self.paint_iters):
            if self.region_countdown <=0:
                break

            pos = self.region_dab_points[self.region_countdown-1]
                
            if(self.pass_level > 3):
                color = self.lookup_color(color, pos)

            self.canvas = paint_at(self.canvas, color, brush, pos)
            self.region_countdown -= 1
        # time.sleep(0.1)

    def lookup_color(self, default_color, pos):
        rgb_color = np.array([[self.photo[pos[0], pos[1], :]]])
        return rgb_color
    
    def setup_video_recorder(self):
        outfile = os.getcwd() + "/videos/" + os.path.splitext(self.photo_filename)[0] + ".mp4"
        self.out = cv2.VideoWriter(outfile,
                                   0x00000021, 30, (1242,932))

    def save_frame(self):
        self.out.write(self.canvas)

    def finish_video(self):
        self.out.release()

    def run(self, display=True, record=False):
        if record:
            self.setup_video_recorder()
        
        self.set_new_pass_level(1)

        while not self.painting_done and self.running:
            self.paint()

            if display:
                self.running = pu.display(self.canvas)

            if record:
                self.save_frame()


        if record:
            self.finish_video()
            
        print "Painting finished"
        while(display and self.running):
            self.running = pu.display(self.canvas)


def main():
    # fp = "goose.jpg"
    fp = "Brad_with_victor.jpg"
    # fp = "BradSaund.png"
    pic = Painter(fp)
    # pic = WIP("Brad_with_victor.jpg")

    pr = cProfile.Profile()
    pr.enable()

    pic.run(display=True, record=True)

    pr.disable()
    pr.print_stats(sort='time')

    

def wip():
    fp = "Brad_with_victor.jpg"
    pic = WIP(fp)
    pic.run()
    
        
if __name__=="__main__":
    main()
    # wip()
