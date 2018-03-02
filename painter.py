#!/usr/bin/python

import Tkinter as tk
from PIL import Image, ImageOps, ImageTk
import random
import IPython
import rrt
import numpy as np
import time
from numpy.random import choice


brush1 = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                   [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0., 0.0],
                   [0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                   [0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
                   [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.5, 0.0, 0.2, 0.0, 0.0],
                   [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0],
                   [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.3, 0.1, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0],
                   [0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.1, 0.0, 0.0, 0.2, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])

brush2 = np.array([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.4, 0.4, 0.3, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
                   [0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.3, 0.8, 0.0, 0.0, 0.7, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
                   [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
                   [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0, 0.0]])

brush3 = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                   [0.2, 0.3, 0.0, 0.0, 0.0],
                   [0.1, 0.2, 0.7, 0.0, 0.0],
                   [0.0, 0.0, 0.2, 0.3, 0.0],
                   [0.1, 0.0, 0.0, 0.0, 0.6]])

brush4 = np.array([[4.0]])





class AABB:
    def __init__(self, l, r, t, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.area = (t-b)*(r-l)

    def contains(self, p):
        x, y = p
        if x < self.l or x > self.r:
            return False
        if y < self.b or y > self.t:
            return False
        return True

def sample_box_with_target_area(area, l, r, t, b):
    sl = np.sqrt(area)
    aspect_ratio = 0.5 + random.random()
    x = random.random() * (r - sl - l) + l
    y = random.random() * (t - sl - b) + b

    w = sl * aspect_ratio
    h = sl / aspect_ratio

    return AABB(l=l+x, r=l+x+w, b=b+y, t=b+y+h)
    


class Painter:
    def __init__(self, image_path, width=800, height=600):

        self.root = tk.Tk()

                
        self.photo_img = ImageOps.fit(Image.open(image_path), (width, height), Image.ANTIALIAS)
        self.photo = self.photo_img.load()
        # self.new_img = ImageTk.PhotoImage(Image.open(image_path))
        self.canvas_img = Image.new('RGB', (width, height))
        self.canvas = self.canvas_img.load()
        # self.disp_img = ImageTk.PhotoImage(Image.new('RGB', (width, height)))
        self.disp_img = ImageTk.PhotoImage(self.canvas_img)
        self.width = width
        self.height = height




        dims = np.array([width, height])
        self.rrt = rrt.RRT(dims/2, (0,0), dims)

                
        self.panel = tk.Label(self.root, image=self.disp_img)
        self.panel.pack(side="bottom", fill="both", expand="yes")


        self.s_in_stroke = False
        self.s_in_box = False

        self.brush_pos = None
        self.brush_goal = None
        self.brush_dir = None
        self.brush_color = None

        self.pixels_of_cur_box_filled_in = 0
        self.target_area = self.width*self.height/4
        # self.target_area = 100

        self.count_at_current = 0

        self.pixels_filled_in_iter = 0

        self.active_brush = brush1
        # IPython.embed()



    def clipx(self, x):
        return min(max(int(x), 0), self.width-1)
    
    def clipy(self, y):
        return min(max(int(y), 0), self.height-1)

    def to_pxl(self, point):
        x = self.clipx(point[0])
        y = self.clipy(point[1])
        return (x,y)

    def in_grid(self, p):
        x, y = p
        if x < 0 or x >= self.width:
            return False
        if y < 0 or y >= self.height:
            return False
        return True



    def add_brush_blob_at(self, xy, brush, color):

        for bx in range(brush.shape[0]):
            for by in range(brush.shape[1]):
                bw = brush[bx, by]
                if bw == 0.0:
                    continue
                p = (xy[0] + bx, xy[1] + by)
                if not self.in_grid(p):
                    continue
                cur_color = self.canvas[p]
                a = 1-bw
                self.canvas[p] = tuple([int(a*c + (1-a)*n) for c,n in zip(cur_color, color)])
                self.pixels_of_cur_box_filled_in += 1
                self.pixels_filled_in_iter += 1
                
        
    def add_stroke(self, endpoints):
        cur, end = endpoints
        delta = (end - cur) / np.linalg.norm(end - cur)

        while rrt.dist(cur, end) > 1:
            cur = cur + delta
            print cur
            self.add_brush_blob_at(cur, brush1, (0, 255, 0))
            # self.disp_img.paste(self.canvas_img)
            time.sleep(.1)
            self.disp_img.paste(self.canvas_img)

    

    def sample_from_box(self):
        box = self.box
        x = random.random() * (box.r - box.l) + box.l
        y = random.random() * (box.t - box.b) + box.b
        return self.to_pxl((x,y))

    def shrink_target_area(self):
        self.count_at_current += self.target_area

        if self.count_at_current > self.width * self.height/2:

            self.sample_brush()
            self.target_area *= .7
            self.target_area = max(self.target_area, 1)
            print "Shrinking to " + str(self.target_area)
            self.count_at_current = 0

    def sample_color(self):
        self.brush_color = self.photo[self.sample_from_box()]        

    def sample_box(self):
        # self.box = AABB(l=0, r=self.width, b=0, t=self.height)
        self.box = sample_box_with_target_area(self.target_area,
                                               l=0, r=self.width, b=0, t=self.height)
        self.shrink_target_area()
        self.sample_color()

        self.brush_dir = np.array([random.random(), random.random()])
        self.brush_dir = self.brush_dir / np.linalg.norm(self.brush_dir)
        self.s_in_box = True
        self.pixels_of_cur_box_filled_in = 0


    def sample_brush(self):
        i= 0
        for brush in [brush1, brush2, brush3, brush4]:

            i+=1
            if brush.size < self.box.area/10:
                self.active_brush = brush
                print "active brush: " + str(i)

                return
        self.active_brush = brush4


    def start_stroke(self):
        if self.pixels_of_cur_box_filled_in > 3*self.box.area:
            self.s_in_box = False
            return
        
        self.s_in_stroke = True
        self.brush_goal = self.rrt.sample()
        self.brush_pos = self.rrt.sample()
        

    def continue_stroke(self):
        # dist = np.linalg.norm(self.brush_goal - self.brush_pos)
        # if dist < 1:
        #     self.s_in_stroke = False
        #     return

        if not self.box.contains(self.brush_pos):
            self.s_in_stroke = False
            return
        
        delta = self.brush_dir
        # delta = (self.brush_goal - self.brush_pos) / dist
        self.brush_pos += delta
        self.add_brush_blob_at(self.brush_pos, self.active_brush, self.brush_color)
        
    def continue_state_machine(self):
        if self.s_in_stroke:
            self.continue_stroke()
            return

        if self.s_in_box:
            self.start_stroke()
            return

        self.sample_box()
            


    def update(self):
        # self.add_stroke((self.rrt.sample(), self.rrt.sample()))
        # for _ in range(100):
        while self.pixels_filled_in_iter < 10000 and\
              self.pixels_filled_in_iter < self.target_area/1:
            self.continue_state_machine()
        self.pixels_filled_in_iter = 0
        self.disp_img.paste(self.canvas_img)
        self.root.after(1, self.update)


    def run(self):
        self.root.after(1000, self.update)    
        self.root.wm_title("RRT Picture")
        self.root.mainloop()
        
# root.attributes("-fullscreen", True)






if __name__=="__main__":
    pic = Painter("Brad_with_victor.jpg")
    pic.run()
