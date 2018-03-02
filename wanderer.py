#!/usr/bin/python

import Tkinter as tk
from PIL import Image, ImageOps, ImageTk
import random
import IPython
import rrt
import numpy as np
from numpy.random import choice

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, outline='', **kwargs)
tk.Canvas.create_circle = _create_circle

def to_two_digit_hex(num):
    h = hex(num)[2:]
    if len(h) == 1:
        h = '0' + h
    return h
        
def color_to_hex(color):
    cstr = '#'
    for c in color:
        cstr += to_two_digit_hex(c)
    return cstr
    

class RRTPicture:
    def __init__(self, image_path, width=800, height=600):

        self.root = tk.Tk()

                
        self.img = ImageOps.fit(Image.open(image_path), (width, height), Image.ANTIALIAS)
        # self.new_img = ImageTk.PhotoImage(Image.open(image_path))
        self.new_img = Image.new('RGB', (width, height))
        # self.disp_img = ImageTk.PhotoImage(Image.new('RGB', (width, height)))
        self.disp_img = ImageTk.PhotoImage(self.new_img)
        self.width = width
        self.height = height




        dims = np.array([width, height])
        self.rrt = rrt.RRT(dims/2, (0,0), dims)


        
                
        self.panel = tk.Label(self.root, image=self.disp_img)
        self.panel.pack(side="bottom", fill="both", expand="yes")

        self.wanderer_pos = np.array([int(c) for c in self.rrt.sample()])

        self.initial_brush_radius = 40
        self.wanderer_step = 12
        self.steps_per_tick = 10
        self.brush_radius = self.initial_brush_radius


        self.prev_fill_radius = [[self.initial_brush_radius]*self.height for _ in range(self.width)]        
        # IPython.embed()



    def clipx(self, x):
        return min(max(int(x), 0), self.width-1)
    
    def clipy(self, y):
        return min(max(int(y), 0), self.height-1)

    def to_pxl(self, point):
        x = self.clipx(point[0])
        y = self.clipy(point[1])
        return (x,y)

    def is_unfilled(self, x,y):
        return self.prev_fill_radius[x][y] == self.initial_brush_radius


    def add_dot(self, coordinates, radius, pxl):
        x_c, y_c = coordinates
        pxl = np.array(pxl)

        for x in range(self.clipx(x_c-radius-1), self.clipx(x_c+radius+1)):
            for y in range(self.clipy(y_c-radius), self.clipy(y_c+radius)):
                d = rrt.dist((x_c, y_c), (x,y))
                if d >= radius:
                    continue


                cur_pxl = np.array(self.new_img.getpixel((x,y)))
                true_pxl = np.array(self.img.getpixel((x,y)))

                
                a = min(4*float(radius-d)/radius , 1.0)

                # if self.is_unfilled(x,y):
                #     a = .5
                
                new_pxl = a*pxl + (1-a) * cur_pxl

                # b = .95
                # new_pxl = b*new_pxl + (1-b)*true_pxl
                
                new_pxl = [int(v) for v in new_pxl]
                self.new_img.putpixel((x,y), tuple(new_pxl))

                if d < float(radius)/4:
                    self.prev_fill_radius[x][y] = radius


    def add_stroke(self, endpoints, stroke_width):
        cur, end = endpoints
        delta = (end - cur) / np.linalg.norm(end - cur)
        x,y = self.to_pxl(cur)
        pxl = self.img.getpixel((x,y))

        while rrt.dist(cur, end) > 1:
            # update pixel
            self.add_dot(cur, stroke_width, pxl)
            # self.new_img.putpixel((x,y), pxl)
            cur = cur + delta
        

    def add_line(self, endpoints):
        cur, end = endpoints
        delta = (end - cur) / np.linalg.norm(end - cur)
        while rrt.dist(cur, end) > 1:
            # update pixel
            x,y = self.to_pxl(cur)
            pxl = self.img.getpixel((x,y))
            self.new_img.putpixel((x,y), pxl)
            cur = cur + delta

    def add_line_until_occupied(self, endpoints):
        cur, end = endpoints
        delta = (end - cur) / np.linalg.norm(end - cur)
        num_updated = 0
        x,y = self.to_pxl(cur)
        pxl = self.img.getpixel((x,y))

        while num_updated == 0 or rrt.dist(cur, end) > 1:
            # update pixel
            x,y = self.to_pxl(cur)
        
        

            existing_pxl = self.new_img.getpixel((x,y))
            already_occ = (existing_pxl[0] != 0 or existing_pxl[1] != 0)
            # self.new_img.putpixel((x,y), pxl)
            
            num_updated += 1
            
            cur = cur + delta
            if already_occ:
                return num_updated
        return num_updated

    def in_grid(self, p):
        x, y = p
        if x < 0 or x >= self.width:
            return False
        if y < 0 or y >= self.height:
            return False
        return True

    def get_neighbors(self, coord, step_size):
        coord = np.array([int(c) for c in coord])

        neighbors = [coord + np.array([step_size,0]),
                     coord + np.array([-step_size,0]),
                     coord + np.array([0,step_size]),
                     coord + np.array([0,-step_size])]


        return [n for n in neighbors if self.in_grid(n)]
                
    def get_color_weights(self, p0, ps):
        pxl0 = np.array(self.img.getpixel(tuple(p0)))
        rgb_dists = []
        for pn in ps:
            pxl = np.array(self.img.getpixel(tuple(pn)))
            color_dist = np.linalg.norm(pxl0 - pxl)
            fill_dist = self.prev_fill_radius[pn[0]][pn[1]]
            rgb_dists.append(color_dist + 1000 * 1.0/fill_dist)

        rgb_weights = [1/(d + 10) for d in rgb_dists]
        tot = sum(rgb_weights)
        return [w/tot for w in rgb_weights]

    def update_wanderer(self):
        cur = self.wanderer_pos
        x,y = cur
        step = int(np.ceil(self.prev_fill_radius[x][y]/2))
        # step = 1 + int(4*random.random())
        
        n = self.get_neighbors(cur, step)
        w = self.get_color_weights(cur, n)
        # IPython.embed()
        new_ind = choice(range(len(n)), p=w)
        new = n[new_ind]

        self.wanderer_pos = new

        pxl = self.img.getpixel(tuple(new))

        x,y = new

        new_brush_rad = float(self.prev_fill_radius[x][y])/2
        a = .9
        self.brush_radius = int(self.brush_radius*a + new_brush_rad*(1-a))

        self.add_dot(new, new_brush_rad, pxl)
        # self.add_dot(new, 20, pxl)


        


    def update(self):
        """Replots the LEDs circles
        Recalls itself every 100 millis
        """

        # num_updated = 0
        # while num_updated < 1000:
        #     num_updated += self.add_line_until_occupied((self.rrt.sample(), self.rrt.sample()))
            # self.add_line(self.rrt.step())
        #     self.add_dot((random.random()*self.width, random.random()*self.height), 2)

        # self.add_stroke((self.rrt.sample(), self.rrt.sample()), 3)
        for _ in range(10):
            self.update_wanderer()
        self.disp_img.paste(self.new_img)
        self.root.after(1, self.update)


    def run(self):
        self.root.after(10, self.update)    
        self.root.wm_title("RRT Picture")
        self.root.mainloop()
        
# root.attributes("-fullscreen", True)






if __name__=="__main__":
    pic = RRTPicture("Brad_with_victor.jpg")
    pic.run()
