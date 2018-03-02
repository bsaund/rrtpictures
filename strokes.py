#!/usr/bin/python

import Tkinter as tk
from PIL import Image, ImageOps, ImageTk
import random
import IPython
import rrt
import numpy as np

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

        self.weights = [[1]*self.height for _ in range(self.width)]


        dims = np.array([width, height])
        self.rrt = rrt.RRT(dims/2, (0,0), dims)


        

        # self.canvas = tk.Canvas(self.root, width=width, height=height, borderwidth=0, highlightthickness=0, bg="white")
        # self.canvas.grid()
                
        self.panel = tk.Label(self.root, image=self.disp_img)
        self.panel.pack(side="bottom", fill="both", expand="yes")
        
        # IPython.embed()



    def clipx(self, x):
        return min(max(int(x), 0), self.width-1)
    
    def clipy(self, y):
        return min(max(int(y), 0), self.height-1)

    def to_pxl(self, point):
        x = self.clipx(point[0])
        y = self.clipy(point[1])
        return (x,y)

    def update_pixel(self, coords, new_color):
        cur = self.new_img.getpixel(coords)
        w = self.weights[coords[0]][coords[1]]

        updated = (float(c*w + n)/(w+1) for c,n in zip(cur, new_color))

        updated = tuple([int(u)+1 if u-int(u) > random.random() else int(u) for u in updated])

        # print(updated)

        self.new_img.putpixel(coords, updated)
        self.weights[coords[0]][coords[1]] = min(1, 1 + w)

    def add_dot(self, coordinates, radius, pxl):
        x_c, y_c = coordinates

        for x in range(self.clipx(x_c-radius), self.clipx(x_c+radius)):
            for y in range(self.clipy(y_c-radius), self.clipy(y_c+radius)):
                if rrt.dist((x_c, y_c), (x,y)) < radius:
                    self.new_img.putpixel((x,y), pxl)


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



    def update(self):
        """Replots the LEDs circles
        Recalls itself every 100 millis
        """

        # num_updated = 0
        # while num_updated < 1000:
        #     num_updated += self.add_line_until_occupied((self.rrt.sample(), self.rrt.sample()))
            # self.add_line(self.rrt.step())
        #     self.add_dot((random.random()*self.width, random.random()*self.height), 2)

        self.add_stroke((self.rrt.sample(), self.rrt.sample()), 3)
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
