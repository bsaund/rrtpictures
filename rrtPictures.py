#!/usr/bin/python

import Tkinter as tk
from PIL import Image, ImageOps, ImageTk
import random
import IPython

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


        

        # self.canvas = tk.Canvas(self.root, width=width, height=height, borderwidth=0, highlightthickness=0, bg="white")
        # self.canvas.grid()
                
        self.panel = tk.Label(self.root, image=self.disp_img)
        self.panel.pack(side="bottom", fill="both", expand="yes")
        
        # IPython.embed()




    # def plot_leds(canvas, led_colors):
    #     """Plots the 2d led_colors array as circles in a grid on a canvas"""
    #     canvas.delete("all")
    #     y_off = 100
        
    #     for row in led_colors:
    #         x_off = 100
    #         for color in row:
    #             canvas.create_circle(x_off, y_off, 10, fill=color)
    #             x_off += 50
    #         y_off += 50

    def clipx(self, x):
        return min(max(int(x), 0), self.width-1)
    
    def clipy(self, y):
        return min(max(int(y), 0), self.height-1)

    def update_pixel(self, coords, new_color):
        cur = self.new_img.getpixel(coords)
        w = self.weights[coords[0]][coords[1]]

        updated = (float(c*w + n)/(w+1) for c,n in zip(cur, new_color))

        updated = tuple([int(u)+1 if u-int(u) > random.random() else int(u) for u in updated])

        # print(updated)

        self.new_img.putpixel(coords, updated)
        self.weights[coords[0]][coords[1]] = min(1, 1 + w)

    def add_dot(self, coordinates, radius):
        color = self.img.getpixel(coordinates)
        colorstr = color_to_hex(color)
        # print colorstr
        # self.canvas.create_circle(coordinates[0], coordinates[1], 2, fill=colorstr)
        x_c, y_c = coordinates
        px = self.img.getpixel((self.clipx(x_c), self.clipy(y_c)))
        for x in range(self.clipx(x_c-radius), self.clipx(x_c+radius)):
            for y in range(self.clipy(y_c-radius), self.clipy(y_c+radius)):
                self.update_pixel((x,y), px)


    def update(self):
        """Replots the LEDs circles
        Recalls itself every 100 millis
        """

        for i in range(1000):
            self.add_dot((random.random()*self.width, random.random()*self.height), 2)
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
