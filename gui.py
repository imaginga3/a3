import PIL
from PIL import Image
from PIL import ImageTk
import numpy as np
from Tkinter import *
import stillImages
import movingObjects

# NOTE: IMAGES MUST BE 800X528

# Based upon Resource:
# https://stackoverflow.com/questions/29789554/tkinter-draw-rectangle-using-a-mouse
# User: userfhdrsdg

class GUI(Frame):

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        # Set the x and y locations to be passed to inpainting function
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        pass

    def removeDefect(self):
        # [x1, y1, x2, y2]
        rect = np.array(self.canvas.coords(self.rect), copy=True, dtype=np.float64)
        # Ensure the rectangle does not go out of bounds of image
        rect[0] = np.clip(rect[0], a_min=0, a_max=self.width) # x1
        rect[1] = np.clip(rect[1], a_min=0, a_max=self.height) # y1
        rect[2] = np.clip(rect[2], a_min=0, a_max=self.width) # x2
        rect[3] = np.clip(rect[3], a_min=0, a_max=self.height) # y2
        region = np.array(rect, copy=True, dtype=np.uint16)
        # Make sure that (x1,y1) is less than (x2,y2)
        if region[0] > region[2]:
            t = region[0]
            region[0] = region[2]
            region[2] = t
        if region[1] > region[3]:
            t = region[1]
            region[1] = region[3]
            region[3] = t
        # Apply the inpainting
        stillImages.removeStillObject(output_img,region,output_img)
        # Reload the new image onto the tkinter canvas
        self.im = PIL.Image.open(output_img)
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_im)
    def finished(self):
        root.destroy()

    def __init__(self,master):
        movingObjects.reconstruct(moving_dir,output_img)
        Frame.__init__(self,master=None)
        # Setup the frame size of the image
        self.x = self.y = 0
        self.width = 800
        self.height = 528
        self.canvas = Canvas(self,cursor="cross",width=self.width,height=self.height,confine=True)
        #buttons
        self.removeButton = Button(self,text='Remove',command=self.removeDefect)
        self.removeButton.grid(row=1,column=0)
        self.finishButton = Button(self,text='Finished',command=self.finished)
        self.finishButton.grid(row=2,column=0)
        # Setup the canvas for the image to be displayed, and the button bindings
        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        # Set global variables to be used later
        self.rect = None
        self.start_x = None
        self.start_y = None
        # Add the image to the canvas
        self.im = PIL.Image.open(output_img)
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.image_on_canvas = self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)

moving_dir = ""
output_img = ""

if __name__ == "__main__":
    moving_dir = sys.argv[1]
    output_img = sys.argv[2]
    root=Tk()
    root.geometry('900x600')
    root.title('Holiday Snap Editor')
    app = GUI(root)
    app.pack()
    app.pack_propagate(0)
    root.mainloop()
