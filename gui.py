import PIL as PIL
from PIL import Image
from PIL import ImageTk
import numpy as np
from Tkinter import *

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
		curX = self.canvas.canvasx(event.x)
		curY = self.canvas.canvasy(event.y)

		w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

		"""
		if event.x > 0.9*w:
		    self.canvas.xview_scroll(1, 'units')
		elif event.x < 0.1*w:
		    self.canvas.xview_scroll(-1, 'units')
		if event.y > 0.9*h:
		    self.canvas.yview_scroll(1, 'units')
		elif event.y < 0.1*h:
		    self.canvas.yview_scroll(-1, 'units')
		"""
		# expand rectangle as you drag the mouse
		self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

	def on_button_release(self, event):
		pass

	def removeDefect(self):
		pass

	def finished(self):
		pass

	def __init__(self,master):
		Frame.__init__(self,master=None)
		self.x = self.y = 0
		self.canvas = Canvas(self,cursor="cross",width=800,height=500,confine=True)

		#buttons
		self.removeButton = Button(self,text='Remove',command=self.removeDefect)
		self.removeButton.grid(row=1,column=0)

		self.finishButton = Button(self,text='Finished',command=self.finished)
		self.finishButton.grid(row=2,column=0)

		self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
		self.canvas.bind("<ButtonPress-1>", self.on_button_press)
		self.canvas.bind("<B1-Motion>", self.on_move_press)
		self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

		self.rect = None
		self.start_x = None
		self.start_y = None

		self.im = PIL.Image.open("images/set1/20190317153942_IMG_0234.jpg").resize((800,500))

		#scale images
		#self.width = 800
		#self.height = 500
		#self.scaleW = self.width/self.im.size[0]
		#self.scaleH = self.height/self.im.size[1]
		#self.im.resize((800,500))

		#self.wazil,self.lard=self.im.size
		#self.canvas.config(scrollregion=canvas.bbox(ALL))
		self.tk_im = ImageTk.PhotoImage(self.im)
		#self.tk_im.zoom(self.scaleW,self.scaleH)
		self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)

if __name__ == "__main__":
	root=Tk()
	root.geometry('900x600')
	root.title('Holiday Snap Editor')
	app = GUI(root)
	app.pack()
	app.pack_propagate(0)
	root.mainloop()
