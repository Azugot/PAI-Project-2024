from tkinter import Label, filedialog, Canvas, Button, Tk
from PIL import Image, ImageTk
import numpy as np 
import cv2

class CropApp:
    def __init__(self, root):
        self.mask = np.ones((434, 636))
        self.image = None
        self.image_for_mask_multiplication = None
        self.lasx, self.lasy = 0, 0
        self.img = None

        self.app = root
        self.app.title('CROP')
        self.app.geometry('500x700')
        self.title = Label(self.app, text='CROP THE IMAGE', font='arial 30 bold', fg='#068481')
        self.title.pack()

        self.image_area = Canvas(self.app, width=434, height=636, bg='#C8C8C8')
        self.image_area.pack(pady=(10,0))

        self.open_image = Button(self.app, width=20, text='OPEN IMAGE', font='none 12', command=self.openAndPut)
        self.open_image.pack(pady=(10,5))

        self.crop_area = Button(self.app, width=20, text='SELECT AREA', font='none 12', command=self.select_area)
        self.crop_area.pack(pady=(0,5))

        self.show_area = Button(self.app, width=20, text='SHOW AREA', font='none 12', command=self.show_mask)
        self.show_area.pack(pady=(0,5))

        self.save_image = Button(self.app, width=20, text='SAVE IMAGE', font='none 12', command=self.save_image)
        self.save_image.pack()

    def openAndPut(self):
        path = filedialog.askopenfilename()
        if path:
            self.image = Image.open(path)
            self.image_for_mask_multiplication = Image.open(path)
            self.image = self.image.resize((434 , 636), Image.LANCZOS)
            self.image_for_mask_multiplication = self.image_for_mask_multiplication.resize((434, 636), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(self.image)
            self.image_area.create_image(0, 0, image=self.image, anchor='nw')
    
    def get_x_and_y(self, event):
        self.lasx, self.lasy = event.x, event.y

    def draw_smth(self, event):
        self.image_area.create_line((self.lasx, self.lasy, event.x, event.y), fill='red', width=3)
        self.lasx, self.lasy = event.x, event.y

        if 0 <= self.lasx < 500 and 0 <= self.lasy < 400:
            self.mask[self.lasy][self.lasx] = 0 
            self.mask[self.lasy+1][self.lasx+1] = 0 
            self.mask[self.lasy-1][self.lasx-1] = 0 
            self.mask[self.lasy+1][self.lasx-1] = 0 
            self.mask[self.lasy-1][self.lasx+1] = 0 

    def select_area(self):
        self.image_area.bind("<Button-1>", self.get_x_and_y)
        self.image_area.bind("<B1-Motion>", self.draw_smth)

    def retrun_shape(self, image_in):
        image = image_in
        gray = image_in
        edged = cv2.Canny(gray, 30, 200) 

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        cv2.drawContours(image, contours, -1, (0, 0, 0), 3)  
        th, im_th = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))
        im_floodfill = np.abs(im_floodfill-np.ones((434 ,636))*255)
        return im_floodfill

    def show_mask(self):
        mask_3_channels = np.ones((434, 636, 3)) 

        image_mattt = (self.mask * 255).astype(np.uint8)
        the_real_mask = self.retrun_shape(image_mattt)
        mask_3_channels[:,:,0] = the_real_mask/255
        mask_3_channels[:,:,1] = the_real_mask/255
        mask_3_channels[:,:,2] = the_real_mask/255

        real_area = np.array(self.image_for_mask_multiplication) * mask_3_channels
        real_area = Image.fromarray(np.uint8(real_area)).convert('RGB')
        
        self.img = real_area.convert("RGBA")
        datas = self.img.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        self.img.putdata(newData)
        self.img.show()

    def save_image(self):
        path_save = filedialog.asksaveasfilename()
        if path_save:
            self.img.save(str(path_save), "PNG")

if __name__ == "__main__":
    root = Tk()
    app = CropApp(root)
    root.mainloop()
