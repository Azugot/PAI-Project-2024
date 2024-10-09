from tkinter import Label, filedialog, Canvas, Button, Tk
from PIL import Image, ImageTk
import numpy as np 
import cv2
import scipy.io

class CropApp:
    def __init__(self, root, uiWidth=434, uiHeight=636):
        self.mat_file_is_open = False
        self.num_pacient = 0
        self.img_pacient = 0
        self.mat_file = None
        
        self.uiWidth = uiWidth
        self.uiHeight = uiHeight
        self.mask = np.ones((uiWidth, uiHeight))
        self.image = None
        self.image_for_mask_multiplication = None
        self.lasx, self.lasy = 0, 0
        self.img = None

        self.app = root
        self.app.title('CROP')
        self.app.geometry('700x700')
        self.title = Label(self.app, text='CROP THE IMAGE', font='arial 30 bold', fg='#068481')
        self.title.pack()

        self.image_area = Canvas(self.app, width=uiWidth, height=uiHeight, bg='#C8C8C8')
        self.image_area.pack(pady=(10,0))

        self.open_image = Button(self.app, width=20, text='OPEN IMAGE', font='none 12', command=self.openAndPut)
        self.open_image.pack(pady=(10,5))
        
        self.open_mat = Button(self.app, width=20, text='OPEN MAT DATASET', font='none 12', command=self.read_mat_files)
        self.open_mat.pack(pady=(10,5))
        
        self.next_pacient = Button(self.app, width=20, text='NEXT PACIENT', font='none 12', command=self.next_mat_pacient)
        self.next_pacient.pack(pady=(10,5))
        
        self.previous_pacient = Button(self.app, width=20, text='PREVIOUS PACIENT', font='none 12', command=self.previous_mat_pacient)
        self.previous_pacient.pack(pady=(10,5))

        self.next_pacient_image = Button(self.app, width=20, text='NEXT PACIENT IMAGE', font='none 12', command=self.next_mat_pacient_image)
        self.next_pacient_image.pack(pady=(10,5))
        
        self.previous_pacient_image = Button(self.app, width=20, text='PREVIOUS PACIENT IMAGE', font='none 12', command=self.previous_mat_pacient_image)
        self.previous_pacient_image.pack(pady=(10,5))

        self.crop_area = Button(self.app, width=20, text='SELECT AREA', font='none 12', command=self.select_area)
        self.crop_area.pack(pady=(0,5))

        self.show_area = Button(self.app, width=20, text='SHOW AREA', font='none 12', command=self.show_mask)
        self.show_area.pack(pady=(0,5))

        self.save_image = Button(self.app, width=20, text='SAVE IMAGE', font='none 12', command=self.save_image)
        self.save_image.pack()

    def openAndPut(self):
        self.mat_file_is_open = False
        self.mat_file = None
        path = filedialog.askopenfilename()
        if path:
            self.image = Image.open(path)
            self.image_for_mask_multiplication = Image.open(path)
            self.image = self.image.resize((self.uiWidth , self.uiHeight), Image.LANCZOS)
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
        im_floodfill = np.abs(im_floodfill-np.ones((self.uiWidth ,self.uiHeight))*255)
        return im_floodfill

    def show_mask(self):
        mask_3_channels = np.ones((self.uiHeight, self.uiHeight, 3)) 

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

    def read_mat_files(self, num_pacient=0, img_pacient=0):
        
        if not self.mat_file_is_open:
            path = filedialog.askopenfilename()
            self.mat_file = path
        else:
            path = self.mat_file

        if path:
            
            #flag to indicate that a .mat is open
            self.mat_file_is_open = True
            
            #load matrix into data variable
            data = scipy.io.loadmat(path)

            #get the data array
            data_array = data['data'] 

            #get the first input
            input = data_array[0, num_pacient]
            
            #get the images from the input
            mat_imagens = input['images']
            mat_image = mat_imagens[img_pacient]

            # Convert the image to a format usable by PIL
            mat_image = np.array(mat_image)

            # Convert the NumPy array to an Image
            pil_image = Image.fromarray(mat_image)

            # Resize if necessary
            pil_image = pil_image.resize((self.uiWidth, self.uiHeight), Image.LANCZOS)

            # Update the image references
            self.image = ImageTk.PhotoImage(pil_image)
            self.image_for_mask_multiplication = pil_image

            # Display the image in the Canvas widget
            self.image_area.create_image(0, 0, image=self.image, anchor='nw')


    def next_mat_pacient(self):
        if self.mat_file_is_open:
            self.read_mat_files(self.num_pacient+1, self.img_pacient)
            self.num_pacient += 1
            self.img_pacient = 0
            
    def next_mat_pacient_image(self):
        if self.mat_file_is_open:
            self.read_mat_files(self.num_pacient, self.img_pacient+1)
            self.img_pacient += 1
    
    def previous_mat_pacient(self):
        if self.mat_file_is_open:
            self.read_mat_files(self.num_pacient+1, self.img_pacient)
            self.num_pacient -= 1
            self.img_pacient = 0
            
    def previous_mat_pacient_image(self):
        if self.mat_file_is_open:
            self.read_mat_files(self.num_pacient, self.img_pacient+1)
            self.img_pacient -= 1
    

if __name__ == "__main__":
    root = Tk()
    app = CropApp(root, 636, 434)
    root.mainloop()
