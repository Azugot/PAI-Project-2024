from tkinter import Label, filedialog, Canvas, Button, Tk
from PIL import Image, ImageTk
import numpy as np 
import cv2
import scipy.io

class CropApp:
    def __init__(self, root, uiWidth=434, uiHeight=636):
        # Variables to keep track of the current patient and image being displayed and the .mat file being used
        self.numPatient = 0
        self.imgPatient = 0
        self.matFile = None
        self.areaROI = None
        
        # Flags and Boolean variables to be used as flip-flops
        self.roiOn = False
        self.matFileIsOpen = False
        
        # UI Configs
        self.uiWidth = uiWidth
        self.uiHeight = uiHeight
        self.mask = np.ones((uiWidth, uiHeight))
        self.image = None
        self.imageForMaskMultiplication = None
        self.lasX, self.lasY = 0, 0
        self.img = None

        self.app = root
        self.app.title('CROP')
        self.app.geometry('700x700')

        # Canvas to display the image
        self.imageArea = Canvas(self.app, width=uiWidth, height=uiHeight, bg='#C8C8C8')

        # B U T T O N S

        # Button to open Image/Mat file
        self.openImage = Button(self.app, width=20, text='OPEN IMAGE', font='none 12', command=self.openAndPut)
        self.openMat = Button(self.app, width=20, text='OPEN MAT DATASET', font='none 12', command=self.readMatFiles)
        
        self.nextPatient = Button(self.app, width=20, text='NEXT PATIENT', font='none 12', command=self.nextMatPatient)
        self.previousPatient = Button(self.app, width=20, text='PREVIOUS PATIENT', font='none 12', command=self.previousMatPatient)

        self.nextPatientImage = Button(self.app, width=20, text='NEXT PATIENT IMAGE', font='none 12', command=self.nextMatPatientImage)
        self.previousPatientImage = Button(self.app, width=20, text='PREVIOUS PATIENT IMAGE', font='none 12', command=self.previousMatPatientImage)

        self.showArea = Button(self.app, width=20, text='SHOW AREA', font='none 12', command=self.showMask)
        
        self.chooseRoi = Button(self.app, width=20, text='SELECT ROI', font='none 12', command=self.toggleRoi)

        #self.saveSelectedROI = Button(self.app, width=20, text='SAVE ROI', font='none 12', command=self.saveROI)


        # Grid Layout
        self.imageArea.grid(row=0, column=0)
        self.showArea.grid(row=1, column=0)
        self.chooseRoi.grid(row=2, column=0)
        
        self.previousPatientImage.grid(row=0, column=1)
        self.nextPatientImage.grid(row=0, column=2)
        
        self.previousPatient.grid(row=1, column=1)
        self.nextPatient.grid(row=1, column=2)
        
        self.openMat.grid(row=3, column=1)
        self.openImage.grid(row=4, column=1)
        #self.saveSelectedROI.grid()
        
    def openAndPut(self):
        self.matFileIsOpen = False
        self.matFile = None
        path = filedialog.askopenfilename()
        if path:
            self.image = Image.open(path)
            self.imageForMaskMultiplication = Image.open(path)
            self.image = self.image.resize((self.uiWidth , self.uiHeight), Image.LANCZOS)
            self.imageForMaskMultiplication = self.imageForMaskMultiplication.resize((434, 636), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(self.image)
            self.imageArea.create_image(0, 0, image=self.image, anchor='nw')
    
        if 0 <= self.lasX < 500 and 0 <= self.lasY < 400:
            self.mask[self.lasY][self.lasX] = 0 
            self.mask[self.lasY+1][self.lasX+1] = 0 
            self.mask[self.lasY-1][self.lasX-1] = 0 
            self.mask[self.lasY+1][self.lasX-1] = 0 
            self.mask[self.lasY-1][self.lasX+1] = 0 

    def returnShape(self, imageIn):
        image = imageIn
        gray = imageIn
        edged = cv2.Canny(gray, 30, 200) 

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        cv2.drawContours(image, contours, -1, (0, 0, 0), 3)  
        th, imTh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        imFloodfill = imTh.copy()
        h, w = imTh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(imFloodfill, mask, (0,0), (255,255,255))
        imFloodfill = np.abs(imFloodfill-np.ones((self.uiWidth ,self.uiHeight))*255)
        return imFloodfill

    def showMask(self):
        mask3Channels = np.ones((self.uiHeight, self.uiHeight, 3)) 

        imageMattt = (self.mask * 255).astype(np.uint8)
        theRealMask = self.returnShape(imageMattt)
        mask3Channels[:,:,0] = theRealMask/255
        mask3Channels[:,:,1] = theRealMask/255
        mask3Channels[:,:,2] = theRealMask/255

        realArea = np.array(self.imageForMaskMultiplication) * mask3Channels
        realArea = Image.fromarray(np.uint8(realArea)).convert('RGB')
        
        self.img = realArea.convert("RGBA")
        datas = self.img.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        self.img.putdata(newData)
        self.img.show()

    def readMatFiles(self, numPatient=0, imgPatient=0):
        
        if not self.matFileIsOpen:
            path = filedialog.askopenfilename()
            self.matFile = path
        else:
            path = self.matFile

        if path:
            
            # Flag to indicate that a .mat is open
            self.matFileIsOpen = True
            
            # Load matrix into data variable
            data = scipy.io.loadmat(path)

            # Get the data array
            dataArray = data['data'] 

            # Get the first input
            input = dataArray[0, numPatient]
            
            # Get the images from the input
            matImagens = input['images']
            matImage = matImagens[imgPatient]

            # Convert the image to a format usable by PIL
            matImage = np.array(matImage)

            # Convert the NumPy array to an Image
            pilImage = Image.fromarray(matImage)

            # Resize if necessary
            pilImage = pilImage.resize((self.uiWidth, self.uiHeight), Image.LANCZOS)

            # Update the image references
            self.image = ImageTk.PhotoImage(pilImage)
            self.imageForMaskMultiplication = pilImage

            # Display the image in the Canvas widget
            self.imageArea.create_image(0, 0, image=self.image, anchor='nw')

    def deleteROIarea(self):
        if self.areaROI:
            self.imageArea.delete(self.areaROI)
            self.areaROI = None

    def toggleRoi(self):
        if self.roiOn:
            self.roiOn = False
            self.chooseRoi.config(text="SELECT ROI")
            self.imageArea.unbind("<Button-1>")
            
        else:
            self.roiOn = True
            self.chooseRoi.config(text="END SELECT ROI")
            self.imageArea.bind("<Button-1>", self.drawRoi)
            self.deleteROIarea()

            
    def drawRoi(self, event):
        if self.roiOn:
            self.deleteROIarea()
            
            self.startX = event.x
            self.startY = event.y
            
            self.areaROI = self.imageArea.create_rectangle(self.startX-14, self.startY-14, self.startX+14, self.startY+14, outline="red", width=2)
        

    def selectRoi(self, event):
        self.startX = event.x
        self.startY = event.y
        
        self.cutRoi()
        
    def cutRoi(self):
        roi = self.imageForMaskMultiplication.crop
        ((self.startX, self.startY, 
        self.startX+28, self.startY+28))
        
        roi.show()

    def nextMatPatient(self):
        if self.matFileIsOpen:
            self.readMatFiles(self.numPatient+1, self.imgPatient)
            self.numPatient += 1
            self.imgPatient = 0
            
    def nextMatPatientImage(self):
        if self.matFileIsOpen:
            self.readMatFiles(self.numPatient, self.imgPatient+1)
            self.imgPatient += 1
    
    def previousMatPatient(self):
        if self.matFileIsOpen:
            self.readMatFiles(self.numPatient+1, self.imgPatient)
            self.numPatient -= 1
            self.imgPatient = 0
            
    def previousMatPatientImage(self):
        if self.matFileIsOpen:
            self.readMatFiles(self.numPatient, self.imgPatient+1)
            self.imgPatient -= 1
    

if __name__ == "__main__":
    root = Tk()
    app = CropApp(root, 636, 434)
    root.mainloop()
