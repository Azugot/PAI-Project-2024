from tkinter import Label, filedialog, Canvas, Button, Tk
from PIL import Image, ImageTk
import numpy as np 
import cv2
import scipy.io

class CropApp:
    def __init__(self, root, uiWidth=434, uiHeight=636):
        # Variables to keep track of the different parameters that change during the use of the app
        self.numPatient = 0
        self.imgPatient = 0
        self.zoomLevel = 1
        self.matFile = None
        self.areaROI = None
        self.pathROI = None
        
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
        self.openImage = Button(self.app, width=20, text='OPEN IMAGE', font='none 12', command=self.readImage)
        self.openMat = Button(self.app, width=20, text='OPEN MAT DATASET', font='none 12', command=self.readMatFiles)
        
        # Buttons to navigate through the images
        self.nextPatient = Button(self.app, width=20, text='NEXT PATIENT', font='none 12', command=self.nextMatPatient)
        self.previousPatient = Button(self.app, width=20, text='PREVIOUS PATIENT', font='none 12', command=self.previousMatPatient)

        # Buttons to navigate through the patients
        self.nextPatientImage = Button(self.app, width=20, text='NEXT PATIENT IMAGE', font='none 12', command=self.nextMatPatientImage)
        self.previousPatientImage = Button(self.app, width=20, text='PREVIOUS PATIENT IMAGE', font='none 12', command=self.previousMatPatientImage)
        
        #ROI Related Buttons
        self.chooseRoi = Button(self.app, width=20, text='SELECT ROI', font='none 12', command=self.toggleROI)
        self.showArea = Button(self.app, width=20, text='SHOW ROI', font='none 12', command=self.showROI)
        self.saveSelectedROI = Button(self.app, width=20, text='SAVE ROI', font='none 12', command=self.saveROI)

        #Zoom Reset Button
        self.resetZoomButton = Button(self.app, width=20, text='RESET ZOOM', font='none 12', command=self.resetZoom)

        # Grid Layout
        self.imageArea.grid(row=0, column=0)
        self.showArea.grid(row=1, column=0)
        self.chooseRoi.grid(row=2, column=0)
        self.saveSelectedROI.grid(row=3, column=0)
        self.resetZoomButton.grid(row=4, column=0)
        
        self.previousPatientImage.grid(row=0, column=1)
        self.nextPatientImage.grid(row=0, column=2)
        
        self.previousPatient.grid(row=1, column=1)
        self.nextPatient.grid(row=1, column=2)
        
        self.openMat.grid(row=3, column=1)
        self.openImage.grid(row=4, column=1)
        
    def readImage(self):
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

    def readMatFiles(self):
        
        self.path = filedialog.askopenfilename()
        self.matFile = self.path
        self.imgPatient = 0
        self.numPatient = 0

        if (self.path):
            
            # Flag to indicate that a .mat is open
            self.matFileIsOpen = True
            
            # Load matrix into data variable
            data = scipy.io.loadmat(self.path)

            dataArray = data['data'] 
            input = dataArray[0, self.numPatient]
            matImagens = input['images']
            matImage = matImagens[self.imgPatient]

            matImage = np.array(matImage)
            pilImage = Image.fromarray(matImage)

            pilImage = pilImage.resize((self.uiWidth, self.uiHeight), Image.LANCZOS)

            # Update the image references
            self.image = ImageTk.PhotoImage(pilImage)
            self.imageForMaskMultiplication = pilImage

            # Display the image in the Canvas widget
            self.imageArea.create_image(0, 0, image=self.image, anchor='nw')

    def navigateThroughMatFile(self, numPatient,imgPatient):
        if(self.matFileIsOpen and self.path):
            # Load matrix into data variable
            data = scipy.io.loadmat(self.path)

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

    #TODO: Implement this method
    def saveROI(self):
        pass

    def deleteROIarea(self):
        if (self.areaROI):
            self.imageArea.delete(self.areaROI)
            self.areaROI = None

    def toggleROI(self):
        if (self.roiOn):
            self.roiOn = False
            self.chooseRoi.config(text="SELECT ROI")
            self.imageArea.unbind("<Button-1>")
            self.imageArea.unbind("<B1-Motion>")
            self.deleteROIarea()
        else:
            self.roiOn = True
            self.chooseRoi.config(text="END SELECT ROI")
            self.imageArea.bind("<Button-1>", self.startDrawROI)
            self.imageArea.bind("<B1-Motion>", self.finishDrawROI)
           
    def startDrawROI(self, event):
        self.startX = event.x
        self.startY = event.y
        self.deleteROIarea()
        self.areaROI = self.imageArea.create_rectangle(self.startX, self.startY, self.startX, self.startY, outline="red", width=2)
           
    def finishDrawROI(self, event):
        if (self.roiOn and self.areaROI):
            self.imageArea.coords(self.areaROI, self.startX, self.startY, event.x, event.y)

    def drawROIFixed(self, event):
        if (self.roiOn):
            self.deleteROIarea()

            self.startX = event.x
            self.startY = event.y

            self.areaROI = self.imageArea.create_rectangle(self.startX-14, self.startY-14, self.startX+14, self.startY+14, outline="red", width=2)
    
    #TODO: Fix this method for generic Images   
    def showROI(self): 
        if self.areaROI:
            
            self.imageArea.itemconfig(self.areaROI, outline = "green")
            
            x1, y1, x2, y2 = self.imageArea.coords(self.areaROI)
            x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(self.uiWidth, x2), min(self.uiHeight, y2)])
           
            #Need this to make it work for "Normal" images or the multiplication gets broken and the ROI gets too
            
            #Gets the fucking scale right
            originalX, originalY = self.imageForMaskMultiplication.size
            scaleX = originalX / self.uiWidth
            scaleY = originalY / self.uiHeight

            # Convert coordinates to make it right
            correctedX1 = int(x1 * scaleX)
            correctedY1 = int(y1 * scaleY)
            correctedX2 = int(x2 * scaleX)
            correctedY2 = int(y2 * scaleY)

            cutROI = self.imageForMaskMultiplication.crop((correctedX1, correctedY1, correctedX2, correctedY2))

            cutROI.show()

    def nextMatPatient(self):
        if (self.matFileIsOpen):
            if self.numPatient >= 54:
                self.numPatient = 0
            else:
                self.numPatient += 1
            
            self.imgPatient = 0
            
            print("Patient: ", self.numPatient)
            print("Image: ", self.imgPatient)
            
            if (self.imgPatient >= 0 and self.imgPatient <= 54):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)
    
    def previousMatPatient(self):
        if (self.matFileIsOpen):
            if self.numPatient <= 0:
                self.numPatient = 54
            else:
                self.numPatient -= 1
            
            self.imgPatient = 0
                        
            print("Patient: ", self.numPatient)
            print("Image: ", self.imgPatient)            
            
            if (self.imgPatient >= 0 and self.imgPatient <= 54):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)
            
    def nextMatPatientImage(self):
        if (self.matFileIsOpen):
            if (self.imgPatient >= 9):
                self.imgPatient = 0
            else:
                self.imgPatient += 1
            
            print("Patient: ", self.numPatient)
            print("Image: ", self.imgPatient)
            
            if (self.imgPatient >= 0 and self.imgPatient <= 9):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)
              
    def previousMatPatientImage(self):
        if (self.matFileIsOpen):
            if (self.imgPatient <= 0):
                self.imgPatient = 9
            else:
                self.imgPatient -= 1
            
            print("Patient: ", self.numPatient)
            print("Image: ", self.imgPatient)
            
            if (self.imgPatient >= 0 and self.imgPatient <= 9):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)
   
    def resetZoom(self):
       self.zoomLevel = 1
    

if __name__ == "__main__":
    root = Tk()
    app = CropApp(root, 636, 434)
    root.mainloop()
