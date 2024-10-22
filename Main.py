from tkinter import Label, filedialog, Canvas, Button, Tk, Toplevel, Frame, Scrollbar, Canvas
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np 
import cv2
import os
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skimage.feature import graycomatrix, graycoprops as greycoprops
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

class CropApp:
    def __init__(self, root, savePath, uiWidth=434, uiHeight=636):
        # Variables to keep track of the different parameters that change during the use of the app
        self.numPatient = 0
        self.imgPatient = 0
        self.zoomLevel = 1
        self.moveX = 0 
        self.moveY = 0
        self.savePath = savePath
        self.matFile = None
        self.areaROI = None
        self.pathROI = None
        
        # Flags and Boolean variables to be used as flip-flops
        self.roiOn = False
        self.matFileIsOpen = False
        self.zoomEnabled = False
        
        # UI Configs
        self.uiWidth = uiWidth
        self.uiHeight = uiHeight
        self.mask = np.ones((uiWidth, uiHeight))
        self.image = None
        self.imageForMaskMultiplication = None
        self.hist_img_tk = None
        self.lasX, self.lasY = 0, 0

        self.app = root
        self.app.title('CROP')
        self.app.geometry('700x700')

        # Frame para a imagem
        self.imageFrame = tk.Frame(self.app, width=uiWidth, height=uiHeight, bg='#C8C8C8')
        self.imageFrame.grid(row=0, column=0, padx=10, pady=10)

        # Frame para o histograma
        self.histFrame = tk.Frame(self.app, width=uiWidth, height=uiHeight, bg='#E8E8E8')
        self.histFrame.grid(row=0, column=1, padx=5, pady=5)

        # Canvas para a imagem dentro do frame de imagem
        self.imageArea = Canvas(self.imageFrame, width=uiWidth, height=uiHeight, bg='#C8C8C8')
        self.imageArea.pack()

        # Canvas para o histograma - ajuste o tamanho aqui
        self.histCanvas = Canvas(self.histFrame, width=uiWidth, height=uiHeight, bg='#E8E8E8')  # Aumente width e height conforme necessário
        self.histCanvas.pack()

        # B U T T O N S

        # Button to open Image/Mat file
        self.openImage = Button(self.app, width=20, text='OPEN IMAGE', font='none 12', command=self.readImage)
        self.openMat = Button(self.app, width=20, text='OPEN MAT DATASET', font='none 12', command=self.readMatFiles)
        
        # Buttons to navigate through the images (initially hidden)
        self.nextPatient = Button(self.app, width=20, text='NEXT PATIENT', font='none 12', command=self.nextMatPatient)
        self.previousPatient = Button(self.app, width=20, text='PREVIOUS PATIENT', font='none 12', command=self.previousMatPatient)
        self.nextPatientImage = Button(self.app, width=20, text='NEXT PATIENT IMAGE', font='none 12', command=self.nextMatPatientImage)
        self.previousPatientImage = Button(self.app, width=20, text='PREVIOUS PATIENT IMAGE', font='none 12', command=self.previousMatPatientImage)
        self.viewGLCMRadialButton = Button(self.app, width=20, text='VIEW GLCM RADIAL', font='none 12', command=self.viewGLCMRadial)
        
        # ROI Related Buttons (initially hidden)
        self.chooseRoi = Button(self.app, width=20, text='SELECT ROI', font='none 12', command=self.toggleROI)
        self.showArea = Button(self.app, width=20, text='SHOW ROI', font='none 12', command=self.showROI)
        self.saveSelectedROI = Button(self.app, width=20, text='SAVE ROI', font='none 12', command=self.saveROI)
        self.viewHistogramsButton = Button(self.app, width=20, text='VIEW HISTOGRAMS', font='none 12', command=self.viewHistograms)
        self.viewGLCMButton = Button(self.app, width=20, text='VIEW GLCM & TEXTURE', font='none 12', command=self.viewGLCM)
        self.viewNTDescriptorButton = Button(self.app, width=20, text='VIEW NT DESCRIPTOR', font='none 12', command=self.viewNTDescriptorMatricula)
        
        # Zoom Reset Button (initially hidden)
        self.resetZoomButton = Button(self.app, width=20, text='RESET ZOOM', font='none 12', command=self.resetZoom)
        self.zoomOutButton = Button(self.app, width=20, text='-', font='none 12', command=self.zoomOut)
        self.zoomInButton = Button(self.app, width=20, text='+', font='none 12', command=self.zoomIn)
        self.toggleZoomButton = Button(self.app, width=20, text='ENABLE ZOOM', font='none 12', command=self.toggleZoom)
        
        # Grid Layout
        self.openImage.grid(row=1, column=0, sticky="n")
        self.openMat.grid(row=2, column=0, sticky="n")
        self.imageArea.grid(row=0, column=0, columnspan=3)
        
        self.imageArea.bind("<ButtonPress-2>", self.startMove)
        self.imageArea.bind("<B2-Motion>", self.moveImage)

    #M OSTRAR BOTOES DPS DE ENVIAR A IMAGEM
    def showAdditionalButtons(self):
        self.showArea.grid(row=3, column=0, sticky="n")
        self.chooseRoi.grid(row=4, column=0, sticky="n")
        self.toggleZoomButton.grid(row=4, column=1, sticky="n", padx=5)
        self.saveSelectedROI.grid(row=5, column=0, sticky="n")
        self.viewHistogramsButton.grid(row=7, column=0, sticky="n")
        self.viewGLCMButton.grid(row=8, column=0, sticky="n")
        self.viewNTDescriptorButton.grid(row=9, column=0, sticky="n")
        self.previousPatientImage.grid(row=11, column=0, sticky="n", padx=5)
        self.previousPatient.grid(row=12, column=0, sticky="n", padx=5)
        self.nextPatientImage.grid(row=11, column=1, sticky="n", padx=5)
        self.nextPatient.grid(row=12, column=1, sticky="n", padx=5)
        self.viewGLCMRadialButton.grid(row=13, column=0, sticky="n")
        self.resetZoomButton.grid(row=10, column=0, sticky="n")
        self.zoomInButton.grid(row=14, column=2, sticky="n", padx=5)
        self.zoomOutButton.grid(row=15, column=2, sticky="n", padx=5)

    def readImage(self):
        self.matFileIsOpen = False
        self.matFile = None
        path = filedialog.askopenfilename()
        if path:
            self.imageArea.delete("all")
            self.histCanvas.delete("all")
            self.image = Image.open(path)
            self.imageForMaskMultiplication = self.image.resize((self.uiWidth, self.uiHeight), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(self.imageForMaskMultiplication)
            self.imageArea.create_image(0, 0, image=self.image, anchor='nw')
            self.showHistogram(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            self.showAdditionalButtons()
    
    
        if 0 <= self.lasX < 500 and 0 <= self.lasY < 400:
            self.mask[self.lasY][self.lasX] = 0 
            self.mask[self.lasY+1][self.lasX+1] = 0 
            self.mask[self.lasY-1][self.lasX-1] = 0 
            self.mask[self.lasY+1][self.lasX-1] = 0 
            self.mask[self.lasY-1][self.lasX+1] = 0 


        if 0 <= self.lasX < 500 and 0 <= self.lasY < 400:
            self.mask[self.lasY][self.lasX] = 0 
            self.mask[self.lasY+1][self.lasX+1] = 0 
            self.mask[self.lasY-1][self.lasX-1] = 0 
            self.mask[self.lasY+1][self.lasX-1] = 0 
            self.mask[self.lasY-1][self.lasX+1] = 0

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

            self.showAdditionalButtons()
            self.showHistogram(matImage)

    def showHistogram(self, matImage):
        if len(matImage.shape) == 3: 
            matImage = cv2.cvtColor(matImage, cv2.COLOR_BGR2GRAY)

        height, width = matImage.shape
        total_pixels = height * width

        hist = np.zeros(256, dtype=int)
        for i in range(height):
            for j in range(width):
                intensity_value = matImage[i, j]
                hist[intensity_value] += 1

        assert sum(hist) == total_pixels, "A soma do histograma deve ser igual ao número total de pixels"

        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(hist, color='black')
        ax.set_title(f"Histogram - Paciente {self.numPatient}, Imagem {self.imgPatient}")
        ax.set_xlim([0, 255])
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Number of Pixels')

        max_value = np.percentile(hist, 95)
        ax.set_ylim([0, max_value * 1.1])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        hist_image = Image.open(buf)
        hist_width, hist_height = hist_image.size

        self.histCanvas.config(width=hist_width, height=hist_height)

        self.hist_img_tk = ImageTk.PhotoImage(hist_image)
        self.histCanvas.create_image(0, 0, image=self.hist_img_tk, anchor='nw')

        buf.close()
        plt.close(fig)


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
            self.showHistogram(matImage)

#Z O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 0 O O O O O O O O O O O MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM 
    def toggleZoom(self):
        if (self.zoomEnabled):
            self.toggleZoomButton.config(text="ENABLE ZOOM")
            self.zoomEnabled = False
            self.resetZoom()
        else:
            self.toggleZoomButton.config(text="DISABLE ZOOM")
            self.zoomEnabled = True
            #ROI STUFF
            self.roiOn = False
            self.chooseRoi.config(text="SELECT ROI")
            self.imageArea.unbind("<Button-1>")
            self.imageArea.unbind("<B1-Motion>")
            self.deleteROIarea()
    
    def zoomIn(self):
        if (self.zoomEnabled):
            self.zoomLevel += 0.1
            
            self.imageZoomUpdate()
            
    def zoomOut(self):
        if (self.zoomEnabled):
            self.zoomLevel -= 0.1
            
            self.imageZoomUpdate()
            
    def imageZoomUpdate(self):
        width, height = self.imageForMaskMultiplication.size
        new_width = int(width * self.zoomLevel)
        new_height = int(height * self.zoomLevel)
        resized_image = self.imageForMaskMultiplication.resize((new_width, new_height), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(resized_image)
        self.imageArea.create_image(self.moveX, self.moveY, image=self.image, anchor='nw')
    
    def resetZoom(self):
       self.zoomLevel = 1
       self.moveX, self.moveY = 0, 0
       self.imageZoomUpdate()

#M OOOOOOO VEMENT DA IMAGEM
    def startMove(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def moveImage(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.moveX += dx
        self.moveY += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.imageZoomUpdate()

#ROI RELATED METHODS
    def listSavedROIFiles(self):
        # Function to list all saved ROI files
        return [f for f in os.listdir(self.savePath) if os.path.isfile(os.path.join(self.savePath, f))]    

    def saveROI(self):
        cutROI = self.acquireROI()
        
        # Ensure the save path directory exists
        os.makedirs(self.savePath, exist_ok=True)

        # Save the image with a different alias
        if (self.numPatient < 10):
            file_name = os.path.join(self.savePath, f"ROI_0{self.numPatient}_{self.imgPatient}.png")
        else:
            file_name = os.path.join(self.savePath, f"ROI_{self.numPatient}_{self.imgPatient}.png")
        cutROI.save(file_name, "PNG")

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
            #self.imageArea.bind("<Button-1>", self.drawROIFixed)
            self.imageArea.bind("<Button-1>", self.startDrawROI)
            self.imageArea.bind("<B1-Motion>", self.finishDrawROI)
            #ZOOM STUFF
            self.toggleZoomButton.config(text="ENABLE ZOOM")
            self.zoomEnabled = False
            self.resetZoom()
            self.imageZoomUpdate()
           
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
    def acquireROI(self):    
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
        return (self.imageForMaskMultiplication.crop((correctedX1, correctedY1, correctedX2, correctedY2)))
    
    def showROI(self): 
        if self.areaROI:
            
            cutROI = self.acquireROI()

            cutROI.show()
            
            
    # S C R O L L
    def createScrollableCanvas(self, parentWindow):
        # Function to create a scrollable canvas
        canvas = Canvas(parentWindow)
        scrollbar = Scrollbar(parentWindow, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return scrollable_frame
    
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

    # H I S T O G R A M A
    def viewHistograms(self):
        # Cria uma nova janela
        histWindow = Toplevel(self.app)
        histWindow.title("Histograms of Saved ROIs")
        histWindow.geometry("1200x800")

        # Coloca um scroll
        scrollable_frame = self.createScrollableCanvas(histWindow)

        # Pega todos os arquivos salvos
        roiFiles = self.listSavedROIFiles()

        # Processa cada imagem
        for roiFile in roiFiles:
            roiPath = os.path.join(self.savePath, roiFile)
            roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE) 

            height, width = roiImage.shape
            total_pixels = height * width

            hist = np.zeros(256, dtype=int)
            for i in range(height):
                for j in range(width):
                    intensity_value = roiImage[i, j]
                    hist[intensity_value] += 1

            assert sum(hist) == total_pixels, "A soma do histograma deve ser igual ao número total de pixels"

            frame = Frame(scrollable_frame)
            frame.pack(pady=10)

            roiPIL = Image.fromarray(roiImage) 
            roiPIL = roiPIL.resize((200, 200), Image.LANCZOS)
            roiPhoto = ImageTk.PhotoImage(roiPIL)
            imgLabel = Label(frame, image=roiPhoto)
            imgLabel.image = roiPhoto
            imgLabel.pack(side='left', padx=10)

            # Plotar o histograma
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.set_title(f"Histogram for {roiFile}")
            ax.plot(hist, color='black')
            ax.set_xlim([0, 255])
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Number of Pixels')

            canvas_hist = FigureCanvasTkAgg(fig, master=frame)
            canvas_hist.draw()
            canvas_hist.get_tk_widget().pack(side='right', padx=10)


    # M A T R I Z
    def viewGLCM(self):
        # Cria uma janela nova
        glcmWindow = Toplevel(self.app)
        glcmWindow.title("GLCM & Texture Features of Saved ROIs")
        glcmWindow.geometry("1200x800")

        #Barra de rolagem
        scrollable_frame = self.createScrollableCanvas(glcmWindow)

        # Pega todos os arquivos salvos
        roiFiles = self.listSavedROIFiles()

        # Calcula a matriz GLCM
        for roiFile in roiFiles:
            roiPath = os.path.join(self.savePath, roiFile)
            roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)

            # Cria GLCM matrix
            glcm = graycomatrix(roiImage, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

            # Calcula a textura
            # Cacula a intensidade de contraste entre um pixel e seus vizinhos
            contrast = greycoprops(glcm, 'contrast')[0, 0]
            # Mede a dissimilaridade entre pixels
            dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
            # Mede a homogeneidade de valores altos indicam regiões uniformes
            homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            # Mede a uniformidade da textura
            energy = greycoprops(glcm, 'energy')[0, 0]
            # Mede a correlação linear entre pixels
            correlation = greycoprops(glcm, 'correlation')[0, 0]

            #Exibe as imagens e o resultado um do lado do outro
            frame = Frame(scrollable_frame)
            frame.pack(pady=10)

            # Mostra a ROI
            roiPIL = Image.fromarray(roiImage)
            roiPIL = roiPIL.resize((200, 200), Image.LANCZOS)
            roiPhoto = ImageTk.PhotoImage(roiPIL)
            imgLabel = Label(frame, image=roiPhoto)
            imgLabel.image = roiPhoto 
            imgLabel.pack(side='left', padx=10)

            # Mostra os textos
            featuresText = (
                f"Contrast: {contrast:.4f}\n"
                f"Dissimilarity: {dissimilarity:.4f}\n"
                f"Homogeneity: {homogeneity:.4f}\n"
                f"Energy: {energy:.4f}\n"
                f"Correlation: {correlation:.4f}"
            )

            # Alinhamentos
            featuresLabel = Label(frame, text=featuresText, font='none 12', justify='left')
            featuresLabel.pack(side='right', padx=10)   

    def viewNTDescriptorMatricula(self):
        # Calcula NT
        matriculas = [766639, 772198, 1378247]
        NT = sum(matriculas) % 4

        # Cria uma janela nova
        ntWindow = Toplevel(self.app)
        ntWindow.title("NT Descriptor of Saved ROIs")
        ntWindow.geometry("1200x800")

        # Barra de rolagem
        scrollable_frame = self.createScrollableCanvas(ntWindow)

        # Pega todos os arquivos salvos
        roiFiles = self.listSavedROIFiles()

        # Calcula a matriz GLCM
        for roiFile in roiFiles:
            roiPath = os.path.join(self.savePath, roiFile)
            roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)

            # Cria GLCM matrix
            glcm = graycomatrix(roiImage, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

            # Escolhe o descritor baseado em NT
            if NT == 0:
                descriptor_value = greycoprops(glcm, 'contrast')[0, 0]
                descriptor_name = 'Contrast'
            elif NT == 1:
                descriptor_value = greycoprops(glcm, 'dissimilarity')[0, 0]
                descriptor_name = 'Dissimilarity'
            elif NT == 2:
                descriptor_value = greycoprops(glcm, 'homogeneity')[0, 0]
                descriptor_name = 'Homogeneity'
            elif NT == 3:
                descriptor_value = greycoprops(glcm, 'energy')[0, 0]
                descriptor_name = 'Energy'

            # Exibe as imagens e o resultado um do lado do outro
            frame = Frame(scrollable_frame)
            frame.pack(pady=10)

            # Mostra a ROI
            roiPIL = Image.fromarray(roiImage)
            roiPIL = roiPIL.resize((200, 200), Image.LANCZOS)
            roiPhoto = ImageTk.PhotoImage(roiPIL)
            imgLabel = Label(frame, image=roiPhoto)
            imgLabel.image = roiPhoto 
            imgLabel.pack(side='left', padx=10)

            # Mostra o descritor
            featuresText = (
                f"{descriptor_name}: {descriptor_value:.4f}"
            )

            # Alinhamentos
            featuresLabel = Label(frame, text=featuresText, font='none 12', justify='left')
            featuresLabel.pack(side='right', padx=10)     

    def viewGLCMRadial(self):
        # Cria nova aba
        glcmWindow = Toplevel(self.app)
        glcmWindow.title("Radial GLCM & Texture Features of Saved ROIs")
        glcmWindow.geometry("1200x800")

        # Scroll
        scrollable_frame = self.createScrollableCanvas(glcmWindow)

        #Pega todos os arquivos
        roiFiles = self.listSavedROIFiles()

        # Calcula a distancia entre 1, 2, 4, 8
        distances = [1, 2, 4, 8]
        for roiFile in roiFiles:
            roiPath = os.path.join(self.savePath, roiFile)
            roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)

            frame = Frame(scrollable_frame)
            frame.pack(pady=20)

            roiPIL = Image.fromarray(roiImage)
            roiPIL = roiPIL.resize((200, 200), Image.LANCZOS)
            roiPhoto = ImageTk.PhotoImage(roiPIL)
            imgLabel = Label(frame, image=roiPhoto)
            imgLabel.image = roiPhoto
            imgLabel.pack(side='top', pady=10)

            textFrame = Frame(frame)
            textFrame.pack(side='top', pady=10)

            half = len(distances) // 2
            leftFrame = Frame(textFrame)
            leftFrame.pack(side='left', padx=20)

            rightFrame = Frame(textFrame)
            rightFrame.pack(side='left', padx=20)

            # Calcula GLCMs para diferentes texturas e distancias e coloca em duas colunas
            for i, distance in enumerate(distances):
                glcm = graycomatrix(roiImage, distances=[distance], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = greycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
                energy = greycoprops(glcm, 'energy')[0, 0]
                correlation = greycoprops(glcm, 'correlation')[0, 0]

                featuresText = (
                    f"Distance {distance} px:\n"
                    f"  Contrast: {contrast:.4f}\n"
                    f"  Dissimilarity: {dissimilarity:.4f}\n"
                    f"  Homogeneity: {homogeneity:.4f}\n"
                    f"  Energy: {energy:.4f}\n"
                    f"  Correlation: {correlation:.4f}\n"
                )

                # Adiciona as propriedades em duas colunas
                if i < half:
                    featuresLabel = Label(leftFrame, text=featuresText, font='none 12', justify='left', anchor='w')
                    featuresLabel.pack(pady=5)
                else:
                    featuresLabel = Label(rightFrame, text=featuresText, font='none 12', justify='left', anchor='w')
                    featuresLabel.pack(pady=5)        

if __name__ == "__main__":
    root = Tk()
    app = CropApp(root, "./ROISavedFiles", 636, 434)
    root.mainloop()
