# PROCESSAMENTO E ANALISE DE IMAGENS
# TRABALHO 1

# INTEGRANTES DO GRUPO:
#   Augusto Scardua Oliveira
#   Gabriel Pessotti de Deus
#   Vitoria de Lourdes Carvalho Santos


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
from math import log
import pandas as pd
import csv

global_glcm_properties = []
global_sfm_properties = {}

class CropApp:
    def __init__(self, root, savePath, uiWidth=434, uiHeight=636):
        #Variables for the ROI Window(Yes a separate window, please do not touch thank you xoxo)
        self.ROIimage = None
        self.zoomEnabledROI = True
        self.zoomLevelROI = 1
        self.displayingROIPath = None
        
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
        self.app.title('Trabalho PAI')
        self.app.geometry('1300x1000')

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
        self.histCanvas = Canvas(self.histFrame, width=uiWidth, height=uiHeight, bg='#E8E8E8')
        self.histCanvas.pack()

        self.roi1 = None
        self.roi2 = None 
        self.areaROI1 = None
        self.areaROI2 = None

        # B U T T O N S

        # Button to open Image/Mat file
        self.openImage = Button(self.app, width=20, text='OPEN IMAGE', font='none 12', command=self.readImage)
        self.openMat = Button(self.app, width=20, text='OPEN MAT DATASET', font='none 12', command=self.readMatFiles)
        
        # Buttons to navigate through the images (initially hidden)
        self.nextPatient = Button(self.app, width=20, text='NEXT PATIENT', font='none 12', command=self.nextMatPatient)
        self.previousPatient = Button(self.app, width=20, text='PREVIOUS PATIENT', font='none 12', command=self.previousMatPatient)
        self.nextPatientImage = Button(self.app, width=20, text='NEXT PATIENT IMAGE', font='none 12', command=self.nextMatPatientImage)
        self.previousPatientImage = Button(self.app, width=20, text='PREVIOUS PATIENT IMAGE', font='none 12', command=self.previousMatPatientImage)
        
        # ROI Related Buttons (initially hidden)
        self.chooseRoi = Button(self.app, width=20, text='SELECT ROI', font='none 12', command=self.toggleROI)
        self.showArea = Button(self.app, width=20, text='SHOW ROI', font='none 12', command=self.showROI)
        self.saveSelectedROI = Button(self.app, width=20, text='SAVE ROI', font='none 12', command=self.saveROI)
        self.openROIWindow = Button(self.app, width=20, text='ROI WINDOW', font='none 12', command=self.showROIWindow)
        self.resetROISelection = Button(self.app, width=20, text='RESET ROI', font='none 12', command=self.deleteDualROIarea)
        
        # Zoom Reset Button (initially hidden)
        self.resetZoomButton = Button(self.app, width=20, text='RESET ZOOM', font='none 12', command=self.resetZoom)
        self.zoomOutButton = Button(self.app, width=20, text='-', font='none 12', command=self.zoomOut)
        self.zoomInButton = Button(self.app, width=20, text='+', font='none 12', command=self.zoomIn)
        self.toggleZoomButton = Button(self.app, width=20, text='ENABLE ZOOM', font='none 12', command=self.toggleZoom)
        
        # Grid Layout
        self.openImage.grid(row=1, column=0, sticky="n")
        self.openMat.grid(row=2, column=0, sticky="n")
        self.imageArea.grid(row=0, column=0, columnspan=3)
        self.openROIWindow.grid(row=3, column=0, columnspan=3)

# MOSTRAR BOTOES DEPOIS DE ENVIAR A IMAGEM
    def showAdditionalButtons(self):
        self.showArea.grid(row=5, column=0, sticky="n")
        self.chooseRoi.grid(row=3, column=0, sticky="n")
        self.toggleZoomButton.grid(row=4, column=1, sticky="n", padx=5)
        self.saveSelectedROI.grid(row=4, column=0, sticky="n")
        self.previousPatientImage.grid(row=2, column=2, sticky="n", padx=5)
        self.previousPatient.grid(row=5, column=2, sticky="n", padx=5)
        self.nextPatientImage.grid(row=1, column=2, sticky="n", padx=5)
        self.nextPatient.grid(row=4, column=2, sticky="n", padx=5)
        self.resetZoomButton.grid(row=7, column=1, sticky="n")
        self.zoomInButton.grid(row=5, column=1, sticky="n", padx=5)
        self.zoomOutButton.grid(row=6, column=1, sticky="n", padx=5)
        self.chooseRoi.grid(row=3, column=0, sticky="n", padx=10, pady=10)
        self.resetROISelection.grid(row=6, column=0, sticky="n", padx=10, pady=10)
        
    def readImage(self):
        self.matFileIsOpen = False
        self.matFile = None
        path = filedialog.askopenfilename()
        if (path):
            self.imageArea.delete("all")
            self.histCanvas.delete("all")
            self.image = Image.open(path)
            self.imageForMaskMultiplication = self.image.resize((self.uiWidth, self.uiHeight), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(self.imageForMaskMultiplication)
            self.imageArea.create_image(0, 0, image=self.image, anchor='nw')
            self.showHistogram(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            self.showAdditionalButtons()
    
    
        if (0 <= self.lasX < 500 and 0 <= self.lasY < 400):
            self.mask[self.lasY][self.lasX] = 0 
            self.mask[self.lasY+1][self.lasX+1] = 0 
            self.mask[self.lasY-1][self.lasX-1] = 0 
            self.mask[self.lasY+1][self.lasX-1] = 0 
            self.mask[self.lasY-1][self.lasX+1] = 0 


        if (0 <= self.lasX < 500 and 0 <= self.lasY < 400):
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
        #Caso a imagem for colorida passa ela para cinza
        if (len(matImage.shape) == 3): 
            matImage = cv2.cvtColor(matImage, cv2.COLOR_BGR2GRAY)

        #Pega a altura e a largura da imagem e o total de pixels
        height, width = matImage.shape
        total_pixels = height * width

        # inicialmente o histograma tem 256 posições com valores iniciais zero
        hist = np.zeros(256, dtype=int)
        
        # Percorre cada pixel e incrementando a posição
        for i in range(height):
            for j in range(width):
                intensity_value = matImage[i, j]
                hist[intensity_value] += 1

        #Apenas validade se deu o total de pixeis
        assert sum(hist) == total_pixels

        # Cria o gráfico para mostrar o histograma
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(hist, color='black')
        ax.set_title(f"Histogram - Paciente {self.numPatient}, Imagem {self.imgPatient}")
        ax.set_xlim([0, 255])  # O eixo x é intensidade dos pixels
        ax.set_xlabel('Pixel Intensity') 
        ax.set_ylabel('Number of Pixels')

        # Define o limite superior do eixo y como 110% do valor do percentil 95 do histograma para visualização melhorada
        max_value = np.percentile(hist, 95)
        ax.set_ylim([0, max_value * 1.1])

        # Salva a imagem do histograma em formato png
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        #Pega suas dimensões do histogrma
        hist_image = Image.open(buf)
        hist_width, hist_height = hist_image.size

        # Pega as dimensoes do Canvas para exibir o histograma
        self.histCanvas.config(width=hist_width, height=hist_height)

        # Converte a imagem para o formato ImageTk para exibir no canvas
        self.hist_img_tk = ImageTk.PhotoImage(hist_image)
        self.histCanvas.create_image(0, 0, image=self.hist_img_tk, anchor='nw')

        # Fecha o buffer e a figura
        buf.close()
        plt.close(fig)


    def navigateThroughMatFile(self, numPatient, imgPatient):
        # Olha se o arquivo .mat esta aberto e o caminho é válido
        if (self.matFileIsOpen and self.path):
            # Armazena a matriz do arquivo .mat
            data = scipy.io.loadmat(self.path)

            # Extrai o array de dados da variável 'data'
            dataArray = data['data'] 

            # Pega a primeira entrada referente ao paciente
            input = dataArray[0, numPatient]
            
            # Extrai as imagens dessa entrada
            matImagens = input['images']
            matImage = matImagens[imgPatient]

            # Converte a imagem para um formato
            matImage = np.array(matImage)

            # Converte o array NumPy (lista de numeros mais rapidas)
            pilImage = Image.fromarray(matImage)

            # Redimensiona a imagem se necessário
            pilImage = pilImage.resize((self.uiWidth, self.uiHeight), Image.LANCZOS)

            # Atualiza as referências de imagem
            self.image = ImageTk.PhotoImage(pilImage)
            self.imageForMaskMultiplication = pilImage

            # Exibe a imagem no Canvas
            self.imageArea.create_image(0, 0, image=self.image, anchor='nw')

            # Exibe o histograma da imagem no Canvas
            self.showHistogram(matImage)

# ROI RELATED METHODS
    def listSavedROIFiles(self):
        # Function to list all saved ROI files
        return [f for f in os.listdir(self.savePath) if (os.path.isfile(os.path.join(self.savePath, f)))]

    def saveROI(self):
        if (self.roi1 and self.roi2):
            liverROI = self.acquireROI(self.roi1)
            kidneyROI = self.acquireROI(self.roi2)

            liverArray = np.array(liverROI.convert("L"))
            kidneyArray = np.array(kidneyROI.convert("L"))

            liverMean = np.mean(liverArray)
            kidneyMean = np.mean(kidneyArray)

            HI = liverMean / kidneyMean
            liverArrayAdjusted = np.clip(np.round(liverArray * HI), 0, 255).astype(np.uint8)
            liverROIAdjusted = Image.fromarray(liverArrayAdjusted)

            os.makedirs(self.savePath, exist_ok=True)
            file_name = os.path.join(self.savePath, f"ROI_{self.numPatient:02}_{self.imgPatient}.png")
            liverROIAdjusted.save(file_name, "PNG")
            print(f"ROI do fígado salva em: {file_name}")

            # Definindo a classe dos pacientes
            patient_class = "Saudável" if self.numPatient <= 16 else "Esteatose"
            csv_file = os.path.join(self.savePath, 'rois_data.csv')

            # Inicializa o arquivo CSV principal caso ele ainda nao existir
            if not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Arquivo', 'Classificação', 'ROI Fígado ', 'ROI Rim', 'HI'])
                    for _ in range(550):
                        writer.writerow(["", "", "", "", ""])

            # Le todas as linhas do CSV
            with open(csv_file, mode='r') as file:
                rows = list(csv.reader(file))

            # Atualiza a linha específica com as informações da ROI
            row_index = self.numPatient * 10 + self.imgPatient
            rows[row_index + 1] = [f"ROI_{self.numPatient:02}_{self.imgPatient}.png", patient_class, self.roi1[:2], self.roi2[:2], f"{HI:.4f}"]

            # Salva as linhas atualizadas no CSV
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

            # Caminho para o CSV das informações GLCM e SFM
            glcm_sfm_csv = os.path.join(self.savePath, 'glcm_sfm_data.csv')

            # Inicializa o arquivo CSV de GLCM e SFM se ele nao existir
            if not os.path.exists(glcm_sfm_csv):
                with open(glcm_sfm_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        'Arquivo', 'Classificação', 'Distance', 'Angle', 'Contrast', 'Dissimilarity',
                        'Homogeneity', 'Energy', 'Correlation', 'Entropy', 'Coarseness', 'Periodicity', 'Roughness'
                    ])

            # Chama as funções para calcular e armazenar as propriedades GLCM e SFM
            self.displayRadialGLCMInROIWindow(file_name, histogramFrame=None, distances=[1, 2, 4, 8])
            self.displaySFMPropertiesInROIWindow(file_name, ROIDisplay=None)

            # Acessa as variáveis globais para gravar no CSV
            with open(glcm_sfm_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                for glcm_property in global_glcm_properties:
                    writer.writerow([
                        f"ROI_{self.numPatient:02}_{self.imgPatient}.png", patient_class,
                        glcm_property['distance'], glcm_property['angle'],
                        f"{glcm_property['contrast']:.4f}", f"{glcm_property['dissimilarity']:.4f}",
                        f"{glcm_property['homogeneity']:.4f}", f"{glcm_property['energy']:.4f}",
                        f"{glcm_property['correlation']:.4f}", f"{glcm_property['entropy']:.4f}",
                        f"{global_sfm_properties.get('coarseness', 0):.4f}",
                        f"{global_sfm_properties.get('periodicity', 0):.4f}",
                        f"{global_sfm_properties.get('roughness', 0):.4f}"
                    ])

            # Limpa as ROIs apos salvar
            self.roi1 = None
            self.roi2 = None

    def deleteROIarea(self):
        # Verifica se existe uma área de região de interesse definida
        if (self.areaROI):
            # Remove a área de ROI do widget da imagem
            self.imageArea.delete(self.areaROI)
            # Define a área de ROI como nulo para indicar que não há mais ROI ativa
            self.areaROI = None

    def deleteDualROIarea(self):
        # Verifica se existe alguma das áreas de ROI definidas
        if (self.areaROI1 or self.areaROI2):
            # indica que as áreas foram resetadas
            self.roi1 = None
            self.roi2 = None
            # Remove as áreas de ROI1 e ROI2 do widget de imagem
            self.imageArea.delete(self.areaROI1)
            self.imageArea.delete(self.areaROI2)
            # Define ROI1 e ROI2 como None para indicar que as áreas foram removidas
            self.areaROI1 = None
            self.areaROI2 = None
            # Exibe uma mensagem indicando que as ROIs foram resetadas
            print("ROIs resetadas.")
            # Se a variável roiOn estiver ativa, chama o método gotoROI1
            if (self.roiOn):
                self.gotoROI1()


    def toggleROI(self):
        #DO NOT TOUCH THIS PART ===============
        # Olha se a marcação de ROI está ativada
        if (self.roiOn):
            # Desativa a marcação de ROI
            self.roiOn = False
            # Atualiza o texto do botão para indicar que a marcação de ROI está desativada
            self.chooseRoi.config(text="SELECT ROI")
            # Remove os eventos de clique e arrasto do Canvas de imagem
            self.imageArea.unbind("<Button-1>")
            self.imageArea.unbind("<B1-Motion>")
            # Deleta as áreas de ROI duplas se tiver
            self.deleteDualROIarea()

        else:
            #DO NOT TOUCH THIS PART ===============
            # Ativa a marcação de ROI
            self.roiOn = True
            # Atualiza o texto do botão para indicar que a marcação de ROI está ativada
            self.chooseRoi.config(text="END SELECT ROI")

            # Desativa o zoom enquanto a marcação de ROI está ativa
            if (self.zoomEnabled):
                self.toggleZoom()
            #DO NOT TOUCH THIS PART ===============

            # Determina qual ROI será selecionada a seguir
            if (self.roi1 is not None and self.roi2 is None):
                # Se a primeira ROI ja esta marcada e a segunda não passa a marcação da segunda ROI
                self.gotoROI2()
            
            elif (self.roi1 is None):
                # Se nenhuma ROI esta marcada começa pela primeira ROI
                self.gotoROI1()
            
            else:
                # Se ambas as ROIs ja estão marcadas
                print("ROIs já foram marcadas")
                # Atualiza o botão para indicar que ambas as ROIs estão marcadas
                self.chooseRoi.config(text="ROIs SELECTED")
                # Remove os eventos de clique e arrasto
                self.imageArea.unbind("<Button-1>")
                self.imageArea.unbind("<B1-Motion>")
        
        # Exibe o status atual de marcação de ROI e zoom
        print(f"""TOGGLE STATUS:
                    ROI:{self.roiOn}
                    ZOOM:{self.zoomEnabled}""")

    def gotoROI1(self):
        # Caso nenhuma ROI esteja marcada inicia a marcação da primeira ROI
        self.chooseRoi.config(text="SELECT LIVER ROI (ROI 1)")
        # Define eventos de clique e arrasto para iniciar e finalizar a marcação da primeira ROI
        self.imageArea.bind("<Button-1>", self.startDrawROI)
        self.imageArea.bind("<B1-Motion>", self.finishDrawROI)
        print("Marque a primeira ROI (fígado)")

    def gotoROI2(self):
        # Caso a primeira ROI já esteja marcada permite a marcação da segunda ROI
        self.chooseRoi.config(text="SELECT KIDNEY ROI (ROI 2)")
        # Define eventos de clique e arrasto para iniciar e finalizar a marcação da segunda ROI
        self.imageArea.bind("<Button-1>", self.startDrawROI2)
        self.imageArea.bind("<B1-Motion>", self.finishDrawROI2)
        print("Marque a segunda ROI (rim)")


    def startDrawROI(self, event):
        # Pega o clique inicial
        self.startX = event.x
        self.startY = event.y
        
        # Remove qualquer roi anterior desenhada
        if (self.areaROI1):
            self.imageArea.delete(self.areaROI1)
        
        # Faz o quadrado da roi com tamanho fixo de 28x28 pixels
        self.areaROI1 = self.imageArea.create_rectangle(self.startX-14, self.startY-14, self.startX+14, self.startY+14, outline="green", width=2)
        
        #salva as coordenadas da primeira roi
        self.roi1 = (self.startX-14, self.startY-14, self.startX+14, self.startY+14)
        print(f"Primeira ROI (fígado) selecionada: {self.roi1}")
        
        #muda para a seleção da segunda roi
        self.chooseRoi.config(text="SELECT KIDNEY ROI (ROI 2)")
        self.imageArea.bind("<Button-1>", self.startDrawROI2)

    def finishDrawROI(self, event):
        #salva as coordenadas da primeira roi
        self.roi1 = (self.startX-14, self.startY-14, self.startX+14, self.startY+14)
        print(f"Primeira ROI (fígado) selecionada: {self.roi1}")
        
        #muda para a seleção da segunda ROI (rim)
        self.chooseRoi.config(text="SELECT KIDNEY ROI (ROI 2)")
        self.imageArea.bind("<Button-1>", self.startDrawROI2)
        self.imageArea.bind("<B1-Motion>", self.finishDrawROI2)
        print("Selecione a segunda ROI (rim).")

    def startDrawROI2(self, event):
        # Pega o clique inicial
        self.startX = event.x
        self.startY = event.y
        
        #remove qualquer roi anterior desenhada
        if (self.areaROI2):
            self.imageArea.delete(self.areaROI2)
        
        #faz o quadrado da roi com tamanho fixo de 28x28 pixels
        self.areaROI2 = self.imageArea.create_rectangle(self.startX-14, self.startY-14, self.startX+14, self.startY+14, outline="green", width=2)
        
        #salva as coordenadas da primeira roi        
        self.roi2 = (self.startX-14, self.startY-14, self.startX+14, self.startY+14)
        print(f"Segunda ROI (rim) selecionada: {self.roi2}")
        
        #muda o texto do botao
        self.chooseRoi.config(text="BOTH ROIs SELECTED")
        self.imageArea.unbind("<Button-1>")  #desativa o clique após selecionar ambas as rois

    def finishDrawROI2(self, event):
        #salva as coordenadas da segunda roi
        self.roi2 = (self.startX-14, self.startY-14, self.startX+14, self.startY+14)
        print(f"Segunda ROI (rim) selecionada: {self.roi2}")
        
        #desativa tudo porque ja a dois rois ja foram slecionadas
        self.chooseRoi.config(text="BOTH ROIs SELECTED")
        self.imageArea.unbind("<Button-1>")
        self.imageArea.unbind("<B1-Motion>")
        print("Ambas as ROIs foram selecionadas.")

    def drawROIFixed(self, event):
        if (self.roiOn):
            self.deleteROIarea()

            self.startX = event.x
            self.startY = event.y

            self.areaROI = self.imageArea.create_rectangle(self.startX-14, self.startY-14, self.startX+14, self.startY+14, outline="green", width=2)

    #TODO: Fix this method for generic Images
    def acquireROI(self, roi_coords):
        x1, y1, x2, y2 = roi_coords
        x1, y1 = map(int, [max(0, x1), max(0, y1)])
        
        #tamanho das duas rois
        x2 = x1 + 28
        y2 = y1 + 28
        
        originalX, originalY = self.imageForMaskMultiplication.size
        scaleX = originalX / self.uiWidth
        scaleY = originalY / self.uiHeight
        
        # Corrige as coordenadas com base na escala da imagem original
        correctedX1 = int(x1 * scaleX)
        correctedY1 = int(y1 * scaleY)
        correctedX2 = int(x2 * scaleX)
        correctedY2 = int(y2 * scaleY)
        
        return self.imageForMaskMultiplication.crop((correctedX1, correctedY1, correctedX2, correctedY2))
    
    def showROI(self):  
        if (self.areaROI1 and self.areaROI2):
            liverROI = self.acquireROI(self.roi1)  #salva as coordenadas da roi 1
            kidneyROI = self.acquireROI(self.roi2)  #salva as coordenadas da roi 2

            #mostra as rois
            liverROI.show()
            kidneyROI.show()
        else:
            print("Tem que marca as duaaas")
            
# SCROLL
    def createScrollableCanvas(self, parentWindow):
        #Faz um Canvas dentro da janela pai
        canvas = Canvas(parentWindow)
        #Faz uma barra de rolagem vertical associada ao Canvas
        scrollbar = Scrollbar(parentWindow, orient="vertical", command=canvas.yview)
        #Faz um frame que vai dentro do Canvas é o que realmente vai rolar
        scrollable_frame = Frame(canvas)

        #Configura a área rolável do Canvas quando o frame é redimensionado
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        #Posiciona o frame dentro do Canvas, no lado superior esquerdo
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        # Vincula a barra de rolagem ao Canvas
        canvas.configure(yscrollcommand=scrollbar.set)

        #coloca o Canvas e a barra de rolagem na tela
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Retorna o frame rolável que acabou de ser criado
        return scrollable_frame

    def nextMatPatient(self):
        #se o arquivo .mat estiver aberto vai para o próximo paciente 
        if (self.matFileIsOpen):
            # caso chegue ao último paciente volta para o primeiro paciente
            if (self.numPatient >= 54):
                self.numPatient = 0
            else:
                # se não vai para o próximo paciente
                self.numPatient += 1
            
            self.imgPatient = 0
            
            print("Paciente: ", self.numPatient)
            print("Imagem: ", self.imgPatient)
            
            # Vai para a imagem atual do paciente
            if (self.imgPatient >= 0 and self.imgPatient <= 54):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)

    def previousMatPatient(self):
        # Volta para o paciente anterior se o arquivo .mat estiver aberto
        if (self.matFileIsOpen):
            if (self.numPatient <= 0):
                self.numPatient = 54
            else:
                self.numPatient -= 1
            
            self.imgPatient = 0
                        
            print("Paciente: ", self.numPatient)
            print("Imagem: ", self.imgPatient)        
            
            if (self.imgPatient >= 0 and self.imgPatient <= 54):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)
            
# PASSA A IMAGEM DO PACIENTE
    def nextMatPatientImage(self):
        if (self.matFileIsOpen):
            # Se chegou à última imagem, volta para a primeira
            if (self.imgPatient >= 9):
                self.imgPatient = 0
            else:
                # Caso contrário, vai para a próxima imagem
                self.imgPatient += 1
            
            # Mostra o paciente e a imagem atual
            print("Paciente: ", self.numPatient)
            print("Imagem: ", self.imgPatient)
            
            # Navega para a imagem atual do paciente
            if (self.imgPatient >= 0 and self.imgPatient <= 9):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)
            
    def previousMatPatientImage(self):
        if (self.matFileIsOpen):
            if (self.imgPatient <= 0):
                self.imgPatient = 9
            else:
                self.imgPatient -= 1
            
            print("Paciente: ", self.numPatient)
            print("Imagem: ", self.imgPatient)
            
            if (self.imgPatient >= 0 and self.imgPatient <= 9):
                self.navigateThroughMatFile(self.numPatient, self.imgPatient)

    # ENTROPIA
    def calculo_entropia_glcm(self, glcm):
        glcm_normalized = glcm / np.sum(glcm)  # normaliza a GLCM
        entropy = 0
        for i in range(glcm_normalized.shape[0]):
            for j in range(glcm_normalized.shape[1]):
                if glcm_normalized[i, j] > 0:  # para evitar log(0)
                    entropy -= glcm_normalized[i, j] * np.log2(glcm_normalized[i, j])
        return entropy

  
# ROI IMAGE WINDOW (This is here because the code is already unorganized)
    def showROIWindow(self):
        #print("I AM STEVE")

        ROICanvasWidth = 280
        ROICanvasHeight = 280

        ROIWindowBase = Toplevel(self.app)
        ROIWindowBase.title("ROIs")
        ROIWindowBase.geometry("1200x800")

        sidebarWithScroll = self.createScrollableCanvas(ROIWindowBase)
        sidebarWithScroll.config(bg='white')

        ROISideBar = Frame(sidebarWithScroll, bg='white')
        ROISideBar.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=10, pady=10)

        ROIDisplay = Frame(ROIWindowBase)
        ROIDisplay.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.ROIArea = Canvas(ROIDisplay, width=ROICanvasWidth, height=ROICanvasHeight, bg='#C8C8C8')
        self.ROIArea.pack()

        self.histogramFrame = Frame(ROIWindowBase)
        self.histogramFrame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        resetZoomButtonROI = Button(ROIDisplay, width=20, text='RESET ZOOM', font='none 12', command=self.resetZoomROI)
        zoomOutButtonROI = Button(ROIDisplay, width=20, text='-', font='none 12', command=self.zoomOutROI)
        zoomInButtonROI = Button(ROIDisplay, width=20, text='+', font='none 12', command=self.zoomInROI)

        resetZoomButtonROI.pack(side=tk.BOTTOM)
        zoomInButtonROI.pack(side=tk.BOTTOM)
        zoomOutButtonROI.pack(side=tk.BOTTOM)

        roiFiles = self.listSavedROIFiles()

        for roiFile in roiFiles:
            roiPath = os.path.join(self.savePath, roiFile)
            imgButton = Button(ROISideBar, text=roiFile, font='none 12', command=lambda roiPath=roiPath: self.displayItemsInROICanvas(roiPath, self.ROIArea, ROICanvasHeight, ROICanvasWidth, ROIDisplay))
            imgButton.pack(pady=5)
    
    def displayItemsInROICanvas(self, roiPath, ROIArea, ROICanvasHeight, ROICanvasWidth, ROIDisplay):
        self.displayROIinROIWindowCanvas(roiPath, ROIArea, ROICanvasHeight, ROICanvasWidth)
        self.displayHistogramInROIWindow(roiPath, ROICanvasWidth, ROICanvasHeight)
        self.displayRadialGLCMInROIWindow(roiPath, self.histogramFrame)
        self.displaySFMPropertiesInROIWindow(roiPath, ROIDisplay)

    def displayROIinROIWindowCanvas(self, roiPath, ROIArea, ROICanvasHeight, ROICanvasWidth):
        ROIArea.delete("all")
        print("ROI: " + roiPath)

        # Update the currently displaying ROI path
        self.displayingROIPath = roiPath

        # Reset the zoom level and position before displaying the new image
        self.resetZoomROI()

        # Open the image and convert to PhotoImage
        self.ROIimage = Image.open(roiPath)
        self.ROIimageTk = ImageTk.PhotoImage(self.ROIimage)

        # Calculate the position to center the image
        width, height = self.ROIimage.size
        offsetX = (ROICanvasWidth - width) // 2
        offsetY = (ROICanvasHeight - height) // 2

        # Display the image at the calculated center position
        ROIArea.create_image(offsetX, offsetY, image=self.ROIimageTk, anchor='nw')

    def displayHistogramInROIWindow(self, roiPath, histWidth=280, histHeight=280):
        # Limpa o histograma anterior da imagem
        for widget in self.histogramFrame.winfo_children():
            widget.destroy()
        
        # Carrega a imagem da região da ROI em escala de cinza
        roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)
        
        height, width = roiImage.shape  # Pega a altura e largura da imagem
        total_pixels = height * width   # Pega o total de pixels

        # Cria um array para o histograma com 256 posições
        hist = np.zeros(256, dtype=int)
        # Conta cada nível de intensidade de pixel e incrementa o valor correspondente no histograma
        for i in range(height):
            for j in range(width):
                intensity_value = roiImage[i, j]
                hist[intensity_value] += 1

        # Verifica se a soma dos valores do histograma é igual ao total de pixels da imagem
        assert sum(hist) == total_pixels
        
        # Configura o gráfico tamanho fixo 280x280 pixels
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        ax.set_title(f"Histogram for {os.path.basename(roiPath)}", fontsize=10)
        ax.plot(hist, color='black')
        ax.set_xlim([0, 255])
        # os rótulos e o tamanho da fonte dos eixos
        ax.set_xlabel('Pixel Intensity', fontsize=8)
        ax.set_ylabel('Number of Pixels', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Ajusta o layout para evitar que as etiquetas fiquem cortadas
        fig.tight_layout()
        
        # Insere o gráfico do histograma dentro do frame do histograma
        canvas_hist = FigureCanvasTkAgg(fig, master=self.histogramFrame)
        canvas_hist.draw()
        
        # Garante que o gráfico tenha tamanho fixo 280x280 pixels e adiciona um espaço ao redor
        hist_widget = canvas_hist.get_tk_widget()
        hist_widget.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)
        hist_widget.config(width=histWidth*1.5, height=histHeight)

    def displayRadialGLCMInROIWindow(self, roiPath, histogramFrame=None, distances=[1, 2, 4, 8]):
        global global_glcm_properties
        global_glcm_properties = []

        # Carrega a imagem da ROI em escala de cinza
        roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)

        # Define os ângulos para análise de textura
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        # Calcula a matriz GLCM para todas as distâncias e ângulos
        glcm = graycomatrix(roiImage, distances=distances, angles=angles, symmetric=True, normed=True)

        # Se histogram for fornecido vai criar o layout da interface caso nao ele foi ultilizado para salvar as coisas no csv
        if histogramFrame:
            glcmFrame = self.createScrollableCanvas(histogramFrame)
            radialLabel = Label(glcmFrame, text="GLCM RADIAL", font='none 14 bold', justify='center')
            radialLabel.pack(pady=5)

            # Cria colunas para organizar a exibição
            leftFrame = Frame(glcmFrame)
            leftFrame.pack(side='left', padx=20)
            rightFrame = Frame(glcmFrame)
            rightFrame.pack(side='left', padx=20)

        # Para cada distância e ângulo calcula as propriedades de textura
        for i, distance in enumerate(distances):
            for j, angle in enumerate(angles):
                contrast = greycoprops(glcm, 'contrast')[i, j]
                dissimilarity = greycoprops(glcm, 'dissimilarity')[i, j]
                homogeneity = greycoprops(glcm, 'homogeneity')[i, j]
                energy = greycoprops(glcm, 'energy')[i, j]
                correlation = greycoprops(glcm, 'correlation')[i, j]
                entropy = self.calculo_entropia_glcm(glcm[:, :, i, j])

                # Armazena as propriedades na variável global
                global_glcm_properties.append({
                    'distance': distance,
                    'angle': np.degrees(angle),
                    'contrast': contrast,
                    'dissimilarity': dissimilarity,
                    'homogeneity': homogeneity,
                    'energy': energy,
                    'correlation': correlation,
                    'entropy': entropy
                })

                #Mesma coisa da validacao do csv la encima
                if histogramFrame:
                    # Alterna entre as colunas para cada par de distância e ângulo
                    frame = leftFrame if (i * len(angles) + j) % 2 == 0 else rightFrame
                    featuresText = (
                        f"Distance {distance} px, Angle {np.degrees(angle):.0f}°:\n"
                        f"  Contrast: {contrast:.4f}\n"
                        f"  Dissimilarity: {dissimilarity:.4f}\n"
                        f"  Homogeneity: {homogeneity:.4f}\n"
                        f"  Energy: {energy:.4f}\n"
                        f"  Correlation: {correlation:.4f}\n"
                        f"  Entropy: {entropy:.4f}\n"
                    )
                    featuresLabel = Label(frame, text=featuresText, font='none 12', justify='left', anchor='w')
                    featuresLabel.pack(pady=5)


    def displaySFMPropertiesInROIWindow(self, roiPath, ROIDisplay=None):
            global global_sfm_properties
            global_sfm_properties = {}

            # Carrega a imagem da ROI
            roiImage = cv2.imread(roiPath, cv2.IMREAD_GRAYSCALE)

            # Coarseness - usando filtro de média e variância local
            kernel_size = 3
            local_mean = cv2.blur(roiImage, (kernel_size, kernel_size))
            local_var = cv2.blur((roiImage - local_mean)**2, (kernel_size, kernel_size))
            coarseness = np.mean(local_var)

            # Contrast - calculado com a GLCM
            glcm = graycomatrix(roiImage, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = greycoprops(glcm, 'contrast')[0, 0]

            # Periodicity - usando transformada de Fourier
            f_transform = np.fft.fft2(roiImage)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_transform_shifted)
            periodicity = np.mean(magnitude_spectrum)

            # Roughness - média dos gradientes locais
            grad_x = cv2.Sobel(roiImage, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roiImage, cv2.CV_64F, 0, 1, ksize=3)
            roughness = np.mean(np.sqrt(grad_x**2 + grad_y**2))

            # Armazena as propriedades na variavel global
            global_sfm_properties = {
                'coarseness': coarseness,
                'contrast': contrast,
                'periodicity': periodicity,
                'roughness': roughness
            }

            # Somente exibe na interface gráfica se ROIDisplay estiver definido solucao para salvar no csv as informacoes
            if ROIDisplay:
                sfmText = (
                    f"Coarseness: {coarseness:.4f}\n"
                    f"Contrast: {contrast:.4f}\n"
                    f"Periodicity: {periodicity:.4f}\n"
                    f"Roughness: {roughness:.4f}"
                )

                if hasattr(self, 'sfmLabel') and self.sfmLabel.winfo_exists():
                    self.sfmLabel.destroy()
                if hasattr(self, 'featuresLabel') and self.featuresLabel.winfo_exists():
                    self.featuresLabel.destroy()

                self.sfmLabel = Label(ROIDisplay, text="SMF", font='none 12 bold', justify='center')
                self.sfmLabel.pack(pady=5)
                self.featuresLabel = Label(ROIDisplay, text=sfmText, font='none 12', justify='center')
                self.featuresLabel.pack(pady=10)


# ZOOM
    def zoomInROI(self):
        if (self.zoomEnabledROI):
            self.zoomLevelROI += 0.1
            self.imageZoomUpdateROI()

    def zoomOutROI(self):
        if (self.zoomEnabledROI):
            self.zoomLevelROI = max(0.1, self.zoomLevelROI - 0.1)  # Ensure zoom level doesn't go negative or zero
            self.imageZoomUpdateROI()

    def imageZoomUpdateROI(self):
        # Open the original image
        self.ROIimage = Image.open(self.displayingROIPath)

        # Calculate the new dimensions based on zoom level
        width, height = self.ROIimage.size
        newWidth = int(width * self.zoomLevelROI)
        newHeight = int(height * self.zoomLevelROI)

        # Resize the image
        resizedImage = self.ROIimage.resize((newWidth, newHeight), Image.LANCZOS)

        # Update the displayed image
        self.ROIimageTk = ImageTk.PhotoImage(resizedImage)

        # Clear the canvas and calculate the position to center the image
        self.ROIArea.delete("all")
        offsetX = (self.ROIArea.winfo_width() - newWidth) // 2
        offsetY = (self.ROIArea.winfo_height() - newHeight) // 2

        # Display the image at the calculated center position
        self.ROIArea.create_image(offsetX, offsetY, image=self.ROIimageTk, anchor='nw')

    def resetZoomROI(self):
        self.zoomLevelROI = 1  # Reset zoom level to 1 (default scale)
        self.moveX, self.moveY = 0, 0  # Reset image position
        self.imageZoomUpdateROI()

    def toggleZoom(self):
        if (self.zoomEnabled):
            #DO NOT TOUCH THIS PART ===============
            self.toggleZoomButton.config(text="ENABLE ZOOM")
            self.zoomEnabled = False
            self.imageArea.unbind("<ButtonPress-2>")
            self.imageArea.unbind("<B2-Motion>")
            self.resetZoom()
            #DO NOT TOUCH THIS PART ===============
            
        else:
            #DO NOT TOUCH THIS PART ===============
            self.toggleZoomButton.config(text="DISABLE ZOOM")
            self.zoomEnabled = True
            self.imageArea.bind("<ButtonPress-2>", self.startMove)
            self.imageArea.bind("<B2-Motion>", self.moveImage)
            
            #ROI STUFF
            if (self.roiOn):
                self.toggleROI()
            #DO NOT TOUCH THIS PART ===============
            
        print(f"""TOGGLE STATUS:
                ROI:{self.roiOn}
                ZOOM:{self.zoomEnabled}""")
    
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

# MOVIMENTO DA IMAGEM
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
    

if ( __name__ == "__main__"):
    root = Tk()
    app = CropApp(root, "./ROISavedFiles", 636, 434)
    root.mainloop()