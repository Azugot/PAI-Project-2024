
import tkinter as tk
from tkinter import filedialog

#Set up the UI of the application
def setUpUi():
    ui = tk.Tk()
    ui.title("PAI PROJECT")
    
    #Set the size of the window
    ui.geometry("800x600")
    #Set the Font of the title
    title = tk.Label(ui, text="PAI PROJECT", font=("Arial", 24))
    title.pack()

    #Set Canvas size
    image_area = tk.Canvas(ui, width=400, height=400)
    image_area.pack(pady=(10,0))

    #Open Image Button
    open_image = tk.Button(ui, width=20, text='OPEN IMAGE', font='none 12')
    open_image.pack(pady=(10,5))
    
    #Crop Area Button
    crop_area = tk.Button(ui, width=20, text='SELECT AREA', font='none 12')
    crop_area.pack(pady=(0,5))
    
    #Show Area Button
    show_area = tk.Button(ui, width=20, text='SHOW AREA', font='none 12')
    show_area.pack(pady=(0,5))
    
    #Save Image Button
    save_image = tk.Button(ui, width=20, text='SAVE IMAGE', font='none 12')
    save_image.pack()

    return ui

def openAndPut() :
    path = tk.filedialog.askopenfilename()
    global image
    if path:
        image = tk.Image.open(path)
        image = image.resize ((490,500), tk.Image.ANTIALIAS)
        image = tk.ImageTk.PhotoImage(image)
        image_area.create_image(0, 0, image=image, anchor='nw')


if __name__ == '__main__':
    print('Starting the application')
    
    print("Starting the UI")
    ui = setUpUi()
    ui.mainloop()