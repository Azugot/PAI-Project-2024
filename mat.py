from pathlib import Path
import scipy.io
import matplotlib.pyplot as plt

#caminho da pasta
path_data = Path("dataset_liver_bmodes_steatosis_assessment_IJCARS.mat")
data = scipy.io.loadmat(path_data)

data_array = data['data']

num_paciente = 8
imagem_paciente = 3

entrada = data_array[0, num_paciente]
imagens = entrada['images']


imagem = imagens[imagem_paciente]#pega a imagem

plt.figure(figsize=(9, 9)) #mostra a imagem
plt.imshow(imagem, cmap='gray')  #mostra a imagem em cinza
plt.axis('off')  #oculta os eixos da imagem
plt.show()  #exibe a janela com a imagem
