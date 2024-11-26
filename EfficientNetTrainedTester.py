import os
import re
import csv
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Configuração do dispositivo (CPU ou GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Recriar o modelo exatamente como no treinamento
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Alterar a última camada para 2 classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)

# Carregar os pesos do modelo treinado
path = '2_Epochs_64_BatchSize_0.001_LR'
model.load_state_dict(torch.load(f'./Training Iterations/{path}/FinalResults/final_model.pth', map_location=device))
model.eval()

print("Modelo carregado com sucesso!")

# Transformações aplicadas às imagens
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pasta contendo as imagens de teste
#test_folder = 'ROISavedFiles/'
test_folder = 'TestSet/'
#test_folder = 'FullImageTestSet/'


# Variáveis para cálculo da acurácia
all_labels = []  # Classes verdadeiras
all_predictions = []  # Classes preditas

# Iterar sobre todas as imagens na pasta
results = []  # Para armazenar os resultados
pattern = r'ROI_(\d+)_\d+\.png'  # Regex para extrair o número do paciente

for filename in os.listdir(test_folder):
    file_path = os.path.join(test_folder, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Extrair o número do paciente do nome do arquivo
            match = re.match(pattern, filename)
            if match:
                patient_number = int(match.group(1))
                true_class = 0 if patient_number <= 16 else 1  # Determinar a classe verdadeira
                all_labels.append(true_class)

                # Carregar e transformar a imagem
                image = Image.open(file_path).convert('L')
                image = data_transforms(image)
                image = image.unsqueeze(0).to(device)  # Adicionar dimensão do batch e enviar para o dispositivo

                # Fazer a inferência
                with torch.no_grad():
                    outputs = model(image)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities).item()

                # Salvar a classe predita
                all_predictions.append(predicted_class)

                # Salvar os resultados
                results.append((filename, true_class, predicted_class, probabilities.cpu().numpy()))
                print(f"Imagem: {filename}, Classe Verdadeira: {true_class}, Classe Predita: {predicted_class}, Probabilidades: {probabilities.cpu().numpy()}")
            else:
                print(f"Nome de arquivo inválido: {filename}. Pulando.")

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

# Calcular a acurácia total
if all_labels and all_predictions:
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nAcurácia total no conjunto de teste: {accuracy:.4f}")
else:
    print("\nNão foi possível calcular a acurácia devido à falta de dados suficientes.")

# Salvar os resultados em um arquivo CSV
output_csv = 'test_results.csv'
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'True Class', 'Predicted Class', 'Probabilities'])
    for result in results:
        writer.writerow(result)

print(f"\nResultados salvos em {output_csv}")
