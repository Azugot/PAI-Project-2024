# PROCESSAMENTO E ANALISE DE IMAGENS
# TRABALHO 1

# INTEGRANTES DO GRUPO:
#   Augusto Scardua Oliveira
#   Gabriel Pessotti de Deus
#   Vitoria de Lourdes Carvalho Santos


import threading
from tkinter import Entry, Label, Text, filedialog, Canvas, Button, Tk, Toplevel, Frame, Scrollbar, Canvas
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
import os
import re
import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

global_glcm_properties = []
global_sfm_properties = {}

#CLASSIFICADOR RASO
class ShallowModel:
    def __init__(self):
        self.data = None
        self.output_folder = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.specificities = None
        self.sensitivities = None
        self.accuracies = None
    
    # Remoção de features altamente correlacionadas
    def remove_highly_correlated_features(self, df, numeric_columns, threshold=0.95):
        corr_matrix = df[numeric_columns].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)
    
    # Função para adicionar interações
    def add_interaction_features(self, df, numeric_cols, max_interactions=10):
        interactions = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                interaction_name = f'{col1}_{col2}_interaction'
                if len(interactions) < max_interactions: 
                    df[interaction_name] = df[col1] * df[col2]
                    interactions.append(interaction_name)
        return df
    
    def run(self, csvPath):
        try:
            # Início do temporizador
            self.start_time = time.time()
    
            # Definir a pasta de saída para os gráficos
            self.output_folder = './XGBoost'
            os.makedirs(self.output_folder, exist_ok=True)
    
            # Leitura do arquivo
            csv_path = csvPath
            if not os.path.exists(csv_path):
                print(f"CSV file not found at {csv_path}")
                return
    
            with open(csv_path, 'r', encoding='utf-8') as file:
                total_lines = sum(1 for line in file)
            print(f"Número de linhas no arquivo CSV: {total_lines}")
    
            data = pd.read_csv(csv_path, on_bad_lines='warn', encoding='utf-8')
            print(f"Número de linhas lidas inicialmente: {data.shape[0]}")
    
            # Pré-processamento
            data = data.drop_duplicates()
            data['Paciente'] = data['Arquivo'].str.extract(r'(ROI_\d+)_')
            data.drop(columns=['Periodicity', 'Roughness'], inplace=True)
    
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            data = self.add_interaction_features(data, numeric_columns)
    
            data = self.remove_highly_correlated_features(data, numeric_columns)
    
            # Normalização
            scaler = StandardScaler()
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
            # Remoção de variância zero
            variances = data[numeric_columns].var()
            zero_variance_cols = variances[variances == 0].index
            data.drop(columns=zero_variance_cols, inplace=True)
    
            # Codificação
            encoder = LabelEncoder()
            data['Classificação'] = encoder.fit_transform(data['Classificação'])
            data['Paciente'] = encoder.fit_transform(data['Paciente'])
    
            # Divisão dos dados
            X = data.drop(columns=['Classificação', 'Arquivo'])
            y = data['Classificação']
    
            splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y, groups=X['Paciente']))
    
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
            # Garantia de separação por paciente
            overlap = set(X_train['Paciente']).intersection(set(X_test['Paciente']))
            if overlap:
                raise ValueError(f"Pacientes presentes em ambos os conjuntos: {overlap}")
            else:
                print("Treinamento isolado por paciente está correto.")
    
            # Configuração do pipeline
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k='all')),
                ('sampling', RandomUnderSampler(random_state=42)),
                ('classifier', XGBClassifier(eval_metric="logloss"))
            ])
    
            # Espaço de busca para RandomizedSearchCV
            param_dist = {
                'feature_selection__k': [10, 12, 15, 'all'],
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 4, 5, 6, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__gamma': [0, 0.1, 0.2]
            }
    
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=50,
                cv=GroupKFold(n_splits=5),
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
    
            # Ajuste do RandomizedSearchCV no conjunto de treino
            X_train_no_patient = X_train.drop(columns=['Paciente'])
            train_groups = X_train['Paciente']
            random_search.fit(X_train_no_patient, y_train, groups=train_groups)
    
            # Melhor modelo encontrado
            print("\nMelhores parâmetros encontrados:")
            print(random_search.best_params_)
    
            # Avaliação no conjunto de teste
            X_test_no_patient = X_test.drop(columns=['Paciente'])
            y_pred = random_search.predict(X_test_no_patient)
    
            print("\nRelatório de Classificação - Conjunto de Teste:")
            print(classification_report(y_test, y_pred, target_names=['Saudável', 'Esteatose']))
    
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Saudável', 'Esteatose'])
            disp.plot(cmap='Blues')
            plt.title("Matriz de Confusão - Conjunto de Teste")
            plt.savefig(os.path.join(self.output_folder, 'confusion_matrix.png'))  # Salvar gráfico
            plt.close()
    
            # Cálculo das métricas médias durante a validação cruzada
            accuracies = []
            sensitivities = []
            specificities = []
            all_y_true = []
            all_y_pred = []
    
            for train_index, val_index in GroupKFold(n_splits=5).split(X_train_no_patient, y_train, groups=train_groups):
                X_train_fold, X_val_fold = X_train_no_patient.iloc[train_index], X_train_no_patient.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
                pipeline.set_params(**random_search.best_params_)
                pipeline.fit(X_train_fold, y_train_fold)
                y_pred_fold = pipeline.predict(X_val_fold)
    
                all_y_true.extend(y_val_fold)
                all_y_pred.extend(y_pred_fold)
    
                cm_fold = confusion_matrix(y_val_fold, y_pred_fold)
                if cm_fold.shape == (2, 2):
                    tn, fp, fn, tp = cm_fold.ravel()
                    accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
                    sensitivities.append(recall_score(y_val_fold, y_pred_fold))
                    specificities.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
                else:
                    print(f"Confusion matrix shape {cm_fold.shape} inesperada.")
    
            mean_accuracy = np.mean(accuracies)
            mean_sensitivity = np.mean(sensitivities)
            mean_specificity = np.mean(specificities)
    
            print("\nMétricas médias após validação cruzada:")
            print(f"Acurácia Média: {mean_accuracy:.2f}")
            print(f"Sensibilidade Média: {mean_sensitivity:.2f}")
            print(f"Especificidade Média: {mean_specificity:.2f}")
            
            self.specificities = mean_specificity
            self.sensitivities = mean_sensitivity
            self.accuracies = mean_accuracy
    
            cm_total = confusion_matrix(all_y_true, all_y_pred)
            disp_total = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=['Saudável', 'Esteatose'])
            disp_total.plot(cmap='Blues')
            plt.title("Matriz de Confusão - Validação Cruzada (Agregada)")
            plt.savefig(os.path.join(self.output_folder, 'confusion_matrix_cv.png'))  # Salvar gráfico
            plt.show()
            plt.close()
    
            # Fim do temporizador
            self.end_time = time.time()
            self.execution_time = self.end_time - self.start_time
            
            print(f"\nTempo total de execução: {self.execution_time:.2f} segundos")
        except Exception as e:
            print(f"Erro inesperado: {e}")

#CLASSIFICADOR PROFUNDO (CNN) EfficientNet COMECA AQUI!!!!!
#Classe dataset para carregar as imagens e fazer as transformações
class RoiDataset(Dataset):
    def __init__(self, imagesDir, transform=None, includeRotations=False):
        self.imagesDir = imagesDir
        self.transform = transform
        self.includeRotations = includeRotations
        self.images = []
        self.labels = []
        self.patients = []
        self.rotations = []
        self.imagesPerPatient = defaultdict(int)

        print(f'Preparando o dataset: {imagesDir}')
        pattern = r'ROI_(\d+)_(\d+)\.png'

        self.loadImagesFromDir(self.imagesDir, pattern, rotationDegree=0)

        #Pra testar e melhor com menos imagens pq fica mais rapido, mas pra treinar tem que ser com mais imagens
        #pra tentar evitar overfitting
        if self.includeRotations:
            rotationAngles = [90, 180, 270]
            for angle in rotationAngles:
                rotationDir = os.path.join(self.imagesDir, f'Rotation_{angle}')
                self.prepareRotatedImages(rotationDir, angle)
                self.loadImagesFromDir(rotationDir, pattern, rotationDegree=angle)

        print(f'\nTotal de imagens carregadas: {len(self.images)}')
        print(f'Pacientes únicos encontrados: {sorted(set(self.patients))}')
        print(f'Número de imagens por paciente: {dict(self.imagesPerPatient)}')

    #So carrega as imagens
    def loadImagesFromDir(self, directory, pattern, rotationDegree):
        for filename in sorted(os.listdir(directory)):
            filePath = os.path.join(directory, filename)
            
            if os.path.isfile(filePath) and filename.endswith('.png'):
                match = re.match(pattern, filename)
                
                #Ja adiciona os labels pra cada imagem e paciente de acordo com o mencionado na descricao do trabalho
                if match:
                    patientNum = int(match.group(1))
                    self.patients.append(patientNum)
                    self.images.append(filePath)
                    self.rotations.append(rotationDegree)
                    self.imagesPerPatient[patientNum] += 1
                    self.labels.append(0 if patientNum <= 15 else 1)
                
                else:
                    print(f'Nome de arquivo inesperado: {filename}')
    
    #Rotaciona as imagens obtidas para aumentar "Artificialmente" o dataset
    def prepareRotatedImages(self, rotationDir, angle):
        
        if not os.path.exists(rotationDir):
            os.makedirs(rotationDir)
            print(f'Criando diretório de rotação: {rotationDir}')
            
        if not any(filename.endswith('.png') for filename in os.listdir(rotationDir)):
            print(f'Rotacionando imagens em {angle} graus...')
            
            for filename in sorted(os.listdir(self.imagesDir)):
            
                filePath = os.path.join(self.imagesDir, filename)
                if os.path.isfile(filePath) and filename.endswith('.png') and 'Rotation_' not in filename:
                
                    img = Image.open(filePath)
                    rotatedImg = img.rotate(angle, expand=True)
                    
                    #Salva em memoria secundaria pra nao precisar ficar gerando denovo e denovo e denovo e denovo...
                    rotatedImg.save(os.path.join(rotationDir, filename))
        #Verifica se ja tem imagens rodadas pra economizar tempo e processamento.
        else:
            print(f'Imagens já rotacionadas em {angle} graus.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath = self.images[idx]
        image = Image.open(imgPath).convert('L')
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return image, self.labels[idx], self.patients[idx], self.rotations[idx]

#Treinamento com validação cruzada do modelo da EfficientNet
class CrossValidationTraining:
    def __init__(self, dataset, trainingsArray, numEpochs=10, batchSize=16, learningRate=0.001, experimentName="Experiment", earlyStoppingPatience=5):
        
        self.dataset = dataset
        self.trainingsArray = trainingsArray
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.experimentName = experimentName
        self.totalTestImages = 0
        self.earlyStoppingPatience = earlyStoppingPatience
        
        #O codigo consegue treinar varios modelos um atras do outro, entao cria um diretorio armazenar as pastas de cada um
        #pra nao misturar tudo
        self.experimentDir = os.path.join('Training Iterations', self.experimentName)
        
        #Lugar pra guardar o modelo e resultados do trinamnto
        self.finalResultsDir = os.path.join(self.experimentDir)
        
        #Seleciona GPU ou CPU como dispositivo pra treinar, mas pqp que demora pra treinar na CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #Leave one Group Out para treinamento cruzado. Deixa um grupo de fora e treina com o resto.
        self.logo = LeaveOneGroupOut()

        self.metrics = {
            'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': [], 'aurocs': [],
            'sensitivities': [], 'specificities': [], 'confusion_matrices': [],
            'train_accuracies_per_epoch': [], 'val_accuracies_per_epoch': [],
            'train_losses_per_epoch': [], 'val_losses_per_epoch': [], 'training_times': [],
        }

        
        # Criação do diretórios
        os.makedirs(self.finalResultsDir, exist_ok=True)
        self.confusionMatricesDir = os.path.join(self.finalResultsDir, 'ConfusionMatricesPerEpoch')
        os.makedirs(self.confusionMatricesDir, exist_ok=True)

    def train(self):
        print(f'Usando dispositivo: {self.device}')
        print('\nIniciando o treinamento com validação cruzada...\n')
        
        totalFolds = len(set(self.dataset.patients))
        
        for fold, (trainIdx, testIdx) in enumerate(self.logo.split(np.arange(len(self.dataset)), groups=self.dataset.patients)):
            startTime = time.time()
            
            testPatient = self.dataset.patients[testIdx[0]]
            testPatientIndices = [idx for idx in testIdx if self.dataset.patients[idx] == testPatient]
            if not testPatientIndices:
                print(f'Aviso: Conjunto de teste vazio para o fold {fold + 1}. Pulando este fold.')
            
                continue
            print(f'\nFold {fold + 1}/{totalFolds}:')
            print(f' - Número de imagens de teste: {len(testPatientIndices)}')
            
            #Balanceamento do conjunto de treino, pra nao ficar muito pra um lado ou pro outro
            trainIndices = [idx for idx in trainIdx if self.dataset.patients[idx] != testPatient]
            balancedTrainIdx = self.balanceDataset(trainIndices)
            trainSubset = Subset(self.dataset, balancedTrainIdx)
            testSubset = Subset(self.dataset, testPatientIndices)
            print(f' - Número de imagens de treino balanceadas: {len(trainSubset)}')
            
            #Carrega os dados de treino e teste
            trainLoader = DataLoader(trainSubset, batch_size=self.batchSize, shuffle=True)
            testLoader = DataLoader(testSubset, batch_size=self.batchSize, shuffle=False)
            model = self.initializeModel()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
            
            # Criação do diretório específico para o fold atual
            foldConfusionDir = os.path.join(self.confusionMatricesDir, f'fold_{fold + 1}')
            os.makedirs(foldConfusionDir, exist_ok=True)
            self.trainModel(
                model, criterion, optimizer, trainLoader, testLoader,
                fold, totalFolds, testPatient, startTime, foldConfusionDir
            )
        
        #Salva os resultados finais e pegar o modelo treinado
        self.saveResults()
        torch.save(model.state_dict(), os.path.join(self.finalResultsDir, 'final_model.pth'))
        print(f'Modelo final salvo em {os.path.join(self.finalResultsDir, "final_model.pth")}')

    #Funcao pra balancear o dataset de forma aleatoria
    def balanceDataset(self, trainIndices):
        
        X = np.array(trainIndices).reshape(-1, 1)
        y = np.array([self.dataset.labels[idx] for idx in trainIndices])
        
        rus = RandomUnderSampler()
        XResampled, _ = rus.fit_resample(X, y)
        return XResampled.flatten().tolist()


    def initializeModel(self):
        #Inicializa o modelo EfficientNet com pesos pre-treinados na ImageNet
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        numFtrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(numFtrs, 2)
        return model.to(self.device)

    #O treinamento do modelo em si
    def trainModel(self, model, criterion, optimizer, trainLoader, testLoader, fold, totalFolds, testPatient, startTime, foldConfusionDir):
        
        #Barra de progresso porque fica mais facil de ver no terminal
        progressBar = tqdm(total=self.numEpochs * len(trainLoader), desc=f'Treinando Fold {fold + 1}/{totalFolds}')
        bestValLoss = np.inf
        
        #Usado como paciencia pro early stopping
        epochsNoImprove = 0
        
        metricsPerEpoch = self.initializeEpochMetrics()
        
        #Armazena o melhor modelo
        bestModelWts = None
        
        #Faz o treinamento e validacao do modelo por ecopa
        for epoch in range(self.numEpochs):
            self.runEpoch(model, criterion, optimizer, trainLoader, progressBar, epoch, metricsPerEpoch, mode='train')
            self.runEpoch(model, criterion, optimizer, testLoader, progressBar, epoch, metricsPerEpoch, mode='val')
            
            # Computar e salvar a matriz de confusão após a época de validação
            cm = confusion_matrix(metricsPerEpoch['all_labels'], metricsPerEpoch['all_preds'], labels=[0, 1])
            
            # Gerar a matriz de confusão
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Fold {fold + 1} - Época {epoch + 1} - Matriz de Confusão')
            plt.colorbar()
            tickMarks = np.arange(2)
            plt.xticks(tickMarks, ['Saudável', 'Não Saudável'], rotation=45)
            plt.yticks(tickMarks, ['Saudável', 'Não Saudável'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            
            # Salvar a matriz de confusão como imagem
            confusionMatrixPath = os.path.join(foldConfusionDir, f'epoch_{epoch + 1}.png')
            plt.savefig(confusionMatrixPath)
            plt.close()
            # Fim da adição

            #Funcao de early stopping
            valEpochLoss = metricsPerEpoch['val_loss'][-1]
            if valEpochLoss < bestValLoss:
                bestValLoss = valEpochLoss
                epochsNoImprove = 0
                bestModelWts = model.state_dict()
            else:
                epochsNoImprove += 1
                if epochsNoImprove >= self.earlyStoppingPatience:
                    print(f'\nEarly stopping at epoch {epoch + 1}')
                    if bestModelWts is not None:
                        model.load_state_dict(bestModelWts)
                    break
                
        #Fechando a barra de progresso        
        progressBar.close()
        
        #Metricas metricas e mais metricas
        self.metrics['training_times'].append(time.time() - startTime)
        allPreds = metricsPerEpoch['all_preds']
        allLabels = metricsPerEpoch['all_labels']
        allProbs = metricsPerEpoch['all_probs']
        precision = precision_score(allLabels, allPreds, zero_division=0)
        recall = recall_score(allLabels, allPreds, zero_division=0)
        f1 = f1_score(allLabels, allPreds, zero_division=0)
        auc = roc_auc_score(allLabels, allProbs) if len(np.unique(allLabels)) > 1 else 0.0
        cm = confusion_matrix(allLabels, allPreds, labels=[0, 1])
        self.metrics['precisions'].append(precision)
        self.metrics['recalls'].append(recall)
        self.metrics['f1_scores'].append(f1)
        self.metrics['aurocs'].append(auc)
        self.metrics['confusion_matrices'].append(cm)
        self.metrics['train_accuracies_per_epoch'].append(metricsPerEpoch['train_acc'])
        self.metrics['val_accuracies_per_epoch'].append(metricsPerEpoch['val_acc'])
        self.metrics['train_losses_per_epoch'].append(metricsPerEpoch['train_loss'])
        self.metrics['val_losses_per_epoch'].append(metricsPerEpoch['val_loss'])
        self.updateOverallMetrics(metricsPerEpoch, testPatient, fold)

    #Armazena as Metricas que sao medidas por epoca
    def initializeEpochMetrics(self):
        return {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'all_preds': [],
            'all_labels': [],
            'all_probs': [],
        }

    #Funcao que roda o modelo por epoca
    def runEpoch(self, model, criterion, optimizer, loader, progressBar, epoch, metrics, mode='train'):
        isTrain = mode == 'train'
        model.train() if isTrain else model.eval()
        runningCorrects = 0
        runningLoss = 0.0
        allPreds = []
        allLabels = []
        allProbs = []
        
        
        for inputs, labels, _, _ in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            if isTrain:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(isTrain):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if isTrain:
                    loss.backward()
                    optimizer.step()
            runningCorrects += torch.sum(preds == labels.data)
            runningLoss += loss.item() * inputs.size(0)
            
            #UPDATE DA BARRA DE PROGRESSO
            progressBar.update(1)
            progressBar.set_postfix({'Loss': loss.item(), 'Época': epoch + 1})
            
            #Modo de Validacao
            if not isTrain:
                #adiciona as predicoes, labels e probabilidades
                allPreds.extend(preds.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())
                
                allProbs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        
        #Calcula as metricas de acuracia e perda
        epochLoss = runningLoss / len(loader.dataset)
        epochAcc = runningCorrects.double() / len(loader.dataset)
        metrics[f'{mode}_acc'].append(epochAcc.item())
        metrics[f'{mode}_loss'].append(epochLoss)
        
        
        if not isTrain:
            metrics['all_preds'] = allPreds
            metrics['all_labels'] = allLabels
            metrics['all_probs'] = allProbs
    
    #Atualiza as metricas
    def updateOverallMetrics(self, metricsPerEpoch, testPatient, fold):
        #Adiciona as metricas de acuracia, sensibilidade e especificidade
        valAcc = metricsPerEpoch['val_acc'][-1] if metricsPerEpoch['val_acc'] else 0
        self.metrics['accuracies'].append(valAcc)
        
        cm = self.metrics['confusion_matrices'][-1]
        TP, FN, TN, FP = cm[1, 1], cm[1, 0], cm[0, 0], cm[0, 1]
        
        #Sensibilidade e Especificidade
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        #Adiciona nos arrays de metricas
        self.metrics['sensitivities'].append(sensitivity)
        self.metrics['specificities'].append(specificity)
        
        print(f"Paciente {testPatient} - Acurácia no Teste: {valAcc:.4f}")
        print(f"Tempo de treinamento para o Fold {fold + 1}: {self.metrics['training_times'][-1]:.2f} segundos")

    def saveResults(self):
        self.saveAggregatedMetrics()

        # Defina as chaves que contêm métricas escalares por fold
        scalarKeys = ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities', 'training_times']

        # Calcule a média e o desvio padrão apenas para as métricas escalares
        meanMetrics = {
            key: (np.mean(values), np.std(values))
            for key, values in self.metrics.items()
            if key in scalarKeys and isinstance(values, list) and values
        }

        # Salva os resultados de cada treinamento
        self.trainingsArray.append(meanMetrics['accuracies'][0])

        # Salva as métricas agregadas
        if self.metrics['confusion_matrices']:
            totalCm = np.sum(self.metrics['confusion_matrices'], axis=0)
            self.saveAggregatedConfusionMatrix(totalCm)

        #Funcao pra salvar as metricas "Gerais"
        self.saveOverallMetrics(meanMetrics)

        print(f'\nProcesso concluído. Resultados salvos na pasta {self.experimentName}.')
        print('\nMétricas após Validação Cruzada:')
        
        for metric in ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities']:
            mean, std = meanMetrics[metric]
            print(f'{metric.capitalize()[:-1]} Média: {mean:.4f} ± {std:.4f}')

    def saveAggregatedMetrics(self):
        metrics = ['Accuracies', 'Losses']
        for metricName in metrics:
            metricKey = metricName.lower()
            trainValuesList = self.metrics[f'train_{metricKey}_per_epoch']
            valValuesList = self.metrics[f'val_{metricKey}_per_epoch']
            
            if not trainValuesList or not valValuesList:
                continue
            
            maxEpochs = max(len(values) for values in trainValuesList)
            meanTrainValues = np.mean([np.pad(values, (0, maxEpochs - len(values)), 'constant') for values in trainValuesList], axis=0)
            meanValValues = np.mean([np.pad(values, (0, maxEpochs - len(values)), 'constant') for values in valValuesList], axis=0)
            self.saveMeanMetricGraph(metricName, meanTrainValues, meanValValues)

    #Salva o grafico de metricas medias
    def saveMeanMetricGraph(self, metricName, meanTrainValues, meanValValues):
        graphPath = os.path.join(self.finalResultsDir, 'Graphs', f'mean_{metricName.lower()}.png')
        os.makedirs(os.path.dirname(graphPath), exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(meanTrainValues) + 1), meanTrainValues, label='Treino')
        plt.plot(range(1, len(meanValValues) + 1), meanValValues, label='Validação')
        plt.xlabel('Época')
        plt.ylabel(metricName)
        plt.title(f'Gráfico Médio de {metricName}')
        plt.legend()
        plt.savefig(graphPath)
        plt.show()
        plt.close()

    #Salva a matriz de confusao agregada
    def saveAggregatedConfusionMatrix(self, totalCm):
        plt.figure()
        plt.imshow(totalCm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão Agregada')
        plt.colorbar()
        tickMarks = np.arange(2)
        plt.xticks(tickMarks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tickMarks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        thresh = totalCm.max() / 2.
        for i, j in np.ndindex(totalCm.shape):
            plt.text(j, i, format(totalCm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if totalCm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(self.finalResultsDir, 'aggregated_confusion_matrix.png'))
        plt.show()
        plt.close()
        np.savetxt(os.path.join(self.finalResultsDir, 'aggregated_confusion_matrix.csv'), totalCm, delimiter=',', fmt='%d')

    #Salva as metricas gerais
    def saveOverallMetrics(self, meanMetrics):
        with open(os.path.join(self.finalResultsDir, 'metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fold', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Sensibilidade', 'Especificidade', 'Tempo de Treinamento (s)'])
            for idx in range(len(self.metrics['accuracies'])):
                writer.writerow([
                    idx + 1,
                    self.metrics['accuracies'][idx],
                    self.metrics['precisions'][idx],
                    self.metrics['recalls'][idx],
                    self.metrics['f1_scores'][idx],
                    self.metrics['aurocs'][idx],
                    self.metrics['sensitivities'][idx],
                    self.metrics['specificities'][idx],
                    self.metrics['training_times'][idx]
                ])
            writer.writerow([])
            writer.writerow(['Média'] + [meanMetrics[key][0] for key in ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities', 'training_times']])
            writer.writerow(['Desvio Padrão'] + [meanMetrics[key][1] for key in ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities', 'training_times']])

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

        self.results_text = None
        self.graphs_frame = None
        self.test_dir_model = None        
        self.train_dir_entry = None
        self.test_dir_entry = None
        self.epochs_entry = None
        self.lr_entry = None
        self.batch_entry = None

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
        
        self.openTrainTestWindow = Button(self.app, width=20, text='DEEP MODEL', font='none 12', command=self.openTrainTestWindowFuncDeep)
        self.openTrainTestWindow.grid(row=4, column=0, sticky="n", pady=10,columnspan=3)
        
        self.openTrainTestWindow = Button(self.app, width=20, text='SHALLOW MODEL', font='none 12', command=self.openTrainTestWindowFuncShallow)
        self.openTrainTestWindow.grid(row=5, column=0, sticky="n", pady=10,columnspan=3)

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

    # Método para abrir a janela de treinamento/teste
    def openTrainTestWindowFuncDeep(self):
        trainTestWindow = Toplevel(self.app)
        trainTestWindow.title("Train/Test AI Model")
        trainTestWindow.geometry("800x600")

        # Diretórios de Dados
        train_dir_label = Label(trainTestWindow, text="Training Data Directory:", font='none 12')
        train_dir_label.pack(pady=(20, 5))
        self.train_dir_entry = Entry(trainTestWindow, width=60)
        self.train_dir_entry.pack(pady=5)
        train_dir_button = Button(trainTestWindow, text="Select Training Data Directory", command=self.selectTrainDir)
        train_dir_button.pack(pady=5)

        test_dir_label = Label(trainTestWindow, text="Testing Data Directory:", font='none 12')
        test_dir_label.pack(pady=(20, 5))
        self.test_dir_entry = Entry(trainTestWindow, width=60)
        self.test_dir_entry.pack(pady=5)
        test_dir_button = Button(trainTestWindow, text="Select Testing Data Directory", command=self.selectTestDir)
        test_dir_button.pack(pady=5)
        
        test_model = Label(trainTestWindow, text="Testing Model Directory:", font='none 12')
        test_model.pack(pady=(20, 5))
        self.test_dir_model = Entry(trainTestWindow, width=60)
        self.test_dir_model.pack(pady=5)
        test_dir_button = Button(trainTestWindow, text="Select Testing Model Directory", command=self.selectTestModel)
        test_dir_button.pack(pady=5)

        # Hiperparâmetros
        hyperparams_label = Label(trainTestWindow, text="Hyperparameters:", font='none 14 bold')
        hyperparams_label.pack(pady=(30, 10))

        # Épocas
        epochs_label = Label(trainTestWindow, text="Epochs:", font='none 12')
        epochs_label.pack(pady=5)
        self.epochs_entry = Entry(trainTestWindow, width=20)
        self.epochs_entry.pack(pady=5)
        self.epochs_entry.insert(0, "10")  # Valor padrão

        # Taxa de Aprendizado
        lr_label = Label(trainTestWindow, text="Learning Rate:", font='none 12')
        lr_label.pack(pady=5)
        self.lr_entry = Entry(trainTestWindow, width=20)
        self.lr_entry.pack(pady=5)
        self.lr_entry.insert(0, "0.001")  # Valor padrão

        # Tamanho do Batch
        batch_label = Label(trainTestWindow, text="Batch Size:", font='none 12')
        batch_label.pack(pady=5)
        self.batch_entry = Entry(trainTestWindow, width=20)
        self.batch_entry.pack(pady=5)
        self.batch_entry.insert(0, "32")  # Valor padrão

        # Botões de Treinamento e Teste
        buttons_frame = Frame(trainTestWindow)
        buttons_frame.pack(pady=20)

        train_button = Button(buttons_frame, text="Train", width=15, command=lambda: self.trainModelFromWindow(trainTestWindow))
        train_button.grid(row=0, column=0, padx=10)

        test_button = Button(buttons_frame, text="Test", width=15, command=lambda: self.testModelFromWindow(trainTestWindow))
        test_button.grid(row=0, column=1, padx=10)

        # Área para exibir resultados
        results_label = Label(trainTestWindow, text="Test Results:", font='none 14 bold')
        results_label.pack(pady=(20, 5))

        self.results_text = Text(trainTestWindow, width=80, height=10, wrap='word')
        self.results_text.pack(pady=5)

    def selectTrainDir(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.train_dir_entry.delete(0, tk.END)
            self.train_dir_entry.insert(0, selected_dir)
    
    def selectTestDir(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.test_dir_entry.delete(0, tk.END)
            self.test_dir_entry.insert(0, selected_dir)
            
    def selectTestModel(self):
        selected_file = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pth")])
        if selected_file:
            self.test_dir_model.delete(0, tk.END)
            self.test_dir_model.insert(0, selected_file)
    
    # Função para iniciar o treinamento a partir da janela
    def trainModelFromWindow(self, window):
        train_dir = self.train_dir_entry.get()
        if not os.path.isdir(train_dir):
            self.displayMessage(window, "Invalid Training Directory", "red")
            return

        try:
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_entry.get())
            learning_rate = float(self.lr_entry.get())
        except ValueError:
            self.displayMessage(window, "Invalid Hyperparameters", "red")
            return

        # Executar o treinamento em uma thread separada
        self.trainModel(train_dir, epochs, batch_size, learning_rate, window)

    def testModelFromWindow(self, window):
        test_dir = self.test_dir_entry.get()
        if not os.path.isdir(test_dir):
            self.displayMessage(window, "Invalid Testing Directory", "red")
            return

        # Executar o teste em uma thread separada
        self.testModel(test_dir, window)

    def displayMessage(self, window, message, color="black"):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.tag_add("start", "end-1c linestart", "end")
        self.results_text.tag_config("start", foreground=color)
        self.results_text.see(tk.END)  # Scroll para o final
 
    def trainModel(self, train_dir, epochs, batch_size, learning_rate, window):
        try:
            self.displayMessage(window, f"Starting training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}", "blue")

            seed = "Alexei"
            convertedSeed = sum(ord(char) for char in seed)
            print(f"Converted integer: {convertedSeed}")
            self.setSeed(convertedSeed)

            dataTransforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            dataset = RoiDataset(train_dir, transform=dataTransforms, includeRotations=True)
            trainingsArray = []

            experimentName = f"{epochs}_Epochs_{batch_size}_BatchSize_{learning_rate}_LR"
            #self.displayMessage(window, f"Starting training with hyperparameters: {experimentName}", "blue")

            trainer = CrossValidationTraining(
                dataset,
                trainingsArray,
                numEpochs=epochs,
                batchSize=batch_size,
                learningRate=learning_rate,
                experimentName=experimentName,
                earlyStoppingPatience=5
            )
            trainer.train()

            self.displayMessage(window, f"Training completed. Results: {trainingsArray}", "green")
            self.displayMessage(window, "Training completed successfully! See results in Terminal", "green")
        except Exception as e:
            self.displayMessage(window, f"Error during training: {e}", "red")

    #Usa uma seed pra deixar o treinamento mais deterministico
    #porque deixar o treinamento totalmente aleatorio deu uns resultados
    #muito distoantes um dos outros, a default seed e 42 mas fodasse.
    # NAO NECESSARIAMENTE FICA 100% DETERMINISTICO, MAS FICA MAIS ESTAVEL
    def setSeed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Método de teste (implementação fictícia, ajuste conforme necessário)
    def testModel(self, test_dir, window):
        try:
            self.displayMessage(window, "Loading the model...", "blue")

            # Configuração do dispositivo (CPU ou GPU)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Recriar o modelo exatamente como no treinamento
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)

            # Alterar a última camada para 2 classes
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
            model = model.to(device)

            # Carregar os pesos do modelo treinado
            model_path = self.test_dir_model.get()
            if not os.path.exists(model_path):
                self.displayMessage(window, f"Model file not found at {model_path}", "red")
                return

            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()

            self.displayMessage(window, "Model loaded successfully!", "green")

            # Transformações aplicadas às imagens
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Variáveis para cálculo das métricas
            all_labels = []        # Classes verdadeiras
            all_predictions = []   # Classes preditas
            all_probabilities = [] # Probabilidades preditas

            # Iterar sobre todas as imagens na pasta
            results = []  # Para armazenar os resultados
            pattern = r'ROI_(\d+)_\d+\.png'  # Regex para extrair o número do paciente

            self.displayMessage(window, f"Starting inference on test data in {test_dir}...", "blue")

            for filename in os.listdir(test_dir):
                file_path = os.path.join(test_dir, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Extrair o número do paciente do nome do arquivo
                        match = re.match(pattern, filename)
                        if match:
                            patient_number = int(match.group(1))
                            true_class = 0 if patient_number <= 15 else 1  # Determinar a classe verdadeira
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
                            all_probabilities.append(probabilities.cpu().numpy())

                            # Salvar os resultados
                            results.append((filename, true_class, predicted_class, probabilities.cpu().numpy()))
                            self.displayMessage(
                                window, 
                                f"Image: {filename}, True Class: {true_class}, Predicted Class: {predicted_class}, Probabilities: {probabilities.cpu().numpy()}",
                                "black"
                            )
                        else:
                            self.displayMessage(window, f"Invalid filename format: {filename}. Skipping.", "orange")

                    except Exception as e:
                        self.displayMessage(window, f"Error processing {filename}: {e}", "red")

            # Calcular as métricas
            if all_labels and all_predictions:
                accuracy = accuracy_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions, zero_division=0)
                recall = recall_score(all_labels, all_predictions, zero_division=0)  # Sensibilidade
                f1 = f1_score(all_labels, all_predictions, zero_division=0)
                cm = confusion_matrix(all_labels, all_predictions)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    specificity = 0.0

                self.displayMessage(window, f"\nTotal Accuracy on Test Set: {accuracy:.4f}", "green")
                self.displayMessage(window, f"Precision: {precision:.4f}", "green")
                self.displayMessage(window, f"Sensitivity (Recall): {recall:.4f}", "green")
                self.displayMessage(window, f"Specificity: {specificity:.4f}", "green")
                self.displayMessage(window, f"F1 Score: {f1:.4f}", "green")
            else:
                self.displayMessage(window, "\nCannot calculate metrics due to insufficient data.", "red")

            # Salvar os resultados em um arquivo CSV
            output_csv = 'test_results.csv'
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Filename', 'True Class', 'Predicted Class', 'Probabilities'])
                for result in results:
                    writer.writerow(result)

            self.displayMessage(window, f"\nResults saved to {output_csv}", "blue")

            # Gerar e exibir a matriz de confusão
            if all_labels and all_predictions:
                cm = confusion_matrix(all_labels, all_predictions)
                fig, ax = plt.subplots(figsize=(4, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Fatty'])
                disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
                plt.title("Confusion Matrix")
                plt.show()

        except Exception as e:
            self.displayMessage(window, f"Error during testing: {e}", "red")

    
            
    def displayShallowMessage(self, message, color="black"):
        if hasattr(self, 'shallow_results_text') and self.shallow_results_text:
            self.shallow_results_text.insert(tk.END, message + "\n")
            self.shallow_results_text.tag_add("start", "end-1c linestart", "end")
            self.shallow_results_text.tag_config("start", foreground=color)
            self.shallow_results_text.see(tk.END)  # Scroll para o final
        else:
            print(message)
            
    def trainShallowModelFromWindow(self, window):
        train_dir = self.shallow_train_dir_entry.get()
        
        # Executar o treinamento em uma thread separada
        self.trainShallowModel(train_dir, window)

    def trainShallowModel(self, train_dir, window):
        try:
            self.displayShallowMessage("Starting training...", "blue")

            # Instanciar a classe ShallowModel
            shallow_model = ShallowModel()
            shallow_model.output_folder = './XGBoost'  # Definir a pasta de saída
            shallow_model.run(train_dir)

            # Obter os atributos desejados
            execution_time = shallow_model.execution_time
            specificities = shallow_model.specificities
            sensitivities = shallow_model.sensitivities
            accuracies = shallow_model.accuracies

            # Exibir os resultados na interface
            self.displayShallowMessage(f"\nExecution Time: {execution_time:.2f} seconds", "green")
            self.displayShallowMessage(f"Specificity: {specificities:.2f}", "green")
            self.displayShallowMessage(f"Sensitivity: {sensitivities:.2f}", "green")
            self.displayShallowMessage(f"Accuracy: {accuracies:.2f}", "green")
            
            self.displayShallowMessage("Training completed successfully! See results in Terminal", "green")

        except Exception as e:
            self.displayShallowMessage(f"Error during shallow training: {e}", "red")
  
    def selectShallowTrainDir(self):
        selected_dir = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if selected_dir:
            self.shallow_train_dir_entry.delete(0, tk.END)
            self.shallow_train_dir_entry.insert(0, selected_dir)
            
    def openTrainTestWindowFuncShallow(self):
        shallowWindow = Toplevel(self.app)
        shallowWindow.title("Train/Test Shallow Classifier")
        shallowWindow.geometry("800x700")

        # Diretórios de Dados
        train_dir_label = Label(shallowWindow, text="Training Data Directory:", font='none 12')
        train_dir_label.pack(pady=(20, 5))
        self.shallow_train_dir_entry = Entry(shallowWindow, width=60)
        self.shallow_train_dir_entry.pack(pady=5)
        train_dir_button = Button(shallowWindow, text="Select Training Data Directory", command=self.selectShallowTrainDir)
        train_dir_button.pack(pady=5)

        # Botões de Treinamento e Teste
        buttons_frame = Frame(shallowWindow)
        buttons_frame.pack(pady=20)

        train_button = Button(buttons_frame, text="Train Shallow Classifier", width=25, command=lambda: self.trainShallowModelFromWindow(shallowWindow))
        train_button.grid(row=0, column=0, padx=10)

        # Área para exibir resultados
        results_label = Label(shallowWindow, text="Test Results:", font='none 14 bold')
        results_label.pack(pady=(20, 5))

        self.shallow_results_text = Text(shallowWindow, width=80, height=10, wrap='word')
        self.shallow_results_text.pack(pady=5)

    

if ( __name__ == "__main__"):
    root = Tk()
    app = CropApp(root, "./ROISavedFiles", 636, 434)
    root.mainloop()