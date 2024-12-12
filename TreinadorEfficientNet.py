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
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset


def setSeed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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

        if self.includeRotations:
            rotationAngles = [90, 180, 270]
            for angle in rotationAngles:
                rotationDir = os.path.join(self.imagesDir, f'Rotation_{angle}')
                self.prepareRotatedImages(rotationDir, angle)
                self.loadImagesFromDir(rotationDir, pattern, rotationDegree=angle)

        print(f'\nTotal de imagens carregadas: {len(self.images)}')
        print(f'Pacientes únicos encontrados: {sorted(set(self.patients))}')
        print(f'Número de imagens por paciente: {dict(self.imagesPerPatient)}')

    def loadImagesFromDir(self, directory, pattern, rotationDegree):
        for filename in sorted(os.listdir(directory)):
            filePath = os.path.join(directory, filename)
            if os.path.isfile(filePath) and filename.endswith('.png'):
                match = re.match(pattern, filename)
                if match:
                    patientNum = int(match.group(1))
                    self.patients.append(patientNum)
                    self.images.append(filePath)
                    self.rotations.append(rotationDegree)
                    self.imagesPerPatient[patientNum] += 1
                    self.labels.append(0 if patientNum <= 15 else 1)
                else:
                    print(f'Nome de arquivo inesperado: {filename}')

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
                    rotatedImg.save(os.path.join(rotationDir, filename))
        else:
            print(f'Imagens já rotacionadas em {angle} graus.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath = self.images[idx]
        image = Image.open(imgPath).convert('L')
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return image, self.labels[idx], self.patients[idx], self.rotations[idx]


class CrossValidationTraining:
    def __init__(self, dataset, trainingsArray, numEpochs=10, batchSize=16,
                 learningRate=0.001, experimentName="Experiment", earlyStoppingPatience=5):
        self.dataset = dataset
        self.trainingsArray = trainingsArray
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.experimentName = experimentName
        self.experimentDir = os.path.join('Training Iterations', self.experimentName)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logo = LeaveOneGroupOut()
        self.earlyStoppingPatience = earlyStoppingPatience
        self.totalTestImages = 0
        self.finalResultsDir = os.path.join(self.experimentDir, 'FinalResults')

        self.metrics = {
            'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': [], 'aurocs': [],
            'sensitivities': [], 'specificities': [], 'confusion_matrices': [],
            'train_accuracies_per_epoch': [], 'val_accuracies_per_epoch': [],
            'train_losses_per_epoch': [], 'val_losses_per_epoch': [], 'training_times': [],
        }

        os.makedirs(self.finalResultsDir, exist_ok=True)
        # Criação do diretório para salvar as matrizes de confusão por época
        self.confusionMatricesDir = os.path.join(self.finalResultsDir, 'ConfusionMatricesPerEpoch')
        os.makedirs(self.confusionMatricesDir, exist_ok=True)  # Criar o diretório principal

    def initializeModel(self):
        #Inicializa o modelo EfficientNet com pesos pre-treinados na ImageNet
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        numFtrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(numFtrs, 2)
        return model.to(self.device)

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
            trainIndices = [idx for idx in trainIdx if self.dataset.patients[idx] != testPatient]
            balancedTrainIdx = self.balanceDataset(trainIndices)
            trainSubset = Subset(self.dataset, balancedTrainIdx)
            testSubset = Subset(self.dataset, testPatientIndices)
            print(f' - Número de imagens de treino balanceadas: {len(trainSubset)}')
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
                fold, totalFolds, testPatient, startTime, foldConfusionDir  # Passar o diretório do fold
            )
        self.saveResults()
        torch.save(model.state_dict(), os.path.join(self.finalResultsDir, 'final_model.pth'))
        print(f'Modelo final salvo em {os.path.join(self.finalResultsDir, "final_model.pth")}')

    def balanceDataset(self, trainIndices):
        X = np.array(trainIndices).reshape(-1, 1)
        y = np.array([self.dataset.labels[idx] for idx in trainIndices])
        rus = RandomUnderSampler()
        XResampled, _ = rus.fit_resample(X, y)
        return XResampled.flatten().tolist()

    def initialize_model(self):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        return model.to(self.device)

    def trainModel(self, model, criterion, optimizer, trainLoader, testLoader,
                   fold, totalFolds, testPatient, startTime, foldConfusionDir):
        progressBar = tqdm(total=self.numEpochs * len(trainLoader), desc=f'Treinando Fold {fold + 1}/{totalFolds}')
        bestValLoss = np.inf
        epochsNoImprove = 0
        metricsPerEpoch = self.initializeEpochMetrics()
        bestModelWts = None
        for epoch in range(self.numEpochs):
            self.runEpoch(
                model, criterion, optimizer, trainLoader, progressBar, epoch,
                metricsPerEpoch, mode='train'
            )
            self.runEpoch(
                model, criterion, optimizer, testLoader, progressBar, epoch,
                metricsPerEpoch, mode='val'
            )
            # Computar e salvar a matriz de confusão após a época de validação
            cm = confusion_matrix(metricsPerEpoch['all_labels'], metricsPerEpoch['all_preds'], labels=[0, 1])
            # Plotar a matriz de confusão
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
        progressBar.close()
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
            progressBar.update(1)
            progressBar.set_postfix({'Loss': loss.item(), 'Época': epoch + 1})
            if not isTrain:
                allPreds.extend(preds.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())
                allProbs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        epochLoss = runningLoss / len(loader.dataset)
        epochAcc = runningCorrects.double() / len(loader.dataset)
        metrics[f'{mode}_acc'].append(epochAcc.item())
        metrics[f'{mode}_loss'].append(epochLoss)
        if not isTrain:
            metrics['all_preds'] = allPreds
            metrics['all_labels'] = allLabels
            metrics['all_probs'] = allProbs

    def updateOverallMetrics(self, metricsPerEpoch, testPatient, fold):
        valAcc = metricsPerEpoch['val_acc'][-1] if metricsPerEpoch['val_acc'] else 0
        self.metrics['accuracies'].append(valAcc)
        cm = self.metrics['confusion_matrices'][-1]
        TP, FN, TN, FP = cm[1, 1], cm[1, 0], cm[0, 0], cm[0, 1]
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        self.metrics['sensitivities'].append(sensitivity)
        self.metrics['specificities'].append(specificity)
        print(f"Paciente {testPatient} - Acurácia no Teste: {valAcc:.4f}")
        print(f"Tempo de treinamento para o Fold {fold + 1}: {self.metrics['training_times'][-1]:.2f} segundos")

    def saveResults(self):
        self.saveAggregatedMetrics()

        # Defina as chaves que contêm métricas escalares por fold
        scalarKeys = [
            'accuracies', 'precisions', 'recalls', 'f1_scores',
            'aurocs', 'sensitivities', 'specificities', 'training_times'
        ]

        # Calcule a média e o desvio padrão apenas para as métricas escalares
        meanMetrics = {
            key: (np.mean(values), np.std(values))
            for key, values in self.metrics.items()
            if key in scalarKeys and isinstance(values, list) and values
        }

        self.trainingsArray.append(meanMetrics['accuracies'][0])

        if self.metrics['confusion_matrices']:
            totalCm = np.sum(self.metrics['confusion_matrices'], axis=0)
            self.saveAggregatedConfusionMatrix(totalCm)

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
        plt.close()

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
        plt.close()
        np.savetxt(os.path.join(self.finalResultsDir, 'aggregated_confusion_matrix.csv'), totalCm, delimiter=',', fmt='%d')

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


if __name__ == "__main__":
    seed = "Alexei"
    convertedSeed = sum(ord(char) for char in seed)
    print(f"Converted integer: {convertedSeed}")
    setSeed(convertedSeed)
    dataTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = RoiDataset('ROISavedFiles', transform=dataTransforms, includeRotations=True)
    hyperparameterSets = [
        {'num_epochs': 2, 'batch_size': 64, 'learning_rate': 0.001},
        {'num_epochs': 2, 'batch_size': 64, 'learning_rate': 0.002},
        {'num_epochs': 5, 'batch_size': 64, 'learning_rate': 0.001},
        {'num_epochs': 5, 'batch_size': 64, 'learning_rate': 0.002},
        {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.001}
    ]
    trainingsArray = []
    for params in hyperparameterSets:
        startTime = time.time()
        numEpochs = params['num_epochs']
        batchSize = params['batch_size']
        learningRate = params['learning_rate']
        experimentName = f"{numEpochs}_Epochs_{batchSize}_BatchSize_{learningRate}_LR"
        print(f"\nIniciando treinamento com hiperparâmetros: {experimentName}")
        trainer = CrossValidationTraining(
            dataset,
            trainingsArray,
            numEpochs=numEpochs,
            batchSize=batchSize,
            learningRate=learningRate,
            experimentName=experimentName,
            earlyStoppingPatience=5
        )
        trainer.train()
        elapsedTime = time.time() - startTime
        print(f'\nTempo total de treinamento: {elapsedTime:.2f} segundos')
    print(trainingsArray)