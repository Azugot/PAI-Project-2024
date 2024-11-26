import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import re
import time
import csv
import random
from collections import defaultdict


# Função de fixação de seeds
def setSeed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ROIDataset(Dataset):
    def __init__(self, imagesDir, transform=None, includeRotations=False):
        self.imagesDir = imagesDir
        self.transform = transform
        self.includeRotations = includeRotations
        self.images = []
        self.labels = []
        self.patients = []
        self.rotations = []

        print(f'Preparando o dataset: {imagesDir}')
        pattern = r'ROI_(\d+)_(\d+)\.png'

        self.imagesPerPatient = defaultdict(int)

        self.loadImagesFromDir(self.imagesDir, pattern, rotationDegree=0)

        if self.includeRotations:

            rotationAngles = [90, 180, 270]
            for angle in rotationAngles:
                rotationDir = os.path.join(self.imagesDir, f'Rotation_{angle}')
                if not os.path.exists(rotationDir):
                    os.makedirs(rotationDir)
                    print(f'Criando diretório de rotação: {rotationDir}')
                if not self.checkRotatedImagesExist(rotationDir):
                    print(f'Rotacionando imagens em {angle} graus...')
                    self.rotateAndSaveImages(angle, rotationDir)
                else:
                    print(f'Imagens já rotacionadas em {angle} graus.')
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
                    imageNum = int(match.group(2))
                    self.patients.append(patientNum)
                    self.images.append(filePath)
                    self.rotations.append(rotationDegree)
                    self.imagesPerPatient[patientNum] += 1

                    if patientNum <= 15:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                else:
                    print(f'Nome de arquivo inesperado: {filename}')

    def checkRotatedImagesExist(self, rotationDir):
        return any(filename.endswith('.png') for filename in os.listdir(rotationDir))

    def rotateAndSaveImages(self, angle, rotationDir):
        for filename in sorted(os.listdir(self.imagesDir)):
            filePath = os.path.join(self.imagesDir, filename)
            if os.path.isfile(filePath) and filename.endswith('.png'):

                if f'Rotation_' in filename:
                    continue
                img = Image.open(filePath)
                rotatedImg = img.rotate(angle, expand=True)
                rotatedImg.save(os.path.join(rotationDir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath = self.images[idx]
        image = Image.open(imgPath).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[idx]
        patient = self.patients[idx]
        rotation = self.rotations[idx]
        return image, label, patient, rotation


class CrossValidationTraining:
    def __init__(self, dataset, testDataset, trainingsArray, numEpochs=10, batchSize=16, learningRate=0.001, experimentName="Experiment", earlyStoppingPatience=5, testFullDataset=False):
        self.dataset = dataset
        self.testDataset = testDataset
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.experimentName = experimentName
        self.experimentNameDirectory = os.path.join('Training Iterations', self.experimentName)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logo = LeaveOneGroupOut()
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1Scores = []
        self.aurocs = []
        self.sensitivities = []
        self.specificities = []
        self.confusionMatrices = []
        self.trainAccuraciesPerEpoch = []
        self.valAccuraciesPerEpoch = []
        self.trainLossesPerEpoch = []
        self.valLossesPerEpoch = []
        self.trainPrecisionsPerEpoch = []
        self.valPrecisionsPerEpoch = []
        self.trainRecallsPerEpoch = []
        self.valRecallsPerEpoch = []
        self.trainF1ScoresPerEpoch = []
        self.valF1ScoresPerEpoch = []
        self.trainAurocsPerEpoch = []
        self.valAurocsPerEpoch = []
        self.trainingTimes = []
        self.earlyStoppingPatience = earlyStoppingPatience
        self.testFullDataset = testFullDataset

        self.totalTestImages = 0

        # Criar diretório separado
        self.partialResultsDir = os.path.join(self.experimentNameDirectory, 'PartialResults')
        self.finalResultsDir = os.path.join(self.experimentNameDirectory, 'FinalResults')
        os.makedirs(self.partialResultsDir, exist_ok=True)
        os.makedirs(self.finalResultsDir, exist_ok=True)

        # Criar subpastas em PartialResults
        metrics = ['Accuracies', 'Losses', 'Precisions', 'Recalls', 'F1Scores', 'AUROCs', 'ConfusionMatrices']
        for metric in metrics:
            metricDir = os.path.join(self.partialResultsDir, metric)
            os.makedirs(os.path.join(metricDir, 'Graphs'), exist_ok=True)
            os.makedirs(os.path.join(metricDir, 'CSVs'), exist_ok=True)

    def train(self):
        print(f'Usando dispositivo: {self.device}')
        print('\nIniciando o treinamento com validação cruzada...\n')

        # Número total de folds
        totalFolds = len(set(self.dataset.patients))

        for fold, (trainIdx, testIdx) in enumerate(self.logo.split(np.arange(len(self.dataset)), groups=self.dataset.patients)):
            startTime = time.time()

            # Incluir todas as imagens do paciente de teste
            testPatient = self.dataset.patients[testIdx[0]]
            testPatientIndices = [idx for idx in testIdx if self.dataset.patients[idx] == testPatient]

            if not testPatientIndices:
                print(f'Aviso: Conjunto de teste vazio para o fold {fold + 1}. Pulando este fold.')
                continue

            print(f'\nFold {fold + 1}/{totalFolds}:')
            print(f' - Paciente de teste: {testPatient}')
            print(f' - Número de imagens de teste: {len(testPatientIndices)}')

            self.totalTestImages += len(testPatientIndices)
            print(f' - Total de imagens de teste acumuladas: {self.totalTestImages}')

            # Excluir o paciente de teste do conjunto de treino
            trainIndices = [idx for idx in trainIdx if self.dataset.patients[idx] != testPatient]

            # **Balancear o conjunto de treino**
            healthyTrain = [idx for idx in trainIndices if self.dataset.labels[idx] == 0]
            unhealthyTrain = [idx for idx in trainIndices if self.dataset.labels[idx] == 1]

            nHealthy = len(healthyTrain)
            nUnhealthyAvailable = len(unhealthyTrain)

            if nUnhealthyAvailable < nHealthy:
                print(f' - Aviso: Não há suficientes imagens não saudáveis para equilibrar o fold {fold + 1}. Usando todas as disponíveis ({nUnhealthyAvailable}).')
                selectedUnhealthy = unhealthyTrain
            else:
                random.shuffle(unhealthyTrain)
                selectedUnhealthy = unhealthyTrain[:nHealthy]

            balancedTrainIdx = healthyTrain + selectedUnhealthy

            # Criar Subset com índices de treino balanceados
            trainSubset = Subset(self.dataset, balancedTrainIdx)
            testSubset = Subset(self.dataset, testPatientIndices)

            print(f' - Número de imagens de treino balanceadas: {len(trainSubset)}')
            print(f' - Número total de imagens de teste neste fold: {len(testSubset)}')

            trainLoader = DataLoader(trainSubset, batch_size=self.batchSize, shuffle=True)
            testLoader = DataLoader(testSubset, batch_size=self.batchSize, shuffle=False)

            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
            numFtrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(numFtrs, 2)
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learningRate)

            trainAccEpoch = []
            valAccEpoch = []
            trainLossEpoch = []
            valLossEpoch = []
            trainPrecisionEpoch = []
            valPrecisionEpoch = []
            trainRecallEpoch = []
            valRecallEpoch = []
            trainF1Epoch = []
            valF1Epoch = []
            trainAucEpoch = []
            valAucEpoch = []

            totalSteps = self.numEpochs * len(trainLoader)
            progressBar = tqdm(total=totalSteps, desc=f'Treinando Fold {fold + 1}/{totalFolds}')

            bestValLoss = np.inf
            epochsNoImprove = 0

            for epoch in range(self.numEpochs):
                model.train()
                runningCorrects = 0
                runningLoss = 0.0
                allTrainPreds = []
                allTrainLabels = []
                allTrainProbs = []

                for inputs, labels, _, _ in trainLoader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    runningCorrects += torch.sum(preds == labels.data)
                    runningLoss += loss.item() * inputs.size(0)

                    progressBar.update(1)
                    progressBar.set_postfix({'Loss': loss.item(), 'Epoch': epoch + 1})

                    allTrainPreds.extend(preds.cpu().numpy())
                    allTrainLabels.extend(labels.cpu().numpy())
                    allTrainProbs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

                epochLoss = runningLoss / len(trainSubset)
                epochAcc = runningCorrects.double() / len(trainSubset)
                trainAccEpoch.append(epochAcc.item())
                trainLossEpoch.append(epochLoss)

                # Calcular métricas para o conjunto de treino
                trainPrecision = precision_score(allTrainLabels, allTrainPreds, zero_division=0)
                trainRecall = recall_score(allTrainLabels, allTrainPreds, zero_division=0)
                trainF1 = f1_score(allTrainLabels, allTrainPreds, zero_division=0)
                if len(np.unique(allTrainLabels)) > 1:
                    trainAuc = roc_auc_score(allTrainLabels, allTrainProbs)
                else:
                    trainAuc = 0.0
                trainPrecisionEpoch.append(trainPrecision)
                trainRecallEpoch.append(trainRecall)
                trainF1Epoch.append(trainF1)
                trainAucEpoch.append(trainAuc)

                model.eval()
                valRunningCorrects = 0
                valRunningLoss = 0.0
                allPreds = []
                allLabels = []
                allProbs = []

                with torch.no_grad():
                    for inputs, labels, _, _ in testLoader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        valRunningCorrects += torch.sum(preds == labels.data)
                        valRunningLoss += loss.item() * inputs.size(0)
                        allPreds.extend(preds.cpu().numpy())
                        allLabels.extend(labels.cpu().numpy())
                        allProbs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

                if len(testSubset) > 0:
                    valEpochAcc = valRunningCorrects.double() / len(testSubset)
                    valAccEpoch.append(valEpochAcc.item())
                    valEpochLoss = valRunningLoss / len(testSubset)
                    valLossEpoch.append(valEpochLoss)
                else:
                    valEpochAcc = 0
                    valEpochLoss = 0
                    valAccEpoch.append(valEpochAcc)
                    valLossEpoch.append(valEpochLoss)

                # Calcular métricas para o conjunto de validação
                valPrecision = precision_score(allLabels, allPreds, zero_division=0)
                valRecall = recall_score(allLabels, allPreds, zero_division=0)
                valF1 = f1_score(allLabels, allPreds, zero_division=0)
                if len(np.unique(allLabels)) > 1:
                    valAuc = roc_auc_score(allLabels, allProbs)
                else:
                    valAuc = 0.0
                valPrecisionEpoch.append(valPrecision)
                valRecallEpoch.append(valRecall)
                valF1Epoch.append(valF1)
                valAucEpoch.append(valAuc)

                # Verificação de early stopping
                if valEpochLoss < bestValLoss:
                    bestValLoss = valEpochLoss
                    epochsNoImprove = 0
                    
                    # Salvar os melhores pesos do modelo
                    bestModelWts = model.state_dict()
                else:
                    epochsNoImprove += 1
                    if epochsNoImprove >= self.earlyStoppingPatience:
                        print(f'\nEarly stopping at epoch {epoch + 1}')
                        
                        # Carregar os melhores pesos do modelo
                        model.load_state_dict(bestModelWts)
                        break

            progressBar.close()

            # Registrar o tempo de treinamento
            endTime = time.time()
            elapsedTime = endTime - startTime
            self.trainingTimes.append(elapsedTime)

            # Salvar o gráfico de curva de aprendizado
            self.saveFoldMetrics(fold, testPatient, trainAccEpoch, valAccEpoch, trainLossEpoch, valLossEpoch,
                                 trainPrecisionEpoch, valPrecisionEpoch, trainRecallEpoch, valRecallEpoch,
                                 trainF1Epoch, valF1Epoch, trainAucEpoch, valAucEpoch)

            if valAccEpoch:
                finalAcc = valAccEpoch[-1]
                self.accuracies.append(finalAcc)
            else:
                print(f'Aviso: Acurácia de validação não disponível para o paciente {testPatient}.')

            if allLabels and allPreds:
                cm = confusion_matrix(allLabels, allPreds, labels=[0, 1])
                self.confusionMatrices.append(cm)

                # Calcular sensibilidade e especificidade
                TP = cm[1, 1]
                FN = cm[1, 0]
                TN = cm[0, 0]
                FP = cm[0, 1]

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

                self.sensitivities.append(sensitivity)
                self.specificities.append(specificity)

                self.precisions.append(valPrecisionEpoch[-1])
                self.recalls.append(valRecallEpoch[-1])
                self.f1Scores.append(valF1Epoch[-1])
                self.aurocs.append(valAucEpoch[-1])

                # Salvar matriz de confusão
                self.saveConfusionMatrix(cm, testPatient, fold)
            else:
                print(f'Aviso: Não foi possível gerar a matriz de confusão para o paciente {testPatient}.')

            self.trainAccuraciesPerEpoch.append(trainAccEpoch)
            self.valAccuraciesPerEpoch.append(valAccEpoch)
            self.trainLossesPerEpoch.append(trainLossEpoch)
            self.valLossesPerEpoch.append(valLossEpoch)
            self.trainPrecisionsPerEpoch.append(trainPrecisionEpoch)
            self.valPrecisionsPerEpoch.append(valPrecisionEpoch)
            self.trainRecallsPerEpoch.append(trainRecallEpoch)
            self.valRecallsPerEpoch.append(valRecallEpoch)
            self.trainF1ScoresPerEpoch.append(trainF1Epoch)
            self.valF1ScoresPerEpoch.append(valF1Epoch)
            self.trainAurocsPerEpoch.append(trainAucEpoch)
            self.valAurocsPerEpoch.append(valAucEpoch)

            print(f"Paciente {testPatient} - Acurácia no Teste: {finalAcc:.4f}")
            print(f"Tempo de treinamento para o Fold {fold + 1}: {elapsedTime:.2f} segundos")

        self.saveResults()
        # Treinar o modelo final e avaliar
        self.trainFinalModel()

    def saveFoldMetrics(self, fold, currentPatient, trainAccEpoch, valAccEpoch, trainLossEpoch, valLossEpoch,
                        trainPrecisionEpoch, valPrecisionEpoch, trainRecallEpoch, valRecallEpoch,
                        trainF1Epoch, valF1Epoch, trainAucEpoch, valAucEpoch):
        
        # Salva em arquivos CSV
        metrics = {
            'Accuracies': (trainAccEpoch, valAccEpoch),
            'Losses': (trainLossEpoch, valLossEpoch),
            'Precisions': (trainPrecisionEpoch, valPrecisionEpoch),
            'Recalls': (trainRecallEpoch, valRecallEpoch),
            'F1Scores': (trainF1Epoch, valF1Epoch),
            'AUROCs': (trainAucEpoch, valAucEpoch)
        }

        for metricName, (trainValues, valValues) in metrics.items():
            csvDir = os.path.join(self.partialResultsDir, metricName, 'CSVs')
            csvPath = os.path.join(csvDir, f'{metricName.lower()}_fold_{fold + 1}.csv')
            with open(csvPath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', f'Train {metricName}', f'Validation {metricName}'])
                for epoch in range(len(trainValues)):
                    writer.writerow([epoch + 1, trainValues[epoch], valValues[epoch]])

            # Gerar e salvar gráficos
            graphDir = os.path.join(self.partialResultsDir, metricName, 'Graphs')
            plt.figure()
            plt.plot(range(1, len(trainValues) + 1), trainValues, label='Treino')
            plt.plot(range(1, len(valValues) + 1), valValues, label='Validação')
            plt.xlabel('Época')
            plt.ylabel(metricName)
            plt.title(f'{metricName} - Fold {fold + 1}')
            plt.legend()
            plt.savefig(os.path.join(graphDir, f'{metricName.lower()}_fold_{fold + 1}.png'))
            plt.close()

    def saveConfusionMatrix(self, cm, patientId, fold):
        graphDir = os.path.join(self.partialResultsDir, 'ConfusionMatrices', 'Graphs')
        csvDir = os.path.join(self.partialResultsDir, 'ConfusionMatrices', 'CSVs')

        # Salvar como CSV
        csvPath = os.path.join(csvDir, f'confusion_matrix_patient_{patientId}_fold_{fold + 1}.csv')
        np.savetxt(csvPath, cm, delimiter=',', fmt='%d')

        # Salvar como imagem
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Matriz de Confusão - Paciente {patientId}')
        plt.colorbar()
        tickMarks = np.arange(2)
        plt.xticks(tickMarks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tickMarks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(graphDir, f'confusion_matrix_patient_{patientId}_fold_{fold + 1}.png'))
        plt.close()

    def saveResults(self):
        # Salvar métricas agregadas
        self.saveAggregatedMetrics()

        self.saveMetricsCsv()

        # Calcular média e desvio padrão
        if self.accuracies:
            meanAccuracy = np.mean(self.accuracies)
            stdAccuracy = np.std(self.accuracies)
            self.trainingsArray.append(meanAccuracy)
        else:
            meanAccuracy = None
            stdAccuracy = None

        if self.sensitivities:
            meanSensitivity = np.mean(self.sensitivities)
            stdSensitivity = np.std(self.sensitivities)
        else:
            meanSensitivity = None
            stdSensitivity = None

        if self.specificities:
            meanSpecificity = np.mean(self.specificities)
            stdSpecificity = np.std(self.specificities)
        else:
            meanSpecificity = None
            stdSpecificity = None

        if self.precisions:
            meanPrecision = np.mean(self.precisions)
            stdPrecision = np.std(self.precisions)
        else:
            meanPrecision = None
            stdPrecision = None

        if self.recalls:
            meanRecall = np.mean(self.recalls)
            stdRecall = np.std(self.recalls)
        else:
            meanRecall = None
            stdRecall = None

        if self.f1Scores:
            meanF1 = np.mean(self.f1Scores)
            stdF1 = np.std(self.f1Scores)
        else:
            meanF1 = None
            stdF1 = None

        if self.aurocs:
            meanAuc = np.mean(self.aurocs)
            stdAuc = np.std(self.aurocs)
        else:
            meanAuc = None
            stdAuc = None

        # Salvar matriz de confusão agregada
        if self.confusionMatrices:
            totalCm = np.sum(self.confusionMatrices, axis=0)
            totalImagesInCm = np.sum(totalCm)
            print(f'\nTotal de imagens agregadas na matriz de confusão: {totalImagesInCm}')
            print(f'Número esperado de imagens de teste: {self.totalTestImages}')
            if totalImagesInCm != self.totalTestImages:
                print(f'Erro: A matriz de confusão agregada tem {totalImagesInCm} imagens, mas espera-se {self.totalTestImages} imagens.')
            else:
                print('A matriz de confusão agregada está correta.')

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
            for i in range(totalCm.shape[0]):
                for j in range(totalCm.shape[1]):
                    plt.text(j, i, format(totalCm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if totalCm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(os.path.join(self.finalResultsDir, 'aggregated_confusion_matrix.png'))
            plt.close()

            # Salvar matriz de confusão agregada como CSV
            np.savetxt(os.path.join(self.finalResultsDir, 'aggregated_confusion_matrix.csv'), totalCm, delimiter=',', fmt='%d')
        else:
            print('Aviso: Não foi possível gerar a matriz de confusão agregada.')

        # Salva arquivo CSV
        with open(os.path.join(self.finalResultsDir, 'metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fold', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Sensibilidade', 'Especificidade', 'Tempo de Treinamento (s)'])
            for idx in range(len(self.accuracies)):
                writer.writerow([
                    idx + 1,
                    self.accuracies[idx],
                    self.precisions[idx],
                    self.recalls[idx],
                    self.f1Scores[idx],
                    self.aurocs[idx],
                    self.sensitivities[idx],
                    self.specificities[idx],
                    self.trainingTimes[idx]
                ])
            writer.writerow([])
            writer.writerow(['Média', meanAccuracy, meanPrecision, meanRecall, meanF1, meanAuc, meanSensitivity, meanSpecificity, np.mean(self.trainingTimes)])
            writer.writerow(['Desvio Padrão', stdAccuracy, stdPrecision, stdRecall, stdF1, stdAuc, stdSensitivity, stdSpecificity, np.std(self.trainingTimes)])

        print(f'\nProcesso concluído. Resultados salvos na pasta {self.experimentName}.')

        if meanAccuracy is not None:
            print(f'\nMétricas após Validação Cruzada:')
            print(f'Acurácia Média: {meanAccuracy:.4f} ± {stdAccuracy:.4f}')
            print(f'Precisão Média: {meanPrecision:.4f} ± {stdPrecision:.4f}')
            print(f'Recall Médio: {meanRecall:.4f} ± {stdRecall:.4f}')
            print(f'F1-Score Médio: {meanF1:.4f} ± {stdF1:.4f}')
            print(f'AUC-ROC Médio: {meanAuc:.4f} ± {stdAuc:.4f}')
            print(f'Sensibilidade Média: {meanSensitivity:.4f} ± {stdSensitivity:.4f}')
            print(f'Especificidade Média: {meanSpecificity:.4f} ± {stdSpecificity:.4f}')
            print(f'Tempo Médio de Treinamento por Fold: {np.mean(self.trainingTimes):.2f} segundos')
        else:
            print('\nNão foi possível calcular as métricas médias.')

    def saveAggregatedMetrics(self):

        metrics = {
            'Accuracies': (self.trainAccuraciesPerEpoch, self.valAccuraciesPerEpoch),
            'Losses': (self.trainLossesPerEpoch, self.valLossesPerEpoch),
            'Precisions': (self.trainPrecisionsPerEpoch, self.valPrecisionsPerEpoch),
            'Recalls': (self.trainRecallsPerEpoch, self.valRecallsPerEpoch),
            'F1Scores': (self.trainF1ScoresPerEpoch, self.valF1ScoresPerEpoch),
            'AUROCs': (self.trainAurocsPerEpoch, self.valAurocsPerEpoch)
        }

        for metricName, (trainValuesList, valValuesList) in metrics.items():
            maxEpochs = max(len(values) for values in trainValuesList)
            meanTrainValues = np.zeros(maxEpochs)
            meanValValues = np.zeros(maxEpochs)

            for values in trainValuesList:
                meanTrainValues[:len(values)] += values
            meanTrainValues /= len(trainValuesList)

            for values in valValuesList:
                meanValValues[:len(values)] += values
            meanValValues /= len(valValuesList)

            csvPath = os.path.join(self.finalResultsDir, 'CSVs', f'mean_{metricName.lower()}.csv')
            os.makedirs(os.path.dirname(csvPath), exist_ok=True)
            with open(csvPath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', f'Mean Train {metricName}', f'Mean Validation {metricName}'])
                for epoch in range(len(meanTrainValues)):
                    writer.writerow([epoch + 1, meanTrainValues[epoch], meanValValues[epoch]])

            # Gerar e salvar gráficos agregados
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

    def saveMetricsCsv(self):
        # Salvar métricas por época para cada fold
        metrics = {
            'train_accuracies_per_epoch': self.trainAccuraciesPerEpoch,
            'val_accuracies_per_epoch': self.valAccuraciesPerEpoch,
            'train_losses_per_epoch': self.trainLossesPerEpoch,
            'val_losses_per_epoch': self.valLossesPerEpoch,
            'train_precisions_per_epoch': self.trainPrecisionsPerEpoch,
            'val_precisions_per_epoch': self.valPrecisionsPerEpoch,
            'train_recalls_per_epoch': self.trainRecallsPerEpoch,
            'val_recalls_per_epoch': self.valRecallsPerEpoch,
            'train_f1_scores_per_epoch': self.trainF1ScoresPerEpoch,
            'val_f1_scores_per_epoch': self.valF1ScoresPerEpoch,
            'train_aurocs_per_epoch': self.trainAurocsPerEpoch,
            'val_aurocs_per_epoch': self.valAurocsPerEpoch
        }

        for metricName, valuesList in metrics.items():
            csvPath = os.path.join(self.finalResultsDir, 'CSVs', f'{metricName}.csv')
            os.makedirs(os.path.dirname(csvPath), exist_ok=True)
            with open(csvPath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                maxEpochs = max(len(values) for values in valuesList)
                header = ['Fold'] + [f'Época {i+1}' for i in range(maxEpochs)]
                writer.writerow(header)
                for idx, values in enumerate(valuesList):
                    row = [idx + 1] + values + [''] * (maxEpochs - len(values))
                    writer.writerow(row)

    def trainFinalModel(self):
        print('\nTreinando o modelo final no conjunto completo de treinamento...')

        # Se a flag testFullDataset for True, combinar os conjuntos de treino e teste
        if self.testFullDataset:
            print('Testando o modelo no conjunto completo de dados.')
            combinedDataset = torch.utils.data.ConcatDataset([self.dataset, self.testDataset])
            trainLoader = DataLoader(combinedDataset, batch_size=self.batchSize, shuffle=True)
            testLoader = DataLoader(combinedDataset, batch_size=self.batchSize, shuffle=False)
        else:
            # Excluir imagens rotacionadas do conjunto de teste
            testIndices = [idx for idx in range(len(self.testDataset)) if self.testDataset.rotations[idx] == 0]
            testSubset = Subset(self.testDataset, testIndices)
            
            testLoader = DataLoader(testSubset, batch_size=self.batchSize, shuffle=False)
            trainLoader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True)

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        numFtrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(numFtrs, 2)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learningRate)

        totalSteps = self.numEpochs * len(trainLoader)
        progressBar = tqdm(total=totalSteps, desc='Treinando Modelo Final')

        bestValLoss = np.inf
        epochsNoImprove = 0

        for epoch in range(self.numEpochs):
            model.train()
            runningCorrects = 0
            runningLoss = 0.0

            for inputs, labels, _, _ in trainLoader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                runningCorrects += torch.sum(preds == labels.data)
                runningLoss += loss.item() * inputs.size(0)

                progressBar.update(1)
                progressBar.set_postfix({'Loss': loss.item(), 'Epoch': epoch + 1})

            # Validação no conjunto de teste para early stopping
            model.eval()
            valRunningLoss = 0.0
            with torch.no_grad():
                for inputs, labels, _, _ in testLoader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valRunningLoss += loss.item() * inputs.size(0)

            valEpochLoss = valRunningLoss / len(testLoader.dataset)

            # Verificação de early stopping
            if valEpochLoss < bestValLoss:
                bestValLoss = valEpochLoss
                epochsNoImprove = 0
                # Salvar os melhores pesos do modelo
                bestModelWts = model.state_dict()
            else:
                epochsNoImprove += 1
                if epochsNoImprove >= self.earlyStoppingPatience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    # Carregar os melhores pesos do modelo
                    model.load_state_dict(bestModelWts)
                    break

        progressBar.close()

        print('\nAvaliando o modelo final no conjunto de teste...')
        model.eval()
        allPreds = []
        allLabels = []
        allProbs = []

        with torch.no_grad():
            for inputs, labels, _, _ in testLoader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                allPreds.extend(preds.cpu().numpy())
                allLabels.extend(labels.cpu().numpy())
                allProbs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        testAcc = accuracy_score(allLabels, allPreds)
        testPrecision = precision_score(allLabels, allPreds, zero_division=0)
        testRecall = recall_score(allLabels, allPreds, zero_division=0)
        testF1 = f1_score(allLabels, allPreds, zero_division=0)
        if len(np.unique(allLabels)) > 1:
            testAuc = roc_auc_score(allLabels, allProbs)
        else:
            testAuc = 0.0

        print(f'Acurácia no Conjunto de Teste: {testAcc:.4f}')
        print(f'Precisão no Conjunto de Teste: {testPrecision:.4f}')
        print(f'Recall no Conjunto de Teste: {testRecall:.4f}')
        print(f'F1-Score no Conjunto de Teste: {testF1:.4f}')
        print(f'AUC-ROC no Conjunto de Teste: {testAuc:.4f}')

        # Gerar a matriz de confusão
        cm = confusion_matrix(allLabels, allPreds, labels=[0, 1])

        # Calcular sensibilidade e especificidade
        TP = cm[1, 1]
        FN = cm[1, 0]
        TN = cm[0, 0]
        FP = cm[0, 1]

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f'Sensibilidade no Conjunto de Teste: {sensitivity:.4f}')
        print(f'Especificidade no Conjunto de Teste: {specificity:.4f}')

        # Salvar métricas no conjunto de teste em CSV
        with open(os.path.join(self.finalResultsDir, 'test_set_metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Sensibilidade', 'Especificidade'])
            writer.writerow([testAcc, testPrecision, testRecall, testF1, testAuc, sensitivity, specificity])

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão - Conjunto de Teste')
        plt.colorbar()
        tickMarks = np.arange(2)
        plt.xticks(tickMarks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tickMarks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(self.finalResultsDir, 'confusion_matrix_test_set.png'))
        plt.close()

        # Salvar o modelo final
        torch.save(model.state_dict(), os.path.join(self.finalResultsDir, 'final_model.pth'))
        print(f'Modelo final salvo em {os.path.join(self.finalResultsDir, "final_model.pth")}')

        # Comparar resultados da validação cruzada com os resultados do teste final
        print('\nComparação entre Validação Cruzada e Teste Final:')
        if self.accuracies:
            print(f'Acurácia Média na Validação Cruzada: {np.mean(self.accuracies):.4f}')
        else:
            print('Acurácia Média na Validação Cruzada: N/A')
        print(f'Acurácia no Teste Final: {testAcc:.4f}')

        if self.precisions:
            print(f'Precisão Média na Validação Cruzada: {np.mean(self.precisions):.4f}')
        else:
            print('Precisão Média na Validação Cruzada: N/A')
        print(f'Precisão no Teste Final: {testPrecision:.4f}')

        if self.recalls:
            print(f'Recall Médio na Validação Cruzada: {np.mean(self.recalls):.4f}')
        else:
            print('Recall Médio na Validação Cruzada: N/A')
        print(f'Recall no Teste Final: {testRecall:.4f}')

        if self.f1Scores:
            print(f'F1-Score Médio na Validação Cruzada: {np.mean(self.f1Scores):.4f}')
        else:
            print('F1-Score Médio na Validação Cruzada: N/A')
        print(f'F1-Score no Teste Final: {testF1:.4f}')

        if self.aurocs:
            print(f'AUC-ROC Médio na Validação Cruzada: {np.mean(self.aurocs):.4f}')
        else:
            print('AUC-ROC Médio na Validação Cruzada: N/A')
        print(f'AUC-ROC no Teste Final: {testAuc:.4f}')

        print(f'Sensibilidade Média na Validação Cruzada: {np.mean(self.sensitivities):.4f}')
        print(f'Sensibilidade no Teste Final: {sensitivity:.4f}')
        print(f'Especificidade Média na Validação Cruzada: {np.mean(self.specificities):.4f}')
        print(f'Especificidade no Teste Final: {specificity:.4f}')

    def saveAggregatedMetrics(self):
        metrics = {
            'Accuracies': (self.trainAccuraciesPerEpoch, self.valAccuraciesPerEpoch),
            'Losses': (self.trainLossesPerEpoch, self.valLossesPerEpoch),
            'Precisions': (self.trainPrecisionsPerEpoch, self.valPrecisionsPerEpoch),
            'Recalls': (self.trainRecallsPerEpoch, self.valRecallsPerEpoch),
            'F1Scores': (self.trainF1ScoresPerEpoch, self.valF1ScoresPerEpoch),
            'AUROCs': (self.trainAurocsPerEpoch, self.valAurocsPerEpoch)
        }

        for metricName, (trainValuesList, valValuesList) in metrics.items():
            maxEpochs = max(len(values) for values in trainValuesList)
            meanTrainValues = np.zeros(maxEpochs)
            meanValValues = np.zeros(maxEpochs)

            for values in trainValuesList:
                meanTrainValues[:len(values)] += values
            meanTrainValues /= len(trainValuesList)

            for values in valValuesList:
                meanValValues[:len(values)] += values
            meanValValues /= len(valValuesList)

            # Salvar métricas agregadas em CSV
            csvPath = os.path.join(self.finalResultsDir, 'CSVs', f'mean_{metricName.lower()}.csv')
            os.makedirs(os.path.dirname(csvPath), exist_ok=True)
            with open(csvPath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', f'Mean Train {metricName}', f'Mean Validation {metricName}'])
                for epoch in range(len(meanTrainValues)):
                    writer.writerow([epoch + 1, meanTrainValues[epoch], meanValValues[epoch]])

            # Gerar e salvar gráficos agregados
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

    def saveMetricsCsv(self):
        # Salvar métricas por época para cada fold
        metrics = {
            'train_accuracies_per_epoch': self.trainAccuraciesPerEpoch,
            'val_accuracies_per_epoch': self.valAccuraciesPerEpoch,
            'train_losses_per_epoch': self.trainLossesPerEpoch,
            'val_losses_per_epoch': self.valLossesPerEpoch,
            'train_precisions_per_epoch': self.trainPrecisionsPerEpoch,
            'val_precisions_per_epoch': self.valPrecisionsPerEpoch,
            'train_recalls_per_epoch': self.trainRecallsPerEpoch,
            'val_recalls_per_epoch': self.valRecallsPerEpoch,
            'train_f1_scores_per_epoch': self.trainF1ScoresPerEpoch,
            'val_f1_scores_per_epoch': self.valF1ScoresPerEpoch,
            'train_aurocs_per_epoch': self.trainAurocsPerEpoch,
            'val_aurocs_per_epoch': self.valAurocsPerEpoch
        }

        for metricName, valuesList in metrics.items():
            csvPath = os.path.join(self.finalResultsDir, 'CSVs', f'{metricName}.csv')
            os.makedirs(os.path.dirname(csvPath), exist_ok=True)
            with open(csvPath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                maxEpochs = max(len(values) for values in valuesList)
                header = ['Fold'] + [f'Época {i+1}' for i in range(maxEpochs)]
                writer.writerow(header)
                for idx, values in enumerate(valuesList):
                    row = [idx + 1] + values + [''] * (maxEpochs - len(values))
                    writer.writerow(row)


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
    
    # Dataset de treinamento (usado na validação cruzada), incluindo rotações
    dataset = ROIDataset('ROISavedFiles', transform=dataTransforms, includeRotations=False)
    # Dataset de teste separado, incluindo rotações para garantir todas as imagens
    testDataset = ROIDataset('TestSet', transform=dataTransforms, includeRotations=False)
    
    # Lista de conjuntos de hiperparâmetros
    hyperparameterSets = [
        {'num_epochs': 2, 'batch_size': 64, 'learning_rate': 0.001},
        # {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.002},
        # {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.004},
    ]
    
    trainingsArray = []
    
    for params in hyperparameterSets:
        start_time = time.time()
        numEpochs = params['num_epochs']
        batchSize = params['batch_size']
        learningRate = params['learning_rate']
        # Criar um nome único para o experimento baseado nos hiperparâmetros
        
        experimentName = f"{numEpochs}_Epochs_{batchSize}_BatchSize_{learningRate}_LR"
        print(f"\nIniciando treinamento com hiperparâmetros: {experimentName}")
        trainer = CrossValidationTraining(
            dataset,
            testDataset,
            trainingsArray,
            numEpochs=numEpochs,
            batchSize=batchSize,
            learningRate=learningRate,
            experimentName=experimentName,
            earlyStoppingPatience=5,  
            testFullDataset=True
        )
        trainer.train()
        
        end_time = time.time()
        elapsedTime = end_time - start_time
        print(f'\nTempo total de treinamento: {elapsedTime:.2f} segundos')
        
    print(trainingsArray)