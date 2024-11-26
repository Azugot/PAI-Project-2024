import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
from tqdm import tqdm
import re
import time
import csv
import random  # Adicionado para amostragem aleatória
from collections import defaultdict  # Para contagem de imagens por paciente


# Função de fixação de seeds
def set_seed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ROIDataset(Dataset):
    def __init__(self, images_dir, transform=None, include_rotations=False):
        self.images_dir = images_dir
        self.transform = transform
        self.include_rotations = include_rotations
        self.images = []
        self.labels = []
        self.patients = []
        self.rotations = []  # Para rastrear o grau de rotação

        print(f'Preparando o dataset: {images_dir}')
        pattern = r'ROI_(\d+)_(\d+)\.png'

        # Contador de imagens por paciente para verificação
        self.images_per_patient = defaultdict(int)

        # Incluir imagens originais
        self._load_images_from_dir(self.images_dir, pattern, rotation_degree=0)

        if self.include_rotations:
            # Verificar e incluir imagens rotacionadas
            rotation_angles = [90, 180, 270]
            for angle in rotation_angles:
                rotation_dir = os.path.join(self.images_dir, f'Rotation_{angle}')
                if not os.path.exists(rotation_dir):
                    os.makedirs(rotation_dir)
                    print(f'Criando diretório de rotação: {rotation_dir}')
                if not self._check_rotated_images_exist(rotation_dir):
                    print(f'Rotacionando imagens em {angle} graus...')
                    self._rotate_and_save_images(angle, rotation_dir)
                else:
                    print(f'Imagens já rotacionadas em {angle} graus.')
                self._load_images_from_dir(rotation_dir, pattern, rotation_degree=angle)

        print(f'\nTotal de imagens carregadas: {len(self.images)}')
        print(f'Pacientes únicos encontrados: {sorted(set(self.patients))}')
        print(f'Número de imagens por paciente: {dict(self.images_per_patient)}')  # Verificação


    def _load_images_from_dir(self, directory, pattern, rotation_degree):
        for filename in sorted(os.listdir(directory)):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith('.png'):
                match = re.match(pattern, filename)
                if match:
                    patient_num = int(match.group(1))
                    image_num = int(match.group(2))
                    self.patients.append(patient_num)
                    self.images.append(file_path)
                    self.rotations.append(rotation_degree)
                    self.images_per_patient[patient_num] += 1

                    if patient_num <= 15:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                else:
                    print(f'Nome de arquivo inesperado: {filename}')

    def _check_rotated_images_exist(self, rotation_dir):
        # Verificar se o diretório contém alguma imagem
        return any(filename.endswith('.png') for filename in os.listdir(rotation_dir))

    def _rotate_and_save_images(self, angle, rotation_dir):
        for filename in sorted(os.listdir(self.images_dir)):
            file_path = os.path.join(self.images_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.png'):
                # Pular diretórios de rotação
                if f'Rotation_' in filename:
                    continue
                img = Image.open(file_path)
                rotated_img = img.rotate(angle, expand=True)
                rotated_img.save(os.path.join(rotation_dir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[idx]
        patient = self.patients[idx]
        rotation = self.rotations[idx]
        return image, label, patient, rotation


class CrossValidationTraining:
    def __init__(self, dataset, test_dataset, trainingsArray, num_epochs=10, batch_size=16, learning_rate=0.001, experiment_name="Experiment", early_stopping_patience=5, test_full_dataset=False):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.experimentNameDirectory = os.path.join('Training Iterations', self.experiment_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logo = LeaveOneGroupOut()
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.aurocs = []
        self.sensitivities = []
        self.specificities = []
        self.confusion_matrices = []
        self.train_accuracies_per_epoch = []
        self.val_accuracies_per_epoch = []
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []
        self.train_precisions_per_epoch = []
        self.val_precisions_per_epoch = []
        self.train_recalls_per_epoch = []
        self.val_recalls_per_epoch = []
        self.train_f1_scores_per_epoch = []
        self.val_f1_scores_per_epoch = []
        self.train_aurocs_per_epoch = []
        self.val_aurocs_per_epoch = []
        self.training_times = []
        self.early_stopping_patience = early_stopping_patience
        self.test_full_dataset = test_full_dataset

        # Inicializar contador de imagens de teste
        self.total_test_images = 0

        # Criar diretórios específicos para este experimento
        self.partial_results_dir = os.path.join(self.experimentNameDirectory, 'PartialResults')
        self.final_results_dir = os.path.join(self.experimentNameDirectory, 'FinalResults')
        os.makedirs(self.partial_results_dir, exist_ok=True)
        os.makedirs(self.final_results_dir, exist_ok=True)

        # Criar subdiretórios em PartialResults
        metrics = ['Accuracies', 'Losses', 'Precisions', 'Recalls', 'F1Scores', 'AUROCs', 'ConfusionMatrices']
        for metric in metrics:
            metric_dir = os.path.join(self.partial_results_dir, metric)
            os.makedirs(metric_dir, exist_ok=True)
            os.makedirs(os.path.join(metric_dir, 'Graphs'), exist_ok=True)
            os.makedirs(os.path.join(metric_dir, 'CSVs'), exist_ok=True)


    def train(self):
        print(f'Usando dispositivo: {self.device}')
        print('\nIniciando o treinamento com validação cruzada...\n')

        # Número total de folds
        total_folds = len(set(self.dataset.patients))

        for fold, (train_idx, test_idx) in enumerate(self.logo.split(np.arange(len(self.dataset)), groups=self.dataset.patients)):
            start_time = time.time()

            # Incluir todas as imagens do paciente de teste
            test_patient = self.dataset.patients[test_idx[0]]
            test_patient_indices = [idx for idx in test_idx if self.dataset.patients[idx] == test_patient]

            if not test_patient_indices:
                print(f'Aviso: Conjunto de teste vazio para o fold {fold + 1}. Pulando este fold.')
                continue

            print(f'\nFold {fold + 1}/{total_folds}:')
            print(f' - Paciente de teste: {test_patient}')
            print(f' - Número de imagens de teste: {len(test_patient_indices)}')

            # Incrementar o contador de imagens de teste
            self.total_test_images += len(test_patient_indices)
            print(f' - Total de imagens de teste acumuladas: {self.total_test_images}')

            # Excluir o paciente de teste do conjunto de treino
            train_indices = [idx for idx in train_idx if self.dataset.patients[idx] != test_patient]

            # **Balancear o conjunto de treino**
            # Separar índices saudáveis e não saudáveis no conjunto de treino
            healthy_train = [idx for idx in train_indices if self.dataset.labels[idx] == 0]
            unhealthy_train = [idx for idx in train_indices if self.dataset.labels[idx] == 1]

            n_healthy = len(healthy_train)
            n_unhealthy_available = len(unhealthy_train)

            if n_unhealthy_available < n_healthy:
                print(f' - Aviso: Não há suficientes imagens não saudáveis para equilibrar o fold {fold + 1}. Usando todas as disponíveis ({n_unhealthy_available}).')
                selected_unhealthy = unhealthy_train
            else:
                random.shuffle(unhealthy_train)
                selected_unhealthy = unhealthy_train[:n_healthy]

            balanced_train_idx = healthy_train + selected_unhealthy

            # Criar Subset com índices de treino balanceados
            train_subset = Subset(self.dataset, balanced_train_idx)
            test_subset = Subset(self.dataset, test_patient_indices)

            print(f' - Número de imagens de treino balanceadas: {len(train_subset)}')
            print(f' - Número total de imagens de teste neste fold: {len(test_subset)}')

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            train_acc_epoch = []
            val_acc_epoch = []
            train_loss_epoch = []
            val_loss_epoch = []
            train_precision_epoch = []
            val_precision_epoch = []
            train_recall_epoch = []
            val_recall_epoch = []
            train_f1_epoch = []
            val_f1_epoch = []
            train_auc_epoch = []
            val_auc_epoch = []

            total_steps = self.num_epochs * len(train_loader)
            progress_bar = tqdm(total=total_steps, desc=f'Treinando Fold {fold + 1}/{total_folds}')

            best_val_loss = np.inf
            epochs_no_improve = 0

            for epoch in range(self.num_epochs):
                model.train()
                running_corrects = 0
                running_loss = 0.0
                all_train_preds = []
                all_train_labels = []
                all_train_probs = []

                for inputs, labels, _, _ in train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_corrects += torch.sum(preds == labels.data)
                    running_loss += loss.item() * inputs.size(0)

                    progress_bar.update(1)
                    progress_bar.set_postfix({'Loss': loss.item(), 'Epoch': epoch + 1})

                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())
                    all_train_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

                epoch_loss = running_loss / len(train_subset)
                epoch_acc = running_corrects.double() / len(train_subset)
                train_acc_epoch.append(epoch_acc.item())
                train_loss_epoch.append(epoch_loss)

                # Calcular métricas adicionais para o conjunto de treino
                train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
                train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
                train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
                if len(np.unique(all_train_labels)) > 1:
                    train_auc = roc_auc_score(all_train_labels, all_train_probs)
                else:
                    train_auc = 0.0
                train_precision_epoch.append(train_precision)
                train_recall_epoch.append(train_recall)
                train_f1_epoch.append(train_f1)
                train_auc_epoch.append(train_auc)

                model.eval()
                val_running_corrects = 0
                val_running_loss = 0.0
                all_preds = []
                all_labels = []
                all_probs = []

                with torch.no_grad():
                    for inputs, labels, _, _ in test_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        val_running_corrects += torch.sum(preds == labels.data)
                        val_running_loss += loss.item() * inputs.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

                if len(test_subset) > 0:
                    val_epoch_acc = val_running_corrects.double() / len(test_subset)
                    val_acc_epoch.append(val_epoch_acc.item())
                    val_epoch_loss = val_running_loss / len(test_subset)
                    val_loss_epoch.append(val_epoch_loss)
                else:
                    val_epoch_acc = 0
                    val_epoch_loss = 0
                    val_acc_epoch.append(val_epoch_acc)
                    val_loss_epoch.append(val_epoch_loss)

                # Calcular métricas adicionais para o conjunto de validação
                val_precision = precision_score(all_labels, all_preds, zero_division=0)
                val_recall = recall_score(all_labels, all_preds, zero_division=0)
                val_f1 = f1_score(all_labels, all_preds, zero_division=0)
                if len(np.unique(all_labels)) > 1:
                    val_auc = roc_auc_score(all_labels, all_probs)
                else:
                    val_auc = 0.0
                val_precision_epoch.append(val_precision)
                val_recall_epoch.append(val_recall)
                val_f1_epoch.append(val_f1)
                val_auc_epoch.append(val_auc)

                # Verificação de early stopping
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    epochs_no_improve = 0
                    # Salvar os melhores pesos do modelo
                    best_model_wts = model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.early_stopping_patience:
                        print(f'\nEarly stopping at epoch {epoch + 1}')
                        # Carregar os melhores pesos do modelo
                        model.load_state_dict(best_model_wts)
                        break

            progress_bar.close()

            # Registrar o tempo de treinamento
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.training_times.append(elapsed_time)

            # Salvar e exibir o gráfico de curva de aprendizado para este fold
            self.save_fold_metrics(fold, test_patient, train_acc_epoch, val_acc_epoch, train_loss_epoch, val_loss_epoch,
                                   train_precision_epoch, val_precision_epoch, train_recall_epoch, val_recall_epoch,
                                   train_f1_epoch, val_f1_epoch, train_auc_epoch, val_auc_epoch)

            if val_acc_epoch:
                final_acc = val_acc_epoch[-1]
                self.accuracies.append(final_acc)
            else:
                print(f'Aviso: Acurácia de validação não disponível para o paciente {test_patient}.')

            if all_labels and all_preds:
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                self.confusion_matrices.append(cm)

                # Calcular sensibilidade e especificidade
                TP = cm[1, 1]
                FN = cm[1, 0]
                TN = cm[0, 0]
                FP = cm[0, 1]

                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

                self.sensitivities.append(sensitivity)
                self.specificities.append(specificity)

                self.precisions.append(val_precision_epoch[-1])
                self.recalls.append(val_recall_epoch[-1])
                self.f1_scores.append(val_f1_epoch[-1])
                self.aurocs.append(val_auc_epoch[-1])

                # Salvar matriz de confusão
                self.save_confusion_matrix(cm, test_patient, fold)
            else:
                print(f'Aviso: Não foi possível gerar a matriz de confusão para o paciente {test_patient}.')

            self.train_accuracies_per_epoch.append(train_acc_epoch)
            self.val_accuracies_per_epoch.append(val_acc_epoch)
            self.train_losses_per_epoch.append(train_loss_epoch)
            self.val_losses_per_epoch.append(val_loss_epoch)
            self.train_precisions_per_epoch.append(train_precision_epoch)
            self.val_precisions_per_epoch.append(val_precision_epoch)
            self.train_recalls_per_epoch.append(train_recall_epoch)
            self.val_recalls_per_epoch.append(val_recall_epoch)
            self.train_f1_scores_per_epoch.append(train_f1_epoch)
            self.val_f1_scores_per_epoch.append(val_f1_epoch)
            self.train_aurocs_per_epoch.append(train_auc_epoch)
            self.val_aurocs_per_epoch.append(val_auc_epoch)

            print(f"Paciente {test_patient} - Acurácia no Teste: {final_acc:.4f}")
            print(f"Tempo de treinamento para o Fold {fold + 1}: {elapsed_time:.2f} segundos")

        self.save_results()
        # Treinar o modelo final e avaliar no conjunto de teste
        self.train_final_model()

    def save_fold_metrics(self, fold, current_patient, train_acc_epoch, val_acc_epoch, train_loss_epoch, val_loss_epoch,
                          train_precision_epoch, val_precision_epoch, train_recall_epoch, val_recall_epoch,
                          train_f1_epoch, val_f1_epoch, train_auc_epoch, val_auc_epoch):
        # Salvar métricas por época em arquivos CSV
        metrics = {
            'Accuracies': (train_acc_epoch, val_acc_epoch),
            'Losses': (train_loss_epoch, val_loss_epoch),
            'Precisions': (train_precision_epoch, val_precision_epoch),
            'Recalls': (train_recall_epoch, val_recall_epoch),
            'F1Scores': (train_f1_epoch, val_f1_epoch),
            'AUROCs': (train_auc_epoch, val_auc_epoch)
        }

        for metric_name, (train_values, val_values) in metrics.items():
            csv_dir = os.path.join(self.partial_results_dir, metric_name, 'CSVs')
            csv_path = os.path.join(csv_dir, f'{metric_name.lower()}_fold_{fold + 1}.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', f'Train {metric_name}', f'Validation {metric_name}'])
                for epoch in range(len(train_values)):
                    writer.writerow([epoch + 1, train_values[epoch], val_values[epoch]])

            # Gerar e salvar gráficos
            graph_dir = os.path.join(self.partial_results_dir, metric_name, 'Graphs')
            plt.figure()
            plt.plot(range(1, len(train_values) + 1), train_values, label='Treino')
            plt.plot(range(1, len(val_values) + 1), val_values, label='Validação')
            plt.xlabel('Época')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} - Fold {fold + 1}')
            plt.legend()
            plt.savefig(os.path.join(graph_dir, f'{metric_name.lower()}_fold_{fold + 1}.png'))
            plt.close()

    def save_confusion_matrix(self, cm, patient_id, fold):
        graph_dir = os.path.join(self.partial_results_dir, 'ConfusionMatrices', 'Graphs')
        csv_dir = os.path.join(self.partial_results_dir, 'ConfusionMatrices', 'CSVs')

        # Salvar como CSV
        csv_path = os.path.join(csv_dir, f'confusion_matrix_patient_{patient_id}_fold_{fold + 1}.csv')
        np.savetxt(csv_path, cm, delimiter=',', fmt='%d')

        # Salvar como imagem
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Matriz de Confusão - Paciente {patient_id}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')

        # Adicionar anotações com os valores
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, f'confusion_matrix_patient_{patient_id}_fold_{fold + 1}.png'))
        plt.close()

    def save_results(self):
        # Salvar métricas agregadas
        self.save_aggregated_metrics()

        # Salvar resultados em arquivos CSV
        self.save_metrics_csv()

        # Calcular média e desvio padrão
        if self.accuracies:
            mean_accuracy = np.mean(self.accuracies)
            std_accuracy = np.std(self.accuracies)
            trainingsArray.append(mean_accuracy)
        else:
            mean_accuracy = None
            std_accuracy = None

        if self.sensitivities:
            mean_sensitivity = np.mean(self.sensitivities)
            std_sensitivity = np.std(self.sensitivities)
        else:
            mean_sensitivity = None
            std_sensitivity = None

        if self.specificities:
            mean_specificity = np.mean(self.specificities)
            std_specificity = np.std(self.specificities)
        else:
            mean_specificity = None
            std_specificity = None

        if self.precisions:
            mean_precision = np.mean(self.precisions)
            std_precision = np.std(self.precisions)
        else:
            mean_precision = None
            std_precision = None

        if self.recalls:
            mean_recall = np.mean(self.recalls)
            std_recall = np.std(self.recalls)
        else:
            mean_recall = None
            std_recall = None

        if self.f1_scores:
            mean_f1 = np.mean(self.f1_scores)
            std_f1 = np.std(self.f1_scores)
        else:
            mean_f1 = None
            std_f1 = None

        if self.aurocs:
            mean_auc = np.mean(self.aurocs)
            std_auc = np.std(self.aurocs)
        else:
            mean_auc = None
            std_auc = None

        # Salvar matriz de confusão agregada
        if self.confusion_matrices:
            total_cm = np.sum(self.confusion_matrices, axis=0)
            total_images_in_cm = np.sum(total_cm)
            print(f'\nTotal de imagens agregadas na matriz de confusão: {total_images_in_cm}')
            print(f'Número esperado de imagens de teste: {self.total_test_images}')
            if total_images_in_cm != self.total_test_images:
                print(f'Erro: A matriz de confusão agregada tem {total_images_in_cm} imagens, mas espera-se {self.total_test_images} imagens.')
            else:
                print('A matriz de confusão agregada está correta.')

            plt.figure()
            plt.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Matriz de Confusão Agregada')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
            plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')

            # Adicionar anotações com os valores
            thresh = total_cm.max() / 2.
            for i in range(total_cm.shape[0]):
                for j in range(total_cm.shape[1]):
                    plt.text(j, i, format(total_cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if total_cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(os.path.join(self.final_results_dir, 'aggregated_confusion_matrix.png'))
            plt.close()

            # Salvar matriz de confusão agregada como CSV
            np.savetxt(os.path.join(self.final_results_dir, 'aggregated_confusion_matrix.csv'), total_cm, delimiter=',', fmt='%d')
        else:
            print('Aviso: Não foi possível gerar a matriz de confusão agregada.')

        # Salvar métricas em arquivo CSV
        with open(os.path.join(self.final_results_dir, 'metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fold', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Sensibilidade', 'Especificidade', 'Tempo de Treinamento (s)'])
            for idx in range(len(self.accuracies)):
                writer.writerow([
                    idx + 1,
                    self.accuracies[idx],
                    self.precisions[idx],
                    self.recalls[idx],
                    self.f1_scores[idx],
                    self.aurocs[idx],
                    self.sensitivities[idx],
                    self.specificities[idx],
                    self.training_times[idx]
                ])
            writer.writerow([])
            writer.writerow(['Média', mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, mean_sensitivity, mean_specificity, np.mean(self.training_times)])
            writer.writerow(['Desvio Padrão', std_accuracy, std_precision, std_recall, std_f1, std_auc, std_sensitivity, std_specificity, np.std(self.training_times)])

        print(f'\nProcesso concluído. Resultados salvos na pasta {self.experiment_name}.')

        if mean_accuracy is not None:
            print(f'\nMétricas após Validação Cruzada:')
            print(f'Acurácia Média: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
            print(f'Precisão Média: {mean_precision:.4f} ± {std_precision:.4f}')
            print(f'Recall Médio: {mean_recall:.4f} ± {std_recall:.4f}')
            print(f'F1-Score Médio: {mean_f1:.4f} ± {std_f1:.4f}')
            print(f'AUC-ROC Médio: {mean_auc:.4f} ± {std_auc:.4f}')
            print(f'Sensibilidade Média: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}')
            print(f'Especificidade Média: {mean_specificity:.4f} ± {std_specificity:.4f}')
            print(f'Tempo Médio de Treinamento por Fold: {np.mean(self.training_times):.2f} segundos')
        else:
            print('\nNão foi possível calcular as métricas médias.')

    def save_aggregated_metrics(self):
        # Salvar métricas agregadas por época
        metrics = {
            'Accuracies': (self.train_accuracies_per_epoch, self.val_accuracies_per_epoch),
            'Losses': (self.train_losses_per_epoch, self.val_losses_per_epoch),
            'Precisions': (self.train_precisions_per_epoch, self.val_precisions_per_epoch),
            'Recalls': (self.train_recalls_per_epoch, self.val_recalls_per_epoch),
            'F1Scores': (self.train_f1_scores_per_epoch, self.val_f1_scores_per_epoch),
            'AUROCs': (self.train_aurocs_per_epoch, self.val_aurocs_per_epoch)
        }

        for metric_name, (train_values_list, val_values_list) in metrics.items():
            max_epochs = max(len(values) for values in train_values_list)
            mean_train_values = np.zeros(max_epochs)
            mean_val_values = np.zeros(max_epochs)

            for values in train_values_list:
                mean_train_values[:len(values)] += values
            mean_train_values /= len(train_values_list)

            for values in val_values_list:
                mean_val_values[:len(values)] += values
            mean_val_values /= len(val_values_list)

            # Salvar métricas agregadas em CSV
            csv_path = os.path.join(self.final_results_dir, 'CSVs', f'mean_{metric_name.lower()}.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', f'Mean Train {metric_name}', f'Mean Validation {metric_name}'])
                for epoch in range(len(mean_train_values)):
                    writer.writerow([epoch + 1, mean_train_values[epoch], mean_val_values[epoch]])

            # Gerar e salvar gráficos agregados
            graph_path = os.path.join(self.final_results_dir, 'Graphs', f'mean_{metric_name.lower()}.png')
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            plt.figure()
            plt.plot(range(1, len(mean_train_values) + 1), mean_train_values, label='Treino')
            plt.plot(range(1, len(mean_val_values) + 1), mean_val_values, label='Validação')
            plt.xlabel('Época')
            plt.ylabel(metric_name)
            plt.title(f'Gráfico Médio de {metric_name}')
            plt.legend()
            plt.savefig(graph_path)
            plt.close()

    def save_metrics_csv(self):
        # Salvar métricas por época para cada fold
        metrics = {
            'train_accuracies_per_epoch': self.train_accuracies_per_epoch,
            'val_accuracies_per_epoch': self.val_accuracies_per_epoch,
            'train_losses_per_epoch': self.train_losses_per_epoch,
            'val_losses_per_epoch': self.val_losses_per_epoch,
            'train_precisions_per_epoch': self.train_precisions_per_epoch,
            'val_precisions_per_epoch': self.val_precisions_per_epoch,
            'train_recalls_per_epoch': self.train_recalls_per_epoch,
            'val_recalls_per_epoch': self.val_recalls_per_epoch,
            'train_f1_scores_per_epoch': self.train_f1_scores_per_epoch,
            'val_f1_scores_per_epoch': self.val_f1_scores_per_epoch,
            'train_aurocs_per_epoch': self.train_aurocs_per_epoch,
            'val_aurocs_per_epoch': self.val_aurocs_per_epoch
        }

        for metric_name, values_list in metrics.items():
            csv_path = os.path.join(self.final_results_dir, 'CSVs', f'{metric_name}.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                max_epochs = max(len(values) for values in values_list)
                header = ['Fold'] + [f'Época {i+1}' for i in range(max_epochs)]
                writer.writerow(header)
                for idx, values in enumerate(values_list):
                    row = [idx + 1] + values + [''] * (max_epochs - len(values))
                    writer.writerow(row)

    def train_final_model(self):
        print('\nTreinando o modelo final no conjunto completo de treinamento...')

        # Se a flag test_full_dataset for True, combinar os conjuntos de treino e teste
        if self.test_full_dataset:
            print('Testando o modelo no conjunto completo de dados.')
            combined_dataset = torch.utils.data.ConcatDataset([self.dataset, self.test_dataset])
            train_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            # Excluir imagens rotacionadas do conjunto de teste
            test_indices = [idx for idx in range(len(self.test_dataset)) if self.test_dataset.rotations[idx] == 0]
            test_subset = Subset(self.test_dataset, test_indices)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

            # Conjunto de treino inclui rotações se especificado
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        total_steps = self.num_epochs * len(train_loader)
        progress_bar = tqdm(total=total_steps, desc='Treinando Modelo Final')

        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            model.train()
            running_corrects = 0
            running_loss = 0.0

            for inputs, labels, _, _ in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_corrects += torch.sum(preds == labels.data)
                running_loss += loss.item() * inputs.size(0)

                progress_bar.update(1)
                progress_bar.set_postfix({'Loss': loss.item(), 'Epoch': epoch + 1})

            # Validação no conjunto de teste para early stopping
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels, _, _ in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(test_loader.dataset)

            # Verificação de early stopping
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
                # Salvar os melhores pesos do modelo
                best_model_wts = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    # Carregar os melhores pesos do modelo
                    model.load_state_dict(best_model_wts)
                    break

        progress_bar.close()

        print('\nAvaliando o modelo final no conjunto de teste...')
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels, _, _ in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, zero_division=0)
        test_recall = recall_score(all_labels, all_preds, zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, zero_division=0)
        if len(np.unique(all_labels)) > 1:
            test_auc = roc_auc_score(all_labels, all_probs)
        else:
            test_auc = 0.0

        print(f'Acurácia no Conjunto de Teste: {test_acc:.4f}')
        print(f'Precisão no Conjunto de Teste: {test_precision:.4f}')
        print(f'Recall no Conjunto de Teste: {test_recall:.4f}')
        print(f'F1-Score no Conjunto de Teste: {test_f1:.4f}')
        print(f'AUC-ROC no Conjunto de Teste: {test_auc:.4f}')

        # Gerar a matriz de confusão
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

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
        with open(os.path.join(self.final_results_dir, 'test_set_metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Sensibilidade', 'Especificidade'])
            writer.writerow([test_acc, test_precision, test_recall, test_f1, test_auc, sensitivity, specificity])

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão - Conjunto de Teste')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')

        # Adicionar anotações com os valores
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(self.final_results_dir, 'confusion_matrix_test_set.png'))
        plt.close()

        # Salvar o modelo final
        torch.save(model.state_dict(), os.path.join(self.final_results_dir, 'final_model.pth'))
        print(f'Modelo final salvo em {os.path.join(self.final_results_dir, "final_model.pth")}')

        # Comparar resultados da validação cruzada com os resultados do teste final
        print('\nComparação entre Validação Cruzada e Teste Final:')
        if self.accuracies:
            print(f'Acurácia Média na Validação Cruzada: {np.mean(self.accuracies):.4f}')
        else:
            print('Acurácia Média na Validação Cruzada: N/A')
        print(f'Acurácia no Teste Final: {test_acc:.4f}')

        if self.precisions:
            print(f'Precisão Média na Validação Cruzada: {np.mean(self.precisions):.4f}')
        else:
            print('Precisão Média na Validação Cruzada: N/A')
        print(f'Precisão no Teste Final: {test_precision:.4f}')

        if self.recalls:
            print(f'Recall Médio na Validação Cruzada: {np.mean(self.recalls):.4f}')
        else:
            print('Recall Médio na Validação Cruzada: N/A')
        print(f'Recall no Teste Final: {test_recall:.4f}')

        if self.f1_scores:
            print(f'F1-Score Médio na Validação Cruzada: {np.mean(self.f1_scores):.4f}')
        else:
            print('F1-Score Médio na Validação Cruzada: N/A')
        print(f'F1-Score no Teste Final: {test_f1:.4f}')

        if self.aurocs:
            print(f'AUC-ROC Médio na Validação Cruzada: {np.mean(self.aurocs):.4f}')
        else:
            print('AUC-ROC Médio na Validação Cruzada: N/A')
        print(f'AUC-ROC no Teste Final: {test_auc:.4f}')

        print(f'Sensibilidade Média na Validação Cruzada: {np.mean(self.sensitivities):.4f}')
        print(f'Sensibilidade no Teste Final: {sensitivity:.4f}')
        print(f'Especificidade Média na Validação Cruzada: {np.mean(self.specificities):.4f}')
        print(f'Especificidade no Teste Final: {specificity:.4f}')

    def save_aggregated_metrics(self):
        # Salvar métricas agregadas por época
        metrics = {
            'Accuracies': (self.train_accuracies_per_epoch, self.val_accuracies_per_epoch),
            'Losses': (self.train_losses_per_epoch, self.val_losses_per_epoch),
            'Precisions': (self.train_precisions_per_epoch, self.val_precisions_per_epoch),
            'Recalls': (self.train_recalls_per_epoch, self.val_recalls_per_epoch),
            'F1Scores': (self.train_f1_scores_per_epoch, self.val_f1_scores_per_epoch),
            'AUROCs': (self.train_aurocs_per_epoch, self.val_aurocs_per_epoch)
        }

        for metric_name, (train_values_list, val_values_list) in metrics.items():
            max_epochs = max(len(values) for values in train_values_list)
            mean_train_values = np.zeros(max_epochs)
            mean_val_values = np.zeros(max_epochs)

            for values in train_values_list:
                mean_train_values[:len(values)] += values
            mean_train_values /= len(train_values_list)

            for values in val_values_list:
                mean_val_values[:len(values)] += values
            mean_val_values /= len(val_values_list)

            # Salvar métricas agregadas em CSV
            csv_path = os.path.join(self.final_results_dir, 'CSVs', f'mean_{metric_name.lower()}.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Epoch', f'Mean Train {metric_name}', f'Mean Validation {metric_name}'])
                for epoch in range(len(mean_train_values)):
                    writer.writerow([epoch + 1, mean_train_values[epoch], mean_val_values[epoch]])

            # Gerar e salvar gráficos agregados
            graph_path = os.path.join(self.final_results_dir, 'Graphs', f'mean_{metric_name.lower()}.png')
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            plt.figure()
            plt.plot(range(1, len(mean_train_values) + 1), mean_train_values, label='Treino')
            plt.plot(range(1, len(mean_val_values) + 1), mean_val_values, label='Validação')
            plt.xlabel('Época')
            plt.ylabel(metric_name)
            plt.title(f'Gráfico Médio de {metric_name}')
            plt.legend()
            plt.savefig(graph_path)
            plt.close()

    def save_metrics_csv(self):
        # Salvar métricas por época para cada fold
        metrics = {
            'train_accuracies_per_epoch': self.train_accuracies_per_epoch,
            'val_accuracies_per_epoch': self.val_accuracies_per_epoch,
            'train_losses_per_epoch': self.train_losses_per_epoch,
            'val_losses_per_epoch': self.val_losses_per_epoch,
            'train_precisions_per_epoch': self.train_precisions_per_epoch,
            'val_precisions_per_epoch': self.val_precisions_per_epoch,
            'train_recalls_per_epoch': self.train_recalls_per_epoch,
            'val_recalls_per_epoch': self.val_recalls_per_epoch,
            'train_f1_scores_per_epoch': self.train_f1_scores_per_epoch,
            'val_f1_scores_per_epoch': self.val_f1_scores_per_epoch,
            'train_aurocs_per_epoch': self.train_aurocs_per_epoch,
            'val_aurocs_per_epoch': self.val_aurocs_per_epoch
        }

        for metric_name, values_list in metrics.items():
            csv_path = os.path.join(self.final_results_dir, 'CSVs', f'{metric_name}.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                max_epochs = max(len(values) for values in values_list)
                header = ['Fold'] + [f'Época {i+1}' for i in range(max_epochs)]
                writer.writerow(header)
                for idx, values in enumerate(values_list):
                    row = [idx + 1] + values + [''] * (max_epochs - len(values))
                    writer.writerow(row)

if __name__ == "__main__":

    seed = "Alexei"
    convertedSeed = sum(ord(char) for char in seed)
    print(f"Converted integer: {convertedSeed}")
    
    set_seed(convertedSeed)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset de treinamento (usado na validação cruzada), incluindo rotações
    dataset = ROIDataset('ROISavedFiles', transform=data_transforms, include_rotations=False)
    # Dataset de teste separado, incluindo rotações para garantir todas as imagens
    test_dataset = ROIDataset('TestSet', transform=data_transforms, include_rotations=False)
    
    # Lista de conjuntos de hiperparâmetros
    hyperparameter_sets = [
        {'num_epochs': 2, 'batch_size': 64, 'learning_rate': 0.001},
        # {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.002},
        # {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.004},
    ]
    
    trainingsArray = []
    
    for params in hyperparameter_sets:
        start_time = time.time()
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        # Criar um nome único para o experimento baseado nos hiperparâmetros
        
        experiment_name = f"{num_epochs}_Epochs_{batch_size}_BatchSize_{learning_rate}_LR"
        print(f"\nIniciando treinamento com hiperparâmetros: {experiment_name}")
        trainer = CrossValidationTraining(
            dataset,
            test_dataset,
            trainingsArray,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            experiment_name=experiment_name,
            early_stopping_patience=5,  # Você pode ajustar a paciência conforme necessário
            test_full_dataset=True  # Definido como True para testar no conjunto completo
        )
        trainer.train()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'\nTempo total de treinamento: {elapsed_time:.2f} segundos')
        
    print(trainingsArray)