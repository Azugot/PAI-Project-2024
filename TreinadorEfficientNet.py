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
        self.rotations = []
        self.images_per_patient = defaultdict(int)

        print(f'Preparando o dataset: {images_dir}')
        pattern = r'ROI_(\d+)_(\d+)\.png'

        self.load_images_from_dir(self.images_dir, pattern, rotation_degree=0)

        if self.include_rotations:
            rotation_angles = [90, 180, 270]
            for angle in rotation_angles:
                rotation_dir = os.path.join(self.images_dir, f'Rotation_{angle}')
                self.prepare_rotated_images(rotation_dir, angle)
                self.load_images_from_dir(rotation_dir, pattern, rotation_degree=angle)

        print(f'\nTotal de imagens carregadas: {len(self.images)}')
        print(f'Pacientes únicos encontrados: {sorted(set(self.patients))}')
        print(f'Número de imagens por paciente: {dict(self.images_per_patient)}')

    def load_images_from_dir(self, directory, pattern, rotation_degree):
        for filename in sorted(os.listdir(directory)):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith('.png'):
                match = re.match(pattern, filename)
                if match:
                    patient_num = int(match.group(1))
                    self.patients.append(patient_num)
                    self.images.append(file_path)
                    self.rotations.append(rotation_degree)
                    self.images_per_patient[patient_num] += 1
                    self.labels.append(0 if patient_num <= 15 else 1)
                else:
                    print(f'Nome de arquivo inesperado: {filename}')

    def prepare_rotated_images(self, rotation_dir, angle):
        if not os.path.exists(rotation_dir):
            os.makedirs(rotation_dir)
            print(f'Criando diretório de rotação: {rotation_dir}')
        if not any(filename.endswith('.png') for filename in os.listdir(rotation_dir)):
            print(f'Rotacionando imagens em {angle} graus...')
            for filename in sorted(os.listdir(self.images_dir)):
                file_path = os.path.join(self.images_dir, filename)
                if os.path.isfile(file_path) and filename.endswith('.png') and 'Rotation_' not in filename:
                    img = Image.open(file_path)
                    rotated_img = img.rotate(angle, expand=True)
                    rotated_img.save(os.path.join(rotation_dir, filename))
        else:
            print(f'Imagens já rotacionadas em {angle} graus.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return image, self.labels[idx], self.patients[idx], self.rotations[idx]


class CrossValidationTraining:
    def __init__(self, dataset, trainings_array, num_epochs=10, batch_size=16,
                 learning_rate=0.001, experiment_name="Experiment", early_stopping_patience=5):
        self.dataset = dataset
        self.trainings_array = trainings_array
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join('Training Iterations', self.experiment_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logo = LeaveOneGroupOut()
        self.early_stopping_patience = early_stopping_patience
        self.total_test_images = 0
        self.final_results_dir = os.path.join(self.experiment_dir, 'FinalResults')
        
        self.metrics = {
            'accuracies': [],'precisions': [],'recalls': [],'f1_scores': [],'aurocs': [],'sensitivities': [],'specificities': [],'confusion_matrices': [],'train_accuracies_per_epoch': [],'val_accuracies_per_epoch': [],'train_losses_per_epoch': [],'val_losses_per_epoch': [],'training_times': [],
        }
        
        os.makedirs(self.final_results_dir, exist_ok=True)        
        # Criação do diretório para salvar as matrizes de confusão por época
        self.confusion_matrices_dir = os.path.join(self.final_results_dir, 'ConfusionMatricesPerEpoch')
        os.makedirs(self.confusion_matrices_dir, exist_ok=True)  # Criar o diretório principal

    def train(self):
        print(f'Usando dispositivo: {self.device}')
        print('\nIniciando o treinamento com validação cruzada...\n')
        total_folds = len(set(self.dataset.patients))
        for fold, (train_idx, test_idx) in enumerate(self.logo.split(np.arange(len(self.dataset)), groups=self.dataset.patients)):
            start_time = time.time()
            test_patient = self.dataset.patients[test_idx[0]]
            test_patient_indices = [idx for idx in test_idx if self.dataset.patients[idx] == test_patient]
            if not test_patient_indices:
                print(f'Aviso: Conjunto de teste vazio para o fold {fold + 1}. Pulando este fold.')
                continue
            print(f'\nFold {fold + 1}/{total_folds}:')
            print(f' - Número de imagens de teste: {len(test_patient_indices)}')
            train_indices = [idx for idx in train_idx if self.dataset.patients[idx] != test_patient]
            balanced_train_idx = self.balance_dataset(train_indices)
            train_subset = Subset(self.dataset, balanced_train_idx)
            test_subset = Subset(self.dataset, test_patient_indices)
            print(f' - Número de imagens de treino balanceadas: {len(train_subset)}')
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)
            model = self.initialize_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            # Criação do diretório específico para o fold atual
            fold_confusion_dir = os.path.join(self.confusion_matrices_dir, f'fold_{fold + 1}')
            os.makedirs(fold_confusion_dir, exist_ok=True)
            self.train_model(
                model, criterion, optimizer, train_loader, test_loader,
                fold, total_folds, test_patient, start_time, fold_confusion_dir  # Passar o diretório do fold
            )
        self.save_results()
        torch.save(model.state_dict(), os.path.join(self.final_results_dir, 'final_model.pth'))
        print(f'Modelo final salvo em {os.path.join(self.final_results_dir, "final_model.pth")}')

    def balance_dataset(self, train_indices):
        X = np.array(train_indices).reshape(-1, 1)
        y = np.array([self.dataset.labels[idx] for idx in train_indices])
        rus = RandomUnderSampler()
        X_resampled, _ = rus.fit_resample(X, y)
        return X_resampled.flatten().tolist()

    def initialize_model(self):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        return model.to(self.device)

    def train_model(self, model, criterion, optimizer, train_loader, test_loader,
                    fold, total_folds, test_patient, start_time, fold_confusion_dir):
        progress_bar = tqdm(total=self.num_epochs * len(train_loader), desc=f'Treinando Fold {fold + 1}/{total_folds}')
        best_val_loss = np.inf
        epochs_no_improve = 0
        metrics_per_epoch = self.initialize_epoch_metrics()
        best_model_wts = None
        for epoch in range(self.num_epochs):
            self.run_epoch(
                model, criterion, optimizer, train_loader, progress_bar, epoch,
                metrics_per_epoch, mode='train'
            )
            self.run_epoch(
                model, criterion, optimizer, test_loader, progress_bar, epoch,
                metrics_per_epoch, mode='val'
            )
            # Computar e salvar a matriz de confusão após a época de validação
            cm = confusion_matrix(metrics_per_epoch['all_labels'], metrics_per_epoch['all_preds'], labels=[0,1])
            # Plotar a matriz de confusão
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Fold {fold + 1} - Época {epoch + 1} - Matriz de Confusão')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
            plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            # Salvar a matriz de confusão como imagem
            confusion_matrix_path = os.path.join(fold_confusion_dir, f'epoch_{epoch + 1}.png')
            plt.savefig(confusion_matrix_path)
            plt.close()
            #print(f'Matriz de confusão da Época {epoch + 1} salva em {confusion_matrix_path}')
            # Fim da adição

            val_epoch_loss = metrics_per_epoch['val_loss'][-1]
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
                best_model_wts = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f'\nEarly stopping at epoch {epoch + 1}')
                    if best_model_wts is not None:
                        model.load_state_dict(best_model_wts)
                    break
        progress_bar.close()
        self.metrics['training_times'].append(time.time() - start_time)
        all_preds = metrics_per_epoch['all_preds']
        all_labels = metrics_per_epoch['all_labels']
        all_probs = metrics_per_epoch['all_probs']
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        self.metrics['precisions'].append(precision)
        self.metrics['recalls'].append(recall)
        self.metrics['f1_scores'].append(f1)
        self.metrics['aurocs'].append(auc)
        self.metrics['confusion_matrices'].append(cm)
        self.metrics['train_accuracies_per_epoch'].append(metrics_per_epoch['train_acc'])
        self.metrics['val_accuracies_per_epoch'].append(metrics_per_epoch['val_acc'])
        self.metrics['train_losses_per_epoch'].append(metrics_per_epoch['train_loss'])
        self.metrics['val_losses_per_epoch'].append(metrics_per_epoch['val_loss'])
        self.update_overall_metrics(metrics_per_epoch, test_patient, fold)

    def initialize_epoch_metrics(self):
        return {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'all_preds': [],
            'all_labels': [],
            'all_probs': [],
        }

    def run_epoch(self, model, criterion, optimizer, loader, progress_bar, epoch, metrics, mode='train'):
        is_train = mode == 'train'
        model.train() if is_train else model.eval()
        running_corrects = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        for inputs, labels, _, _ in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if is_train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(is_train):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    optimizer.step()
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * inputs.size(0)
            progress_bar.update(1)
            progress_bar.set_postfix({'Loss': loss.item(), 'Época': epoch + 1})
            if not is_train:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)
        metrics[f'{mode}_acc'].append(epoch_acc.item())
        metrics[f'{mode}_loss'].append(epoch_loss)
        if not is_train:
            metrics['all_preds'] = all_preds
            metrics['all_labels'] = all_labels
            metrics['all_probs'] = all_probs

    def update_overall_metrics(self, metrics_per_epoch, test_patient, fold):
        val_acc = metrics_per_epoch['val_acc'][-1] if metrics_per_epoch['val_acc'] else 0
        self.metrics['accuracies'].append(val_acc)
        cm = self.metrics['confusion_matrices'][-1]
        TP, FN, TN, FP = cm[1, 1], cm[1, 0], cm[0, 0], cm[0, 1]
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        self.metrics['sensitivities'].append(sensitivity)
        self.metrics['specificities'].append(specificity)
        print(f"Paciente {test_patient} - Acurácia no Teste: {val_acc:.4f}")
        print(f"Tempo de treinamento para o Fold {fold + 1}: {self.metrics['training_times'][-1]:.2f} segundos")

    def save_results(self):
        self.save_aggregated_metrics()

        # Defina as chaves que contêm métricas escalares por fold
        scalar_keys = [
            'accuracies', 'precisions', 'recalls', 'f1_scores',
            'aurocs', 'sensitivities', 'specificities', 'training_times'
        ]

        # Calcule a média e o desvio padrão apenas para as métricas escalares
        mean_metrics = {
            key: (np.mean(values), np.std(values)) 
            for key, values in self.metrics.items()
            if key in scalar_keys and isinstance(values, list) and values
        }

        self.trainings_array.append(mean_metrics['accuracies'][0])

        if self.metrics['confusion_matrices']:
            total_cm = np.sum(self.metrics['confusion_matrices'], axis=0)
            self.save_aggregated_confusion_matrix(total_cm)

        self.save_overall_metrics(mean_metrics)

        print(f'\nProcesso concluído. Resultados salvos na pasta {self.experiment_name}.')
        print('\nMétricas após Validação Cruzada:')
        for metric in ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities']:
            mean, std = mean_metrics[metric]
            print(f'{metric.capitalize()[:-1]} Média: {mean:.4f} ± {std:.4f}')


    def save_aggregated_metrics(self):
        metrics = ['Accuracies', 'Losses']
        for metric_name in metrics:
            metric_key = metric_name.lower()
            train_values_list = self.metrics[f'train_{metric_key}_per_epoch']
            val_values_list = self.metrics[f'val_{metric_key}_per_epoch']
            if not train_values_list or not val_values_list:
                continue
            max_epochs = max(len(values) for values in train_values_list)
            mean_train_values = np.mean([np.pad(values, (0, max_epochs - len(values)), 'constant') for values in train_values_list], axis=0)
            mean_val_values = np.mean([np.pad(values, (0, max_epochs - len(values)), 'constant') for values in val_values_list], axis=0)
            self.save_mean_metric_graph(metric_name, mean_train_values, mean_val_values)

    def save_mean_metric_graph(self, metric_name, mean_train_values, mean_val_values):
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

    def save_aggregated_confusion_matrix(self, total_cm):
        plt.figure()
        plt.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão Agregada')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        thresh = total_cm.max() / 2.
        for i, j in np.ndindex(total_cm.shape):
            plt.text(j, i, format(total_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if total_cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(self.final_results_dir, 'aggregated_confusion_matrix.png'))
        plt.close()
        np.savetxt(os.path.join(self.final_results_dir, 'aggregated_confusion_matrix.csv'), total_cm, delimiter=',', fmt='%d')

    def save_overall_metrics(self, mean_metrics):
        with open(os.path.join(self.final_results_dir, 'metrics.csv'), 'w', newline='') as csvfile:
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
            writer.writerow(['Média'] + [mean_metrics[key][0] for key in ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities', 'training_times']])
            writer.writerow(['Desvio Padrão'] + [mean_metrics[key][1] for key in ['accuracies', 'precisions', 'recalls', 'f1_scores', 'aurocs', 'sensitivities', 'specificities', 'training_times']])

if __name__ == "__main__":
    seed = "Alexei"
    converted_seed = sum(ord(char) for char in seed)
    print(f"Converted integer: {converted_seed}")
    set_seed(converted_seed)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ROIDataset('ROISavedFiles', transform=data_transforms, include_rotations=True)
    hyperparameter_sets = [
        {'num_epochs': 2, 'batch_size': 64, 'learning_rate': 0.002},
        # {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.002},
        # {'num_epochs': 10, 'batch_size': 64, 'learning_rate': 0.004}
    ]
    trainings_array = []
    for params in hyperparameter_sets:
        start_time = time.time()
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        experiment_name = f"{num_epochs}_Epochs_{batch_size}_BatchSize_{learning_rate}_LR"
        print(f"\nIniciando treinamento com hiperparâmetros: {experiment_name}")
        trainer = CrossValidationTraining(
            dataset,
            trainings_array,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            experiment_name=experiment_name,
            early_stopping_patience=5
        )
        trainer.train()
        elapsed_time = time.time() - start_time
        print(f'\nTempo total de treinamento: {elapsed_time:.2f} segundos')
    print(trainings_array)