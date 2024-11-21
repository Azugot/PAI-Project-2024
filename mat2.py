import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score
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

class ROIDataset(Dataset):
    def __init__(self, images_dir, transform=None, include_rotations=False):
        self.images_dir = images_dir
        self.transform = transform
        self.include_rotations = include_rotations
        self.images = []
        self.labels = []
        self.patients = []
        self.rotations = []  # To keep track of rotation degrees

        print(f'Preparando o dataset: {images_dir}')
        pattern = r'ROI_(\d+)_(\d+)\.png'

        # Include original images
        self._load_images_from_dir(self.images_dir, pattern, rotation_degree=0)

        if self.include_rotations:
            # Check for rotated images and include them
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

                    if patient_num <= 16:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                else:
                    print(f'Nome de arquivo inesperado: {filename}')

    def _check_rotated_images_exist(self, rotation_dir):
        # Check if the directory contains any images
        return any(filename.endswith('.png') for filename in os.listdir(rotation_dir))

    def _rotate_and_save_images(self, angle, rotation_dir):
        for filename in sorted(os.listdir(self.images_dir)):
            file_path = os.path.join(self.images_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.png'):
                # Skip rotation directories
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
    def __init__(self, dataset, test_dataset, num_epochs=10, batch_size=16, learning_rate=0.001, experiment_name="Experiment", early_stopping_patience=5):
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
        self.confusion_matrices = []
        self.train_accuracies_per_epoch = []
        self.val_accuracies_per_epoch = []
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []
        self.early_stopping_patience = early_stopping_patience

        # Create directories specific to this experiment
        self.train_graphs_dir = os.path.join(self.experimentNameDirectory, 'TrainGraphs')
        self.final_results_dir = os.path.join(self.experimentNameDirectory, 'FinalResults')
        os.makedirs(self.train_graphs_dir, exist_ok=True)
        os.makedirs(self.final_results_dir, exist_ok=True)

    def train(self):
        print(f'Usando dispositivo: {self.device}')
        print('\nIniciando o treinamento com validação cruzada...\n')

        for fold, (train_idx, test_idx) in enumerate(self.logo.split(np.arange(len(self.dataset)), groups=self.dataset.patients)):
            # Exclude rotated images from the test set
            test_indices = [idx for idx in test_idx if self.dataset.rotations[idx] == 0]
            if not test_indices:
                print(f'Aviso: Conjunto de teste vazio para o fold {fold + 1}. Pulando este fold.')
                continue

            current_patient = self.dataset.patients[test_indices[0]]
            print(f'Fold {fold + 1}/{len(set(self.dataset.patients))} - Paciente de teste: {current_patient}')
            print(f'Tamanho do conjunto de treino: {len(train_idx)}, Tamanho do conjunto de teste: {len(test_indices)}')

            train_subset = Subset(self.dataset, train_idx)
            test_subset = Subset(self.dataset, test_indices)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

            weights = EfficientNet_B0_Weights.DEFAULT
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

            total_steps = self.num_epochs * len(train_loader)
            progress_bar = tqdm(total=total_steps, desc=f'Treinando Fold {fold + 1}/{len(set(self.dataset.patients))}')

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

                epoch_loss = running_loss / len(train_subset)
                epoch_acc = running_corrects.double() / len(train_subset)
                train_acc_epoch.append(epoch_acc.item())
                train_loss_epoch.append(epoch_loss)

                model.eval()
                val_running_corrects = 0
                val_running_loss = 0.0
                all_preds = []
                all_labels = []

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

                # Early stopping check
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    epochs_no_improve = 0
                    # Save the best model for this fold
                    best_model_wts = model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.early_stopping_patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        # Load the best model weights
                        model.load_state_dict(best_model_wts)
                        break

            progress_bar.close()

            # Saving and displaying the learning curve graph for this fold
            plt.figure()
            plt.plot(range(1, len(train_acc_epoch) + 1), train_acc_epoch, label='Treino')
            plt.plot(range(1, len(val_acc_epoch) + 1), val_acc_epoch, label='Validação')
            plt.xlabel('Época')
            plt.ylabel('Acurácia')
            plt.title(f'Gráfico de Aprendizado - Fold {fold + 1}')
            plt.legend()
            plt.savefig(os.path.join(self.train_graphs_dir, f'learning_curve_fold_{fold + 1}.png'))
            plt.close()

            # Also plot loss curves
            plt.figure()
            plt.plot(range(1, len(train_loss_epoch) + 1), train_loss_epoch, label='Treino')
            plt.plot(range(1, len(val_loss_epoch) + 1), val_loss_epoch, label='Validação')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.title(f'Curva de Loss - Fold {fold + 1}')
            plt.legend()
            plt.savefig(os.path.join(self.train_graphs_dir, f'loss_curve_fold_{fold + 1}.png'))
            plt.close()

            if val_acc_epoch:
                final_acc = val_acc_epoch[-1]
                self.accuracies.append(final_acc)
            else:
                print(f'Aviso: Acurácia de validação não disponível para o paciente {current_patient}.')

            if all_labels and all_preds:
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                self.confusion_matrices.append(cm)

                plt.figure()
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Matriz de Confusão - Paciente {current_patient}')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
                plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
                plt.ylabel('Verdadeiro')
                plt.xlabel('Predito')
                plt.tight_layout()
                plt.savefig(os.path.join(self.train_graphs_dir, f'confusion_matrix_patient_{current_patient}.png'))
                plt.close()
            else:
                print(f'Aviso: Não foi possível gerar a matriz de confusão para o paciente {current_patient}.')

            self.train_accuracies_per_epoch.append(train_acc_epoch)
            self.val_accuracies_per_epoch.append(val_acc_epoch)
            self.train_losses_per_epoch.append(train_loss_epoch)
            self.val_losses_per_epoch.append(val_loss_epoch)
            print(f"Paciente {current_patient} - Acurácia no Teste: {final_acc:.4f}")

        self.save_results()
        # Train the final model and evaluate on the test set
        self.train_final_model()

    def save_results(self):
        if self.train_accuracies_per_epoch and self.val_accuracies_per_epoch:
            max_epochs = max(len(acc_list) for acc_list in self.train_accuracies_per_epoch)
            mean_train_acc = np.zeros(max_epochs)
            mean_val_acc = np.zeros(max_epochs)
            mean_train_loss = np.zeros(max_epochs)
            mean_val_loss = np.zeros(max_epochs)

            for acc_list in self.train_accuracies_per_epoch:
                mean_train_acc[:len(acc_list)] += acc_list
            mean_train_acc /= len(self.train_accuracies_per_epoch)

            for acc_list in self.val_accuracies_per_epoch:
                mean_val_acc[:len(acc_list)] += acc_list
            mean_val_acc /= len(self.val_accuracies_per_epoch)

            for loss_list in self.train_losses_per_epoch:
                mean_train_loss[:len(loss_list)] += loss_list
            mean_train_loss /= len(self.train_losses_per_epoch)

            for loss_list in self.val_losses_per_epoch:
                mean_val_loss[:len(loss_list)] += loss_list
            mean_val_loss /= len(self.val_losses_per_epoch)

            plt.figure()
            plt.plot(range(1, len(mean_train_acc) + 1), mean_train_acc, label='Treino')
            plt.plot(range(1, len(mean_val_acc) + 1), mean_val_acc, label='Validação')
            plt.xlabel('Época')
            plt.ylabel('Acurácia')
            plt.title('Gráfico de Aprendizado Médio')
            plt.legend()
            plt.savefig(os.path.join(self.final_results_dir, 'mean_learning_curve.png'))
            plt.close()

            plt.figure()
            plt.plot(range(1, len(mean_train_loss) + 1), mean_train_loss, label='Treino')
            plt.plot(range(1, len(mean_val_loss) + 1), mean_val_loss, label='Validação')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.title('Curva de Loss Média')
            plt.legend()
            plt.savefig(os.path.join(self.final_results_dir, 'mean_loss_curve.png'))
            plt.close()
        else:
            print('Aviso: Não foi possível gerar o gráfico de aprendizado médio.')

        if self.confusion_matrices:
            total_cm = np.sum(self.confusion_matrices, axis=0)

            plt.figure()
            plt.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Matriz de Confusão Agregada')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
            plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            plt.tight_layout()
            plt.savefig(os.path.join(self.final_results_dir, 'aggregated_confusion_matrix.png'))
            plt.close()
        else:
            print('Aviso: Não foi possível gerar a matriz de confusão agregada.')

        # Saving results to .pkl files
        with open(os.path.join(self.train_graphs_dir, 'accuracies.pkl'), 'wb') as f:
            pickle.dump(self.accuracies, f)

        with open(os.path.join(self.train_graphs_dir, 'confusion_matrices.pkl'), 'wb') as f:
            pickle.dump(self.confusion_matrices, f)

        with open(os.path.join(self.train_graphs_dir, 'train_accuracies_per_epoch.pkl'), 'wb') as f:
            pickle.dump(self.train_accuracies_per_epoch, f)

        with open(os.path.join(self.train_graphs_dir, 'val_accuracies_per_epoch.pkl'), 'wb') as f:
            pickle.dump(self.val_accuracies_per_epoch, f)

        # Saving results to CSV files
        if self.accuracies:
            with open(os.path.join(self.final_results_dir, 'accuracies.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Fold', 'Acurácia'])
                for idx, acc in enumerate(self.accuracies):
                    writer.writerow([idx + 1, acc])

        if self.train_accuracies_per_epoch:
            with open(os.path.join(self.final_results_dir, 'train_accuracies_per_epoch.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                max_epochs = max(len(acc_list) for acc_list in self.train_accuracies_per_epoch)
                header = ['Fold'] + [f'Época {i+1}' for i in range(max_epochs)]
                writer.writerow(header)
                for idx, acc_list in enumerate(self.train_accuracies_per_epoch):
                    row = [idx + 1] + acc_list + [''] * (max_epochs - len(acc_list))
                    writer.writerow(row)

        if self.val_accuracies_per_epoch:
            with open(os.path.join(self.final_results_dir, 'val_accuracies_per_epoch.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                max_epochs = max(len(acc_list) for acc_list in self.val_accuracies_per_epoch)
                header = ['Fold'] + [f'Época {i+1}' for i in range(max_epochs)]
                writer.writerow(header)
                for idx, acc_list in enumerate(self.val_accuracies_per_epoch):
                    row = [idx + 1] + acc_list + [''] * (max_epochs - len(acc_list))
                    writer.writerow(row)

        print(f'\nProcesso concluído. Resultados salvos na pasta {self.experiment_name}.')

        if self.accuracies:
            mean_accuracy = np.mean(self.accuracies)
            print(f'\nAcurácia Média após Validação Cruzada: {mean_accuracy:.4f}')
        else:
            print('\nNão foi possível calcular a acurácia média.')

    def train_final_model(self):
        print('\nTreinando o modelo final no conjunto completo de treinamento...')
        # Exclude rotated images from the test dataset
        test_indices = [idx for idx in range(len(self.test_dataset)) if self.test_dataset.rotations[idx] == 0]
        test_subset = Subset(self.test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        # Training dataset includes rotations
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        weights = EfficientNet_B0_Weights.DEFAULT
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

            # Validation on test set for early stopping
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels, _, _ in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(test_subset)

            # Early stopping check
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
                # Save the best model weights
                best_model_wts = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    # Load the best model weights
                    model.load_state_dict(best_model_wts)
                    break

        progress_bar.close()

        print('\nAvaliando o modelo final no conjunto de teste...')
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _, _ in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        print(f'Acurácia no Conjunto de Teste: {test_acc:.4f}')

        # Generating the confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão - Conjunto de Teste')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Saudável', 'Não Saudável'], rotation=45)
        plt.yticks(tick_marks, ['Saudável', 'Não Saudável'])
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig(os.path.join(self.final_results_dir, 'confusion_matrix_test_set.png'))
        plt.close()

        # Save the final model
        torch.save(model.state_dict(), os.path.join(self.final_results_dir, 'final_model.pth'))
        print(f'Modelo final salvo em {os.path.join(self.final_results_dir, "final_model.pth")}')

if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset de treinamento (usado na validação cruzada), incluindo rotações
    dataset = ROIDataset('ROISavedFiles', transform=data_transforms, include_rotations=True)

    # Dataset de teste separado, sem rotações
    test_dataset = ROIDataset('TestSet', transform=data_transforms, include_rotations=False)

    # List of hyperparameter sets
    hyperparameter_sets = [
        {'num_epochs': 1, 'batch_size': 50, 'learning_rate': 0.001},
    ]

    for params in hyperparameter_sets:
        start_time = time.time()

        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']

        # Create a unique experiment name based on hyperparameters
        experiment_name = f"{num_epochs}_Epochs_{batch_size}_BatchSize_{learning_rate}_LR"
        print(f"\nIniciando treinamento com hiperparâmetros: {experiment_name}")

        trainer = CrossValidationTraining(
            dataset,
            test_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            experiment_name=experiment_name,
            early_stopping_patience=5  # You can adjust the patience as needed
        )
        trainer.train()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'\nTempo total de treinamento: {elapsed_time:.2f} segundos')