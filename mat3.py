import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import warnings


class XGBoostCrossValidation:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.X = None
        self.y = None
        self.groups = None
        self.load_data()

    def load_data(self):
        print("Carregando dados...")
        # Carregar o CSV
        self.data = pd.read_csv(self.csv_file, on_bad_lines='skip', encoding='utf-8')
        self.data = self.data.drop_duplicates()

        # Adicionar coluna para rastrear índices originais
        self.data['OriginalIndex'] = self.data.index

        # Mapear classes para valores binários
        self.data['Classificação'] = self.data['Classificação'].map({'Saudável': 0, 'Esteatose': 1})

        # Verificar valores NaN
        if self.data['Classificação'].isnull().any():
            print("Linhas com valores NaN na coluna 'Classificação':")
            print(self.data[self.data['Classificação'].isnull()])
            self.data = self.data.dropna(subset=['Classificação'])

        print("Distribuição das classes:")
        print(self.data['Classificação'].value_counts())

        # Extrair o número do paciente e características
        self.data['Paciente'] = self.data['Arquivo'].apply(lambda x: int(x.split('_')[1]))
        self.features = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 
                        'Entropy', 'Coarseness', 'Periodicity', 'Roughness']
        self.X = self.data[self.features].values
        self.y = self.data['Classificação'].values
        self.groups = self.data['Paciente'].values

        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} características.")


    def cross_validate(self):
        print("Iniciando validação cruzada avançada com todos os pares de pacientes...")
        unique_patients = np.unique(self.groups)
        accuracies = []
        recalls = []
        specificities = []
        confusion_matrices = []

        # Arquivo para salvar os resultados
        results_file = "cross_validation_results.csv"
        with open(results_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Fold", "Patient", "Accuracy", "Recall", "Specificity"])

            # Loop para cada paciente como conjunto de teste
            for fold, test_patient in enumerate(unique_patients):
                # Separar todas as linhas (imagens) do paciente atual para teste
                test_idx = np.where(self.groups == test_patient)[0]
                X_test, y_test = self.X[test_idx], self.y[test_idx]
                test_original_indices = self.data['OriginalIndex'][test_idx]  # Índices originais

                # Separar todas as linhas (imagens) dos outros pacientes para treino
                train_idx = np.where(self.groups != test_patient)[0]
                X_train, y_train = self.X[train_idx], self.y[train_idx]
                train_original_indices = self.data['OriginalIndex'][train_idx]  # Índices originais

                print(f"Fold {fold + 1}: Paciente de Teste={test_patient}")
                print(f"Número de Amostras no Treino: {len(train_idx)}, Teste: {len(test_idx)}")

                # Verificar a presença de ambas as classes no conjunto de treinamento
                if len(np.unique(y_train)) < 2:
                    print(f"Aviso: Apenas uma classe presente no treinamento (Paciente {test_patient}). Pulando.")
                    continue

                # Configurar o modelo XGBoost
                scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                model = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=200,
                    random_state=42
                )

                # Treinar o modelo
                model.fit(X_train, y_train)

                # Fazer previsões
                y_pred = model.predict(X_test)

                # Calcular a matriz de confusão com todas as classes
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                # Calcular métricas
                acc = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, zero_division=0)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Salvar métricas para este paciente
                accuracies.append(acc)
                recalls.append(recall)
                specificities.append(specificity)
                confusion_matrices.append(cm)

                # Salvar os resultados no CSV
                writer.writerow([fold + 1, test_patient, acc, recall, specificity])

                # Exibir resultados para o paciente em teste
                print(f"Paciente de Teste={test_patient}, "
                    f"Acurácia={acc:.4f}, Sensibilidade={recall:.4f}, Especificidade={specificity:.4f}")

        # Calcular métricas médias gerais
        avg_accuracy = np.mean(accuracies)
        avg_recall = np.mean(recalls)
        avg_specificity = np.mean(specificities)
        print("Resultados Médios após Validação Cruzada:")
        print(f"Acurácia Média: {avg_accuracy:.4f}")
        print(f"Sensibilidade Média: {avg_recall:.4f}")
        print(f"Especificidade Média: {avg_specificity:.4f}")

        # Matriz de Confusão Agregada
        aggregated_cm = np.sum(confusion_matrices, axis=0)
        self.plot_confusion_matrix(aggregated_cm, ["Saudável", "Esteatose"], "Matriz de Confusão Agregada")
        print(f"Matriz de Confusão Agregada:{aggregated_cm}")


    @staticmethod
    def plot_confusion_matrix(cm, labels, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()


if __name__ == "__main__":
    # Caminho para o arquivo CSV
    csv_file = './ROISavedFiles/glcm_sfm_data.csv'
    
    # Inicializar e executar
    xgb_cv = XGBoostCrossValidation(csv_file)
    xgb_cv.cross_validate()
