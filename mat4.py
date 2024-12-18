import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


with open('./ROISavedFiles/glcm_sfm_data.csv', 'r', encoding='utf-8') as file:
    total_lines = sum(1 for line in file)
print(f"Número de linhas no arquivo CSV: {total_lines}")


data = pd.read_csv('./ROISavedFiles/glcm_sfm_data.csv', on_bad_lines='warn', encoding='utf-8')
print(f"Número de linhas lidas inicialmente: {data.shape[0]}")

data = data.drop_duplicates()
data['Paciente'] = data['Arquivo'].str.extract(r'(ROI_\d+)_')
data.drop(columns=['Periodicity', 'Roughness'], inplace=True)

def add_interaction_features(df, numeric_cols):
    interactions = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            interaction_name = f'{col1}_{col2}_interaction'
            if len(interactions) < 10: 
                df[interaction_name] = df[col1] * df[col2]
                interactions.append(interaction_name)
    return df

numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data = add_interaction_features(data, numeric_columns)

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df[numeric_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

data = remove_highly_correlated_features(data)

scaler = StandardScaler()
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

variances = data[numeric_columns].var()
zero_variance_cols = variances[variances == 0].index
data.drop(columns=zero_variance_cols, inplace=True)

encoder = LabelEncoder()
data['Classificação'] = encoder.fit_transform(data['Classificação'])
data['Paciente'] = encoder.fit_transform(data['Paciente'])

X = data.drop(columns=['Classificação', 'Arquivo'])
y = data['Classificação']

splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=X['Paciente']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Verificar se há sobreposição entre pacientes nos conjuntos de treino e teste
overlap = set(X_train['Paciente']).intersection(set(X_test['Paciente']))
if overlap:
    raise ValueError(f"Pacientes presentes em ambos os conjuntos: {overlap}")
else:
    print("Treinamento isolado por paciente está correto.")


pipeline = Pipeline([
    ('feature_selection', SelectKBest(f_classif, k='all')),
    ('sampling', RandomUnderSampler(random_state=42)),
    ('classifier', XGBClassifier(eval_metric="logloss"))
])

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

group_kfold = GroupKFold(n_splits=5)
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=group_kfold,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

X_train_no_patient = X_train.drop(columns=['Paciente'])
X_test_no_patient = X_test.drop(columns=['Paciente'])

train_groups = X_train['Paciente']

random_search.fit(X_train_no_patient, y_train, groups=train_groups)

print("\nMelhores parâmetros encontrados:")
print(random_search.best_params_)

y_pred = random_search.predict(X_test_no_patient)
print("\nRelatório de Classificação - Teste Final:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Saudável', 'Esteatose'])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - Conjunto de Teste")
plt.show()

best_model = random_search.best_estimator_.named_steps['classifier']
selected_features = random_search.best_estimator_.named_steps['feature_selection']
feature_mask = selected_features.get_support()
selected_feature_names = X_train_no_patient.columns[feature_mask].tolist()

feature_importance = pd.DataFrame({
    'feature': selected_feature_names,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Features Mais Importantes')
plt.tight_layout()
plt.show()

best_params = pd.DataFrame([random_search.best_params_])
best_params.to_csv('best_parameters.csv', index=False)

feature_importance.to_csv('feature_importance.csv', index=False)