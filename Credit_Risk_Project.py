import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder


modulo_1 = pd.read_stata('dta_files/01_MÓDULO.dta')
modulo_2 = pd.read_stata('dta_files/02_MÓDULO.dta')
modulo_3 = pd.read_stata('dta_files/03_MÓDULO.dta')
modulo_4 = pd.read_stata('dta_files/04_MÓDULO.dta')
modulo_5 = pd.read_stata('dta_files/05_MÓDULO.dta')


# Selección de columnas para cada módulo con solo las variables necesarias
modulo_1 = modulo_1[['IRUC','DEPARTAMENTO','PROVINCIA','DISTRITO','C4','C20','C1RESULTFINAL','C11','C12']]
modulo_2 = modulo_2[['IRUC', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'C4', 'C20', 'C1RESULTFINAL', 'M1P3']]
modulo_3 = modulo_3[['IRUC', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'C4', 'C20', 'C1RESULTFINAL']]
modulo_4 = modulo_4[['IRUC', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'C4', 'C20', 'C1RESULTFINAL','M5P3_1','M6P1_1']]
modulo_5 = modulo_5[['IRUC', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'C4', 'C20', 'C1RESULTFINAL','M8P53_20','M8P30','M8P53_5','M9P4_7','M7P8_5', 'M7P10']]



credito_df = modulo_1.merge(modulo_2, how='left',on=['IRUC','DEPARTAMENTO','PROVINCIA','DISTRITO','C4','C20','C1RESULTFINAL'])
credito_df = credito_df.merge(modulo_3,how='left',on=['IRUC','DEPARTAMENTO','PROVINCIA','DISTRITO','C4','C20','C1RESULTFINAL'])
credito_df = credito_df.merge(modulo_4,how='left',on=['IRUC','DEPARTAMENTO','PROVINCIA','DISTRITO','C4','C20','C1RESULTFINAL'])
credito_df = credito_df.merge(modulo_5,how='left',on=['IRUC','DEPARTAMENTO','PROVINCIA','DISTRITO','C4','C20','C1RESULTFINAL'])


# Quiero ver las cantidades exactas de los que terminaron la encuesta para quedarme con solo esos datos y evitar sesgos
conteo_valores = credito_df['C1RESULTFINAL'].value_counts()

# Muestra las cantidades exactas
for valor, cantidad in conteo_valores.items():
    print(f'{valor}: {cantidad}')



# Sobrescribir el DataFrame original con solo las filas donde 'C1RESULTFINAL' es 'Completo'
credito_df = credito_df.loc[credito_df['C1RESULTFINAL'] == 'Completo']



credito_df.rename(columns={'C12': 'nivel_educativo'}, inplace=True)
credito_df.rename(columns={'C11': 'edad'}, inplace=True)

credito_df.rename(columns={'M1P3': 'outcome'}, inplace=True)

credito_df.rename(columns={'M5P3_1': 'sofware_contrable'}, inplace=True)

credito_df.rename(columns={'M6P1_1': 'robo'}, inplace=True)

credito_df.rename(columns={'M7P8_5': 'seguro_fraude'}, inplace=True)
credito_df.rename(columns={'M7P10': 'liquidez_problema'}, inplace=True)
credito_df.rename(columns={'M9P4_7': 'ingreso'}, inplace=True)
credito_df.rename(columns={'M8P53_5': 'poca_demanda'}, inplace=True)
credito_df.rename(columns={'M8P30': 'atendio_auditorias'}, inplace=True)
credito_df.rename(columns={'M8P53_20': 'informalidad'}, inplace=True)



map_audit={'Si':1,'No':0}
credito_df['atendio_auditorias']=credito_df['atendio_auditorias'].map(map_audit)


credito_df=credito_df.drop('atendio_auditorias',axis=1)



print(credito_df['nivel_educativo'].value_counts(dropna=False))
print(credito_df['nivel_educativo'].isnull().sum())

map_niv={   'Superior universitaria completa':3,
            'Post grado':3,
            'Secundaria completa':2,
            'Superior no universitaria completa':3,
            'Superior universitaria incompleta':2,
            'Superior no universitaria incompleta':2,
            'Secundaria incompleta':1,
            'Primaria Completa':1,
            'Primaria incompleta':0,
            'Sin nivel':0,
            'Inicial':0
}

credito_df['nivel_educativo']=credito_df['nivel_educativo'].map(map_niv)


print(credito_df['nivel_educativo'].value_counts(dropna=False))


print(credito_df['edad'].value_counts(dropna=False))


print(credito_df['edad'].isnull().sum())


mediana_ingreso = credito_df['ingreso'].median()

#Reemplazar los valores NaN por la mediana
credito_df['ingreso'].fillna(mediana_ingreso, inplace=True)

# # Verificar los cambios
print(credito_df['ingreso'].isnull().sum())


credito_df = credito_df.dropna(subset=['ingreso'])
print(credito_df['ingreso'].value_counts(dropna=False))



print(credito_df['C1RESULTFINAL'].value_counts(dropna=False))
print(credito_df['C1RESULTFINAL'].isnull().sum())


# Obtener la cantidad total de registros en el DataFrame
total_registros = credito_df.shape[0]

# Imprimir la cantidad total de registros
print(f"La cantidad total de registros es: {total_registros}")


# Imprimir la cantidad total de registros
print(f"La cantidad total de registros es: {total_registros}")


print(credito_df['outcome'].value_counts(dropna=False))


credito_df = credito_df.dropna(subset=['outcome'])


print(credito_df['outcome'].value_counts(dropna=False))


# Crear una función de mapeo para asignar 1 a "si" y 0 a "no"
def map_credito_inicial(valor):
    return 1 if valor == "Si" else 0

# Aplicar la función de mapeo para crear la nueva columna "outcome"
credito_df['outcome'] = credito_df['outcome'].map(map_credito_inicial)


print(credito_df['outcome'].value_counts(dropna=False))


# Eliminar las columnas que de 'IRUC', 'DEPARTAMENTO', 'PROVINCIA', y 'DISTRITO' que solo nos interesaron para el merge
columnas_a_eliminar = ['IRUC', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']

# Eliminar las columnas
credito_df = credito_df.drop(columnas_a_eliminar, axis=1)


# Eliminar la columna "C1RESULTFINAL" que ya nos delimito la muestra
credito_df = credito_df.drop('C1RESULTFINAL', axis=1)


# # Eliminar las filas con valores nulos en la columna 'tipo_acredor_numerico'
credito_df = credito_df.dropna(subset=['sofware_contrable'])

# # Verificar que las filas con valores nulos han sido eliminadas
print(credito_df['sofware_contrable'].isnull().sum())


# Mapear 'Si' a 1 y 'No' a 0 para problemas de liquidez
credito_df['liquidez_problema_num'] = credito_df['liquidez_problema'].map({'Si': 1, 'No': 0})

# Borrar la antigua variable 
credito_df.drop(columns=['liquidez_problema'], inplace=True)


credito_df.drop(columns=['C4'], inplace=True)

credito_df.drop(columns=['C20'], inplace=True)


#visualización
credito_df.isnull().sum()


# Definir las columnas categóricas
categorical_columns = ['outcome', 'liquidez_problema_num','nivel_educativo']

# Cambiar el tipo de datos de las columnas categóricas a float64
for column in categorical_columns:
    credito_df[column] = credito_df[column].astype('float64')

# Verificar los tipos de datos
print(credito_df.dtypes)



# Uso de microeconometría para asegurar la significancia de ciertas variables
# en este caso usamos Regresión Logit debido a que la variable outcome es 0 o 1

#pip install statsmodels

import statsmodels.api as sm

# Suponiendo que tu DataFrame se llama credito_df
variables_independientes = ['sofware_contrable','edad','ingreso','poca_demanda','informalidad',
                            'seguro_fraude','nivel_educativo','robo',
                            'liquidez_problema_num']

X = credito_df[variables_independientes]
y = credito_df['outcome']

# Agregar una constante para el intercepto del modelo
X = sm.add_constant(X)

# Definir y ajustar el modelo Logit
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Imprimir el resumen del modelo
print(result.summary())


# Aplicación de Machine Learning para predicción de otorgamiento de crédito
# En este caso usammos KNN, Random Forest y XGBoost como métodos
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


X = credito_df.drop('outcome', axis=1)
y = credito_df['outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=250)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir modelos y parámetros para ajuste
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=250),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=250)
}

params = {
    'K-Nearest Neighbors': {'n_neighbors': [5, 10, 20, 50, 100]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Entrenamiento y ajuste de modelos
plt.figure(figsize=(7, 5))
for model_name, model in models.items():
    grid_search = GridSearchCV(model, params[model_name], cv=5, scoring='roc_auc')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Curva ROC de referencia 
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Configuración del gráfico
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Different Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
