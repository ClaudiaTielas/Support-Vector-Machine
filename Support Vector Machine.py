# -*- coding: utf-8 -*-
"""SVM.ipynb

Diciembre 2023

Claudia Tielas Sáez

# Algoritmo Support Vector Machine

Las Máquinas de Vectores Soporte (SVM) son modelos son conocidos por su alta precisión en problemas complejos con un uso eficiente de recursos computacionales.

Las SVM buscan encontrar un hiperplano de separación con margen máximo entre dos clases. A diferencia de otros métodos lineales, seleccionan el hiperplano que logra la máxima distancia entre las instancias frontera. El parámetro crucial es el Coste (C), que equilibra el error de clasificación y la generalización del modelo.
"""

import pandas as pd
from tabulate import tabulate

# Creamos un diccionario con las fortalezas y debilidades
data = {
    'Aspecto': ['Capacidad de generalización', 'Manejo de características no lineales', 'Margen máximo',
                'Eficacia en espacios de alta dimensión', 'Manejo de grandes conjuntos de datos',
                'Susceptibilidad al sobreajuste', 'Versatilidad'],
    'Fortalezas': [
        'Puede manejar datos de alta dimensionalidad eficientemente.',
        'Puede manejar de manera eficaz conjuntos de datos no lineales mediante el uso de funciones de kernel.',
        'Busca el margen máximo entre clases, lo que puede resultar en una buena generalización.',
        'Es eficiente en espacios de alta dimensión, como en problemas de clasificación de texto o imágenes.',
        'Funciona bien en conjuntos de datos pequeños a medianos.',
        'Menos propenso al sobreajuste en comparación con algunos algoritmos más complejos.',
        'Puede ser utilizado para problemas de clasificación y regresión.'
    ],
    'Debilidades': [
        'Requiere ajuste cuidadoso de los parámetros, como el costo (C) y el parámetro de kernel.',
        'La elección del kernel y sus parámetros puede ser desafiante y dependerá del conjunto de datos.',
        'Sensible a outliers, ya que intenta maximizar el margen, y outliers pueden afectar significativamente.',
        'El rendimiento puede deteriorarse en conjuntos de datos grandes debido a la complejidad computacional.',
        'Puede volverse computacionalmente costoso y lento en conjuntos de datos grandes.',
        'La elección inadecuada de parámetros puede llevar al sobreajuste.',
        'Interpretación de los resultados no siempre es directa y puede requerir técnicas adicionales.'
    ]
}

# Lo convertimos a Dataframe
df = pd.DataFrame(data)

# Guardamos tabla
tabla = tabulate(df, headers='keys', tablefmt="html", maxcolwidths=[20, 30, 30], showindex=False, colalign=("center","left","left"))

from IPython.display import HTML, display

# Agregar estilos CSS
estilos_css = """
<style>
  table {
    width: 80%; /* Ajusta el ancho de la tabla según tus necesidades */
    margin-left: auto;
    margin-right: auto;
    border-collapse: collapse;
  }
  th, td {
    padding: 8px; /* Ajusta el relleno de celda según tus necesidades */
    border: 1px solid #dddddd;
    text-align: left;
  }
</style>
"""

# Mostrar la tabla con estilos
display(HTML(estilos_css + tabla))

"""# Secuencias promotoras en E. Coli

Los atributos del fichero de datos son:
1. Un símbolo de {+/-}, indicando la clase (“+” = promotor).
2. El nombre de la secuencia promotora. Las instancias que corresponden a no promotores se denominan
por la posición genómica.
3. Las restantes 57 posiciones corresponden a la secuencia.

La manera elegida para representar los datos es un paso crucial en los algoritmos de clasificación. En el caso que nos ocupa, análisis basados en secuencias, se usará la transformación denominada one-hot encoding.

El one-hot encoding representa cada nucleótido por un vector de 4 componentes, con 3 de ellas a 0 y una a
1 indicando el nucleótido. Pongamos por ejemplo, el nucleótido T se representa por (1,0,0,0), el nucleótido
C por (0,1,0,0), el nucleótido G por (0,0,1,0) y el nucleótido A por (0,0,0,1).

Por tanto, para una secuencia de 57 nucleótidos, como en nuestro caso, se obtendrá un vector de 4*57=228
componentes, resultado de concatenar los vectores para cada uno de los 57 nucleótidos.

Una vez realizada la transformación, one-hot encoding el objetivo se trata de predecir con SVM si la secuencia es un promotor o no, y comparar sus rendimientos

# One-Hot Encoding
Implementar una función para realizar una transformación one-hot encoding de las secuencias del fichero de datos promoters.txt. En caso de no lograr la implementación de dicha transformación, se puede utilizar el fichero promoters_onehot.txt con las secuencias codificados según una codificación one-hot para completar la actividad.
"""

import numpy as np

def one_hot_encoding(sequence):
    """
    Realiza la transformación one-hot encoding de una secuencia de nucleótidos.

    Parámetros:
    - sequence (str): La secuencia de nucleótidos a transformar.

    Devuelve:
    - np.array: Un array NumPy que representa la secuencia en formato one-hot encoding.
    """
    sequence = sequence.upper()  # Convertimos todas las letras a mayúsculas

    # Mapeo de nucleótidos a vectores one-hot
    nucleotide_mapping = {'A': [0, 0, 0, 1],
                          'G': [0, 0, 1, 0],
                          'C': [0, 1, 0, 0],
                          'T': [1, 0, 0, 0]}

    one_hot_sequence = []

    # Limitamos la longitud de la secuencia a 57 nucleótidos
    sequence = sequence[-57:]

    for nucleotide in sequence:
        if nucleotide in nucleotide_mapping:
            one_hot_sequence.extend(nucleotide_mapping[nucleotide])
        else:
            # Manejo de nucleótidos desconocidos (si los hay)
            one_hot_sequence.extend([0, 0, 0, 0])

    return np.array(one_hot_sequence)

def transform_data(file_path):
    """
    Lee un archivo que contiene datos de secuencias de ADN y etiquetas asociadas,
    y realiza la transformación one-hot encoding.

    Parámetros:
    - file_path (str): La ruta del archivo que contiene los datos.

    Devuelve:
    - X (np.array): Un array NumPy que contiene las secuencias transformadas en formato one-hot encoding.
    - y (np.array): Un array NumPy que contiene las etiquetas correspondientes a las secuencias.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extracción de secuencias y etiquetas
        sequences = []
        labels = []

        # Para cada línea, separamos elementos por comas
        for line in lines:
            elements = line.strip().split(',')

            # Verificar si la línea tiene al menos dos elementos
            if len(elements) >= 3:
                # si el primer elemento (símbolo) es +, etiquetamos con 1, si no 0
                labels.append(1 if elements[0] == '+' else 0)
                # los nombres de secuencias van en el segundo elemento
                sequence_name = elements[1]
                # el resto forma parte de la secuencia
                sequence = ','.join(elements[2:])
                sequences.append((sequence_name, sequence))

        # Realizamos one-hot encoding
        encoded_sequences = [one_hot_encoding(sequence) for _, sequence in sequences]

        return np.array(encoded_sequences), np.array(labels), sequences
    except FileNotFoundError:
        print(f"Error: El archivo '{file_path}' no se encontró.")
        return None, None, None
    except Exception as e:
        print(f"Error desconocido: {str(e)}")
        return None, None, None

"""# Clasificador SVM

(a) Leer y codificar los datos con la función one-hot desarrollada.
"""

# Aplicamos la función transformadora a nuestro ejemplo
file_path = 'promoters.txt'
X, y, sequences = transform_data(file_path)

# Imprimimos algunas secuencias y etiquetas después de la transformación one-hot encoding
for i in range(min(2, len(X))):
    sequence_name, original_sequence = sequences[i]
    transformed_sequence = X[i]

    print(f'Secuencia Original ({sequence_name}): {original_sequence}')
    print(f'Secuencia Transformada: {transformed_sequence}')
    print(f'Etiqueta: {y[i]}')
    print()

"""(b) Utilizando la semilla aleatoria 12345, separar los datos en dos partes, una parte para training (67%) y
una parte para test (33%).
"""

from sklearn.model_selection import train_test_split

# Semilla aleatoria para reproducibilidad
random_seed = 12345

# Dividir los datos en conjuntos de entrenamiento y prueba (67% para entrenamiento)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_seed)

# Imprimir las formas de los conjuntos resultantes
print("Forma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_test:", y_test.shape)

"""c) Utilizar el kernel lineal y el kernel RBF para crear sendos modelos SVM basados en el training para
predecir las clases en los datos del test.

Cuando las clases no son linealmente separables o hay mucho ruido, las SVM utilizan el "kernel trick". Este enfoque implica transformar los datos en un espacio más complejo mediante funciones kernel, como polinomiales o Radial Base Function (RBF). Los kernel polinomiales generan polinomios de mayor grado, mientras que los RBF representan funciones no lineales en áreas circulares. La elección del kernel depende del problema, siendo los RBF más propensos al sobreajuste, controlado por el parámetro "gamma" (γ).
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Crear modelos SVM con kernel lineal y RBF
svm_linear = SVC(kernel='linear', random_state=random_seed, probability=True)
svm_rbf = SVC(kernel='rbf', random_state=random_seed, probability=True)

# Entrenamos los modelos con los datos de entrenamiento
svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# Realizamos predicciones en los datos de entrenamiento y prueba
y_pred_train_linear = svm_linear.predict(X_train)
y_pred_test_linear = svm_linear.predict(X_test)

y_pred_train_rbf = svm_rbf.predict(X_train)
y_pred_test_rbf = svm_rbf.predict(X_test)

# Calculamos la precisión de los modelos en entrenamiento y prueba
accuracy_linear_test = accuracy_score(y_test, y_pred_test_linear)
accuracy_rbf_test = accuracy_score(y_test, y_pred_test_rbf)

# Imprimimos la precisión de cada modelo
print("Precisión del modelo SVM con kernel lineal en prueba: {:.4f}".format(accuracy_linear_test))
print("Precisión del modelo SVM con kernel RBF en prueba: {:.4f}".format(accuracy_rbf_test))

# Calculamos la medida F1 para SVM con kernel lineal
f1_linear = f1_score(y_test, y_pred_test_linear)
print("La medida F1 para el clasificador SVM con kernel lineal es {:.4f}".format(f1_linear))

# Calculamos la medida F1 para SVM con kernel RBF
f1_rbf = f1_score(y_test, y_pred_test_rbf)
print("La medida F1 para el clasificador SVM con kernel RBF es {:.4f}".format(f1_rbf))

# Calculamos la matriz de confusión para SVM con kernel lineal
cm_linear = confusion_matrix(y_test, y_pred_test_linear)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='PiYG')
plt.title('SVM con kernel lineal')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.show()

# Calculamos la matriz de confusión para SVM con kernel RBF
cm_rbf = confusion_matrix(y_test, y_pred_test_rbf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='PiYG')
plt.title('SVM con kernel RBF')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.show()

# Calculamos la curva ROC y AUC para SVM con kernel lineal
y_probs_linear = svm_linear.predict_proba(X_test)[:, 1]
fpr_linear, tpr_linear, _ = roc_curve(y_test, y_probs_linear)
roc_auc_linear = auc(fpr_linear, tpr_linear)

# Visualizamos la curva ROC para SVM con kernel lineal
plt.figure()
plt.plot(fpr_linear, tpr_linear, color='yellowgreen', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_linear))
plt.plot([0, 1], [0, 1], color='darkmagenta', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM con kernel lineal')
plt.legend(loc='lower right')
plt.show()

# Calculamos la curva ROC y AUC para SVM con kernel RBF
y_probs_rbf = svm_rbf.predict_proba(X_test)[:, 1]
fpr_rbf, tpr_rbf, _ = roc_curve(y_test, y_probs_rbf)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)

# Visualizamos la curva ROC para SVM con kernel RBF
plt.figure()
plt.plot(fpr_rbf, tpr_rbf, color='yellowgreen', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_rbf))
plt.plot([0, 1], [0, 1], color='darkmagenta', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM con kernel RBF')
plt.legend(loc='lower right')
plt.show()

"""d) Usar el paquete caret con el modelo svmLinear para implementar un SVM con kernel lineal y 3-fold
crossvalidation. Comentar los resultados

`caret` es un paquete de R,sin embargo, en Python, podemos utilizar bibliotecas como `scikit-learn` para realizar SVM con kernel lineal y validación cruzada.
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score

# Creamos un modelo SVM con kernel lineal
svm_linear_fold = SVC(kernel='linear', random_state=random_seed)

# Configuramos la validación cruzada con 3 folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

# Realizamos la validación cruzada y obtenemos los resultados
scores_cv = cross_val_score(svm_linear_fold, X_train, y_train, cv=cv, scoring='accuracy')
accuracy_train_linear_fold = scores_cv.mean()
svm_linear_fold.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred = svm_linear_fold.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy_test_linear_fold = accuracy_score(y_test, y_pred)

# Visualizamos la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='PiYG')
plt.title("SVM con kernel lineal (3-fold cross-validation)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Calculamos la curva ROC
y_probs_linear_fold = svm_linear_fold.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_probs_linear_fold)
roc_auc_linear_fold = auc(fpr, tpr)

# Visualizamos la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='yellowgreen', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_linear_fold))
plt.plot([0, 1], [0, 1], color='darkmagenta', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM con kernel lineal (3-fold cross-validation)')
plt.legend(loc='lower right')
plt.show()

# Calculamos la medida F1
f1_linear_fold = f1_score(y_test, y_pred)

print("La accuracy para el clasificador SVM con kernel lineal (3-fold cross-validation) es {:.4f}".format(accuracy_test_linear_fold))
print("La medida F1 para el clasificador SVM con kernel lineal (3-fold cross-validation) es {:.4f}".format(f1_linear_fold))

"""El resultado en valores de accuracy y matriz de confusión es idéntico al obtenido con SVM con kernel lineal

e) Evaluar el rendimiento del algoritmo SVM con kernel RBF para diferentes valores de los hiperparámetros
C y sigma. Orientativamente, se propone explorar valores de sigma en el intervalo (0.005,0.5) y
valores de C en el intervalo (0.1, 2). Una manera fácil de hacerlo es utilizar el paquete caret con
el modelo svmRadial. Mostrar un gráfico del rendimiento según los valores de los hiperparámetros
explorados. Comentar los resultados.

Podemos realizar la búsqueda de hiperparámetros directamente en Python utilizando la biblioteca scikit-learn, que tiene una implementación de SVM con kernel RBF.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Definimos los parámetros a explorar durante nuestra búsqueda
param_grid = {'C': np.arange(0.1, 2.1, 0.1),
              'gamma': np.arange(0.005, 0.505, 0.005)}

# Creamos un clasificador SVM con kernel RBF
svm_rbf = SVC(kernel='rbf')

# Utilizamos GridSearchCV para buscar los mejores hiperparámetros en la
# cuadrícula definida. Empleamos validación cruzada de 3 folds
grid_search = GridSearchCV(svm_rbf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Resultados de la búsqueda
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Empleamos el mejor modelo para las predicciones en el conjunto de prueba
y_pred_rbf_hiper_train = best_model.predict(X_train)
accuracy_rbf_hiper_train = accuracy_score(y_train, y_pred_rbf_hiper_train)
print("Exactitud en el conjunto de entrenamiento:", accuracy_rbf_hiper_train)

# Empleamos el mejor modelo para las predicciones en el conjunto de prueba
y_pred_rbf_hiper = best_model.predict(X_test)

accuracy_rbf_hiper = accuracy_score(y_test, y_pred_rbf_hiper)
print("Exactitud en el conjunto de prueba:", accuracy_rbf_hiper)

# Los resultados de la búsqueda se convierten a una matriz de puntuaciones
scores = np.array(grid_search.cv_results_['mean_test_score']).reshape(len(param_grid['C']), len(param_grid['gamma']))

# Gráfico de mapa de calor de la exactitud media según los hiperparámetros
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.PiYG)
plt.xlabel('Gamma')  # etiqueta eje x
plt.ylabel('C')      # etiqueta eje y
plt.colorbar(format='%.2f')  # Barra de color

# Ajustamos las etiquetas en los ejes para mejorar la legibilidad
tick_positions_gamma = np.linspace(0, len(param_grid['gamma']) - 1, 20, dtype=int)
tick_positions_c = np.linspace(0, len(param_grid['C']) - 1, 6, dtype=int)

plt.xticks(tick_positions_gamma, ['%.3f' % param_grid['gamma'][i] for i in tick_positions_gamma], rotation=45)
plt.yticks(tick_positions_c, ['%.1f' % param_grid['C'][i] for i in tick_positions_c])
plt.title('Exactitud media en validación cruzada')
plt.show()

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_rbf_hiper)

# Crear un mapa de calor con seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(cm,  annot=True, fmt='d', cmap='PiYG')
plt.title("Matriz de confusión para el clasificador SVM con kernel RBF (mejores hiperparámetros)")
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas verdaderas")
plt.show()

# Reporte de clasificación
print(metrics.classification_report(y_test, y_pred_rbf_hiper))

# F1 Score
f1_rbf_hiper = metrics.f1_score(y_test, y_pred_rbf_hiper)

# Calcular la curva ROC
y_probs_rbf = best_model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_probs_rbf)
roc_auc_hiper = auc(fpr, tpr)

# Visualizar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='yellowgreen', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_hiper))
plt.plot([0, 1], [0, 1], color='darkmagenta', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el clasificador SVM con kernel RBF (mejores hiperparámetros)')
plt.legend(loc='lower right')
plt.show()

"""f) Crear una tabla resumen de los diferentes modelos y sus rendimientos. Comentar y comparar los
resultados de la clasificación en función de los valores generales de la clasificación como accuracy y
otros para los diferentes clasificadores obtenidos. ¿Qué modelo resulta ser el mejor?
"""

from tabulate import tabulate

# Creamos un diccionario con los resultados
results_df = pd.DataFrame({
    'Modelo': ['SVM Lineal', 'SVM RBF', 'SVM Lineal (k-fold)', 'SVM RBF (Grid Search)'],
    'Precisión': [accuracy_linear_test, accuracy_rbf_test, accuracy_test_linear_fold, accuracy_rbf_hiper],
    'F1': [f1_linear, f1_rbf, f1_linear_fold, f1_rbf_hiper],
    'AUC ROC': [roc_auc_linear, roc_auc_rbf, roc_auc_linear_fold, roc_auc_hiper]
})

# Imprimir la tabla de comparación con tabulate
table = tabulate(results_df, headers='keys', tablefmt='html', showindex= False)
# Mostrar la tabla con estilos
display(HTML(estilos_css + table))

"""Estos son los resultados para cada modelo:

1. El modelo **SVM lineal** muestra un rendimiento sólido en términos de precisión y F1, indicando que es capaz de clasificar bien las instancias. Además, el valor AUC ROC de 0.949 sugiere un buen rendimiento en la clasificación binaria.

2. El modelo **SVM con kernel RBF** también proporciona buenos resultados. Aunque la precisión es menor en comparación con el modelo lineal, el F1 y el AUC ROC siguen siendo bastante buenos. El AUC ROC de 0.969 sugiere un rendimiento robusto en la clasificación binaria.

3. El modelo **SVM lineal con validación cruzada de 3 pliegues** muestra resultados similares al modelo SVM lineal sin validación cruzada. Esto sugiere que el modelo es consistente en diferentes particiones de los datos.

4. El modelo **SVM con kernel RBF utilizando la búsqueda de hiperparámetros** muestra un rendimiento equilibrado. Aunque la precisión es menor en comparación con los modelos lineales, el F1 y el AUC ROC son buenos, indicando que el modelo es capaz de clasificar con eficacia.

Por último, aplicaremos una reducción de dimensionalidad (PCA) para mostrar la frontera de decisión de nuestro modelo SVM lineal
"""

from sklearn.decomposition import PCA

# Aplicamos PCA para reducir la dimensionalidad a 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Entrenamos el modelo con los datos reducidos
svm_linear.fit(X_train_pca, y_train)

colors = ("darkmagenta", "yellowgreen")

# Graficamos la frontera de decisión
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette=colors, s=50, edgecolor='k')

# Creamos una malla para graficar la frontera de decisión
h = .02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficamos la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.3, cmap='PiYG')
plt.title('Frontera de Decisión - SVM con lineal - PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

"""El lado magenta muestra la zona en la que entran la mayoría de secuencias no promotoras (0), mientras que la zona verde muetra que las que si se consideran en su mayoría promotoras (1).

Ahora mostraremos la frontera de decisión para el SVM con kernel RBF con los "mejores" hiperparámetros.
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Aplicamos PCA para reducir la dimensionalidad a 3
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Entrenamos el modelo con los datos reducidos
best_model.fit(X_train_pca, y_train)

# Graficamos la frontera de decisión en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Puntos de datos
colors = ["darkmagenta", "yellowgreen"]
scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='viridis', s=50, edgecolor='k')

# Creamos una malla para graficar la frontera de decisión
h = .02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
z_min, z_max = X_train_pca[:, 2].min() - 1, X_train_pca[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))
Z = best_model.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Graficamos la superficie de la frontera de decisión
ax.plot_trisurf(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], cmap='PiYG', alpha=0.3)

# Leyenda para los puntos de datos
legend_labels = ['No promotor', 'Promotor']
legend = ax.legend(handles=scatter.legend_elements()[0], title="Classes", labels=legend_labels)
ax.add_artist(legend)

plt.show()

"""La representación gráfica de las funciones discriminantes en ocasiones parece que
no coinciden con la posición de las instancias del conjunto de datos. Esto es
totalmente natural, dado que en realidad la frontera se visualiza directamente sobre
las nuevas variables que se construyen para poder encontrar una función
discriminante lineal de separación entre los datos.

# Discusión

Las SVM son poderosas para la clasificación, destacando por su capacidad de manejar problemas difíciles mediante el uso de hiperplanos con margen máximo y el kernel trick para datos no lineales. La elección del kernel y el parámetro C son cruciales para obtener un equilibrio entre precisión y generalización.

* El modelo SVM lineal proporciona una interpretación directa de las características, ya que opera en el espacio original de características. Esto puede ser beneficioso si es importante comprender la importancia relativa de las variables en la toma de decisiones del modelo.
* Por otro lado, el modelo SVM con kernel RBF opera en un espacio de características transformado, lo que dificulta la interpretación directa de las características. Sin embargo, su capacidad para manejar relaciones no lineales puede ser crucial en conjuntos de datos más complejos.

Si la precisión es la métrica clave y el rendimiento en la clasificación binaria es aceptable, el modelo SVM Lineal parece ser una buena opción. Sin embargo, si el equilibrio entre precisión y recall es crucial y se puede tolerar una ligera disminución en la precisión, el modelo SVM RBF con búsqueda de hiperparámetros también es sólido, especialmente dada su fuerte AUC ROC.
"""
