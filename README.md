# Support-Vector-Machine
Implementación del algoritmo de clasificación SVM en python para predecir secuencias promotoras en E.coli.

Los atributos del fichero de datos son:

- Un símbolo de {+/-}, indicando la clase (“+” = promotor).
- El nombre de la secuencia promotora. Las instancias que corresponden a no promotores se denominan por la posición genómica.
- Las restantes 57 posiciones corresponden a la secuencia.

La manera elegida para representar los datos es un paso crucial en los algoritmos de clasificación. En el caso que nos ocupa, análisis basados en secuencias, se usará la transformación denominada one-hot encoding.

El one-hot encoding representa cada nucleótido por un vector de 4 componentes, con 3 de ellas a 0 y una a 1 indicando el nucleótido. Pongamos por ejemplo, el nucleótido T se representa por (1,0,0,0), el nucleótido C por (0,1,0,0), el nucleótido G por (0,0,1,0) y el nucleótido A por (0,0,0,1).

Por tanto, para una secuencia de 57 nucleótidos, como en nuestro caso, se obtendrá un vector de 4*57=228 componentes, resultado de concatenar los vectores para cada uno de los 57 nucleótidos.
