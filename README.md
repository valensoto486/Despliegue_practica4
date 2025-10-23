# Proyecto de Minería de Datos: Predicción de Empleabilidad de Estudiantes

Este proyecto tiene como objetivo **predecir si un estudiante será contratado** durante el proceso de reclutamiento en el campus universitario, utilizando técnicas de **Minería de Datos y Machine Learning**.  
El trabajo se divide en tres grandes fases: **Calidad de Datos**, **Minería de Datos** y **Despliegue del Modelo**.

---

## 1. Calidad de Datos

### A. Perfilado de Datos
Se generó un **perfilado de los datos en Python**, donde se evaluaron características estadísticas, tipos de variables, distribución de valores y presencia de datos faltantes.  
El resultado del perfilado fue exportado en formato **HTML** e incluido en este repositorio.

### B. Diagnóstico de Calidad de Datos
A partir del perfilado, se evaluaron las principales **dimensiones de calidad de datos**, entre ellas:

- **Completitud**:  Toda la información está completa, a exepción del campo de salario, ya que este solo aplica para las personas que ya están posicionadas laboralmente
- **Exactitud**: La información es correcta y libre de error. Todos los campos numéricos están en un rango adecuado , que corresponde a un promedio académico de de 0 a 100.
- **Conformidad**: Los valores están conformes con los formatos establecidos. Todas las variables categóricas tienen los formatos de las categorías unificados, sin haber discrepancias entre registros pertiecientes a una misma categoría.
- **Oportunidad**: El dataset es de hace 6 años, por lo que las condiciones educativas y las oportunidades laborales pueden haber cambiado mucho, restándole relevancia a la información que se extraiga de los datos. Sin embargo, puede ser muy útil para el aprendizaje
- **Duplicidad**: No existen registros duplicados, pero existen algunas variables con correlaciones altas
Integridad: Los datos son coherentes y tienen una relación clara. Un ejemplo de esto es que el campo de salario solo aplica para las personas que ya están posicionadas laboralmente, y en los registros en los que la persona no está posicionada, hay nulos

### C. Limpieza y Mejora
Limpieza de los datos:
1.	Detección de duplicados: no había duplicados
2.	Selección de datos: En este caso se seleccionaron todas las variables predictoras a excepción de sl_n que es el id de los registros y salary, ya que solo aplica para los estudiantes que ya están posicionados}
3.	Limpieza de datos atípicos: no hay datos atípicos
4.	Limpieza de datos nulos: no hay datos nulos
Mejora de los datos:
1.	Análisis de correlaciones: Se encontraron correlaciones altas entre las dummies de una misma variable( degree_t y hsc_s ). Se encontraron correlaciones menores a 0.05 respecto a la variable objetivo ( status ) de las siguientes variables: 'degree_t','hsc_s','ssc_b' y 'hsc_b'
2.	Balanceo de datos: En este caso se realiza un balanceo, ya que la variable objetivo es categórica y una de las categorías tiene menos de la mitad de los registros de la otra (148 vs 67 )

---

## 2. Minería de Datos

### A. Objetivo del Modelo
El objetivo del modelo predictivo es **anticipar si un estudiante será colocado en una oferta laboral**, a partir de variables como:
- Rendimiento académico (SSC, HSC, grado, MBA).
- Especialización profesional.
- Experiencia laboral previa.
- Factores personales y de desempeño en pruebas.

Con esto se busca **identificar patrones que influyen en la empleabilidad** y apoyar la toma de decisiones para mejorar la inserción laboral de los estudiantes.

---

### B. Modelos Predictivos Evaluados

Se desarrollaron y evaluaron los siguientes modelos:

| Modelo | Precisión (Accuracy) | Conclusión |
|---------|----------------------|-------------|
| 🌳 Árbol de Decisión | 76.4% | Rendimiento medio, base inicial sólida pero mejorable con tuning. |
| 🌲 Random Forest | 87.64% | Mejor desempeño general, equilibrio entre precisión y recall. |
| 🤝 KNN | 65.17% | Bajo rendimiento, sensible a la distancia entre puntos. |
| 🧠 Red Neuronal | 73.03% | Rendimiento moderado, necesita más iteraciones e hiperajustes. |
| ⚙️ SVM | 85.39% | Muy buen desempeño, comparable al Random Forest. |

**Conclusión general:**  
Los modelos de **Random Forest y SVM** demostraron el mejor desempeño, con altos valores de precisión y F1-score, siendo los más adecuados para este conjunto de datos.

---

### C. Hiperparametrización

Se aplicó **GridSearchCV** para optimizar los hiperparámetros del modelo de **Random Forest**, obteniendo el mejor rendimiento con la siguiente configuración:

```python
RandomForestClassifier(
    bootstrap=True,
    criterion='gini',
    max_depth=None,
    max_features='sqrt',
    max_samples=0.7,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)
```

Por:
Juana Jaramillo y Valentina Soto

