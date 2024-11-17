
import pandas as pd
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pydantic import BaseModel

data = pd.read_excel(r"Data-set-estudiantes-matriculados-2023.xlsx")
data.head()
data.info()

#Eliminar datos faltantes 
data.dropna(inplace=True)

# Eliminar filas con "Sin información" en cualquier columna
data = data[~data.apply(lambda row: row.astype(str).str.contains('Sin información').any(), axis=1)]


#Conteo de subniveles de las diferentes columnas categoricas 
cols_cat =['CÓDIGO DE LA INSTITUCIÓN','INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)','TIPO IES','SECTOR IES',
           'CARÁCTER IES','DEPARTAMENTO DE DOMICILIO DE LA IES','IES ACREDITADA','PROGRAMA ACADÉMICO','PROGRAMA ACREDITADO',
           'NIVEL ACADÉMICO','NIVEL DE FORMACIÓN','MODALIDAD','ÁREA DE CONOCIMIENTO','NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)',
           'DESC CINE CAMPO AMPLIO','DESC CINE CAMPO ESPECIFICO','DESC CINE CAMPO DETALLADO','DEPARTAMENTO DE OFERTA DEL PROGRAMA']

for col in cols_cat:
     print(f'Columna {col}: {data[col].nunique()}subniveles')


print(data.describe())
#Se elimina la columna año porque tiene un unico valor el año 2023 
data= data.drop(columns=['AÑO'])

#Filas repetidas
print(f'Tamaño del set de datos Con filas repetidas:{data.shape}' )
data.drop_duplicates(inplace=True)
print(f'Tamaño del set de datos SIN filas repetidas:{data.shape}' )
#Concluimos que no tiene filas repetidas

#Revisamos si hay valores extremos o Outhliers en las variables numericas
#Generar Graficas individuales para las variables numericas


# Definir las columnas numéricas para los gráficos
cols_num = ['IES PADRE', 'ID SECTOR IES', 'ID CARÁCTER IES', 'CÓDIGO DEL DEPARTAMENTO (IES)', 
            'CÓDIGO DEL MUNICIPIO IES', 'CÓDIGO SNIES DEL PROGRAMA', 'ID NIVEL ACADÉMICO', 
            'ID NIVEL DE FORMACIÓN', 'ID MODALIDAD', 'ID ÁREA', 'ID NÚCLEO', 
            'ID CINE CAMPO AMPLIO', 'ID CINE CAMPO ESPECIFICO', 'ID CINE CAMPO DETALLADO', 
            'CÓDIGO DEL DEPARTAMENTO (PROGRAMA)', 'DEPARTAMENTO DE OFERTA DEL PROGRAMA', 
            'CÓDIGO DEL MUNICIPIO (PROGRAMA)', 'ID SEXO', 'SEMESTRE', 'MATRICULADOS']

# Iterar sobre las columnas numéricas y generar un gráfico por cada una
#for col in cols_num:
    # Crear una nueva figura para cada gráfico
    #plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
    #sns.boxenplot(x=col, data=data)
    #plt.title(f'Boxenplot de {col}')  # Título del gráfico
    #plt.xlabel(col)  # Etiqueta en el eje x
    #plt.ylabel('Valores')  # Etiqueta en el eje y
    #plt.show()  # Mostrar el gráfico

#Errores tipograficos
#Graficar subniveles de cada variable categorica 

#for col in cols_cat:
    # Crear una nueva figura para cada gráfico
    #plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
    #sns.boxenplot(x=col, data=data)
    #plt.title(f'Boxenplot de {col}')  # Título del gráfico
    #plt.xlabel(col)  # Etiqueta en el eje x
    #plt.ylabel('Valores')  # Etiqueta en el eje y
    #plt.show()  # Mostrar el gráfico

#Hata este punto es la limpieza de datos 


#Regresion Lineal 


# 1. Definir las variables independientes (X) y dependientes (y)
X = data[["IES PADRE", "ID NIVEL ACADÉMICO", "ID NIVEL DE FORMACIÓN", 
          "ID CARÁCTER IES", "ID NÚCLEO", "CÓDIGO SNIES DEL PROGRAMA", 
          "CÓDIGO DEL MUNICIPIO (PROGRAMA)", "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)"]]  # Variables predictoras

y = data["CÓDIGO DE LA INSTITUCIÓN"]  # Variable objetivo

# 2. Inicializar y ajustar el modelo de regresión lineal
reg = LinearRegression()
reg.fit(X, y)

# Mostrar los coeficientes y la intersección del modelo
print("Coeficientes (B1):", reg.coef_)
print("Intercepto (B0):", reg.intercept_)

# 3. Hacer predicciones con el modelo ajustado
y_pred = reg.predict(X)

# --- Visualización de los resultados ---

# Corregir: Graficar las predicciones vs los valores reales de "CÓDIGO DE LA INSTITUCIÓN"
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.5, label="Predicciones vs Reales", color='blue')

# Agregar etiquetas y título
plt.title("Comparación de CÓDIGO DE LA INSTITUCIÓN: Reales vs Predicciones")
plt.xlabel("CÓDIGO DE LA INSTITUCIÓN (Reales)")
plt.ylabel("CÓDIGO DE LA INSTITUCIÓN (Predicciones)")
plt.legend()

# Mostrar el gráfico
plt.show()

# Crear un gráfico con la línea de regresión
plt.figure(figsize=(10, 6))

# Graficar los datos reales vs las predicciones
plt.scatter(y, y_pred, color='blue', label='Datos reales vs Predicciones')

# Graficar la línea de regresión (predicciones)
plt.plot(y, y_pred, color='red', label='Línea de regresión')

# Agregar etiquetas y título
plt.xlabel('CÓDIGO DE LA INSTITUCIÓN (Reales)')
plt.ylabel('CÓDIGO DE LA INSTITUCIÓN (Predicciones)')
plt.title('Regresión Lineal entre CÓDIGO DE LA INSTITUCIÓN (Reales y Predicciones)')

# Mostrar la leyenda y activar la cuadrícula
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()

# --- Evaluación del Modelo ---

# Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar el modelo con los datos de entrenamiento
reg.fit(X_train, y_train)

# Hacer las predicciones sobre el conjunto de prueba
y_pred_test = reg.predict(X_test)

# 4. Graficar los resultados del conjunto de prueba
plt.figure(figsize=(10, 6))

# Graficar los datos reales de X_test vs y_test
plt.scatter(y_test, y_pred_test, color='blue', label='Datos reales vs Predicciones')

# Graficar la línea de regresión (predicciones)
plt.plot(y_test, y_pred_test, color='green', label='Línea de regresión')

# Agregar etiquetas y título
plt.title('Regresión Lineal: CÓDIGO DE LA INSTITUCIÓN (Test)')
plt.xlabel('CÓDIGO DE LA INSTITUCIÓN (Reales Test)')
plt.ylabel('CÓDIGO DE LA INSTITUCIÓN (Predicciones Test)')

# Mostrar la leyenda y activar la cuadrícula
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()

# --- Métricas del Modelo ---

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, y_pred_test)

# Calcular el Error Medio Absoluto (MAE)
mae = mean_absolute_error(y_test, y_pred_test)

# Calcular el Coeficiente de Determinación (R^2)
r2 = r2_score(y_test, y_pred_test)

# Mostrar los resultados
print("Error cuadrático medio (MSE):", mse)
print("Error medio absoluto (MAE):", mae)
print("Coeficiente de determinación (R^2):", r2)

# --- FastAPI Service ---
app = FastAPI()

# Modelo de datos para la entrada
class PredictionRequest(BaseModel):
    IES_PADRE: float
    ID_NIVEL_ACADEMICO: float
    ID_NIVEL_FORMACION: float
    ID_CARACTER_IES: float
    ID_NUCLEO: float
    CODIGO_SNIES_PROGRAMA: float
    CODIGO_MUNICIPIO_PROGRAMA: float
    CODIGO_DEPARTAMENTO_PROGRAMA: float

# Ruta para evaluar el modelo
@app.get("/evaluate")
def evaluate():
    # Hacer predicciones en el conjunto de prueba
    y_pred_test = reg.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    return {
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2
    }

# Ruta de prueba
@app.get("/")
def root():
    return {"message": "API para modelo de regresión lineal funcionando correctamente"}

# Iniciar el servidor (para ejecutar: el archivo de Regrecion_lineal.py)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)