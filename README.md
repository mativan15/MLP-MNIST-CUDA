# MLP MNIST CUDA

Integrantes:
- Iván Matthias Sardon Medina
- Gabriela Santos
- Jose Valdivia
- Steffano Ballesteros
- Diego Vasquez

Proyecto educativo en C++17 y CUDA para entrenar y ejecutar una red neuronal
MLP que reconoce digitos MNIST del 0 al 9 en Windows con GPU NVIDIA.

Arquitectura:

```text
784 -> 512 -> 256 -> 128 -> 64 -> 10
```

Cada imagen MNIST de `28x28` pixeles se aplana a un vector de `784` valores.
Las capas ocultas usan ReLU. La salida produce 10 logits, uno por digito, y el
entrenamiento usa softmax + cross entropy. Los pesos se actualizan con SGD por
muestra.

## Requisitos

Instala esto en Windows antes de compilar:

- Windows 10 o Windows 11.
- Una GPU NVIDIA compatible con CUDA.
- Driver NVIDIA instalado.
- CUDA Toolkit instalado y disponible en el `PATH`.
- Visual Studio con el componente "Desktop development with C++".
- CMake 3.18 o superior.
- Python 3 solo para descargar MNIST o preprocesar imagenes propias.
- Pillow solo si usaras `scripts\preprocess_image.py`.

Abre una terminal donde existan `cl`, `cmake` y `nvcc`. Lo mas simple es usar:

```text
x64 Native Tools Command Prompt for VS 2022
```

Verifica las herramientas:

```bat
where cl
where cmake
where nvcc
nvcc --version
```

Si alguno de esos comandos falla, corrige la instalacion antes de compilar.

## Entrar al proyecto

Desde la terminal de Windows, entra a la carpeta del proyecto:

```bat
cd /d C:\ruta\a\MLP-MNIST-CUDA
```

Si estas parado en el directorio padre:

```bat
cd MLP-MNIST-CUDA
```

## Datos MNIST

El programa espera estos archivos:

```text
data\train-images-idx3-ubyte
data\train-labels-idx1-ubyte
data\t10k-images-idx3-ubyte
data\t10k-labels-idx1-ubyte
```

Si `data\` ya contiene esos archivos, no tienes que hacer nada.

Si faltan, descargalos con:

```bat
python scripts\download_mnist.py
```

## Compilar

Opcion directa:

```bat
build_windows.bat
```

Ese script ejecuta:

```bat
cmake -S . -B build
cmake --build build --config Release
```

El ejecutable queda en:

```text
build\Release\mlp_mnist.exe
```

Tambien puedes compilar manualmente con los mismos comandos:

```bat
cmake -S . -B build
cmake --build build --config Release
```

Para limpiar y recompilar desde cero, borra la carpeta `build` y vuelve a
compilar:

```bat
rmdir /s /q build
build_windows.bat
```

## Probar que ejecuta

Muestra las opciones disponibles:

```bat
build\Release\mlp_mnist.exe --help
```

Prueba rapida con pocas muestras:

```bat
build\Release\mlp_mnist.exe --epochs 1 --train-limit 1000 --test-limit 1000 --progress 100 --save models\mlp_mnist.bin
```

Ese comando:

- carga MNIST desde `data\`;
- entrena 1 epoch usando solo 1000 imagenes;
- evalua con 1000 imagenes de prueba;
- imprime progreso cada 100 muestras;
- guarda el modelo en `models\mlp_mnist.bin`.

## Entrenamiento completo

Entrenar 3 epochs con todos los datos:

```bat
build\Release\mlp_mnist.exe --epochs 3 --progress 1000 --save models\mlp_mnist.bin
```

Entrenar con dropout:

```bat
build\Release\mlp_mnist.exe --epochs 3 --dropout 0.1 --progress 1000 --save models\mlp_mnist.bin
```

Entrenar con aumento de datos:

```bat
build\Release\mlp_mnist.exe --epochs 3 --augment --progress 1000 --save models\mlp_mnist.bin
```

Usar una GPU especifica:

```bat
build\Release\mlp_mnist.exe --device 0 --epochs 3 --save models\mlp_mnist.bin
```

## Evaluar un modelo guardado

Evalua el modelo sin entrenar de nuevo:

```bat
build\Release\mlp_mnist.exe --epochs 0 --load models\mlp_mnist.bin
```

Evaluacion corta:

```bat
build\Release\mlp_mnist.exe --epochs 0 --load models\mlp_mnist.bin --test-limit 1000 --examples 10
```

## Predecir una imagen propia

Instala Pillow si no lo tienes:

```bat
python -m pip install pillow
```

Convierte una imagen a un vector de 784 floats:

```bat
python scripts\preprocess_image.py --input images\testsImagenes\test1.png --output images\ready\test1.txt --save-preprocessed images\preprocessed\test1.png
```

Predice usando el modelo guardado:

```bat
build\Release\mlp_mnist.exe --epochs 0 --load models\mlp_mnist.bin --predict images\ready\test1.txt
```

La salida principal tiene esta forma:

```text
PREDICCION: 7
  confidence=98.5%
```

## Opciones del ejecutable

```text
--data DIR              Carpeta con archivos IDX MNIST. Default: data
--device N              GPU CUDA a usar. Default: 0
--epochs N              Numero de epochs. Default: 3
--lr VALUE              Learning rate inicial. Default: 0.01
--lr-decay VALUE        Factor de decaimiento del learning rate. Default: 0.5
--lr-decay-every N      Cada cuantos epochs aplicar decaimiento. Default: 3
--dropout VALUE         Dropout en capas ocultas. Default: 0
--augment               Activa aumento de datos durante entrenamiento
--save PATH             Ruta para guardar el modelo
--load PATH             Ruta para cargar un modelo
--predict PATH          Vector de 784 floats para predecir
--train-limit N         Limita muestras de entrenamiento. 0 usa todo
--test-limit N          Limita muestras de prueba. 0 usa todo
--examples N            Cantidad de predicciones de ejemplo
--progress N            Imprime progreso cada N muestras
--seed N                Semilla de inicializacion
--help                  Muestra ayuda
```

## Archivos principales

- `src\main.cpp`: lee opciones, carga datos, entrena, evalua y predice.
- `src\mlp_cuda.hpp`: interfaz C++ del modelo.
- `src\mlp_cuda.cu`: buffers CUDA, kernels, forward, backpropagation y SGD.
- `src\mnist_loader.cpp`: carga y normaliza archivos IDX de MNIST.
- `src\utils.hpp`: temporizador, rutas, limites y progreso.
- `scripts\download_mnist.py`: descarga los archivos MNIST.
- `scripts\preprocess_image.py`: convierte imagenes propias a vectores de 784 floats.
- `models\`: carpeta para modelos `.bin`.
- `images\ready\`: vectores listos para `--predict`.
- `images\preprocessed\`: vistas previas de 28x28 pixeles.
