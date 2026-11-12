# Generación de pseudodígitos con RNCF y programación genética

Este experimento deriva de `programacion_genetica_gp_piramide.py` y mantiene **14 árboles de programación genética**: `f1`–`f5`, `g1`–`g2`, `h11`–`h23` e `i`. Reutiliza sin modificar los operadores y terminales de `funciones_terminales_gp_multicapa.py` y la implementación de DoG del archivo original.

## Arquitectura

```text
Superficie trigonométrica (28x28)
    -> f1..f5 (5 x 28x28)
    -> g1,g2 (2 x 28x28)
    -> dos pirámides Gaussianas de 4 niveles (28,14,7,4)
    -> DoG fusionados (2 x 28x28)
    -> h11,h12,h13,h21,h22,h23 (6 x 28x28)  [SIN MAXPOOL tras DoG]
    -> i (28x28)
    -> MaxPool final (1 x 14x14)
    -> Flatten (196)
    -> reshape + resize a 28x28 + normalización [0,1]
    -> MLP MNIST previamente entrenado (congelado)
    -> [P(1), P(2), ..., P(9)]
```

El programa original usa `np.arange(1, 28)`, que produce 27x27. Aquí se utiliza `np.arange(1, 29)` para producir 28x28, manteniendo `a=3` y `b=60`. El último MaxPool se conserva para respetar el resto de la arquitectura; `conservar_maxpool_final=False` lo desactiva experimentalmente, produciendo 784 valores nativos tras Flatten.

## Archivos

- `gp_generacion_pseudodigitos.py`: programa principal, entrenamiento/caché del MLP, GP, evaluación, exportación.
- `rncf_generativa.py`: adaptación de la ejecución RNCF que omite el MaxPool tras DoG.
- `prueba_rncf_generativa.py`: prueba de dimensiones sin DEAP ni MNIST.
- `funciones_terminales_gp_multicapa.py`: originales del usuario; **sin cambios**.
- `red_convolucional_funcional_gp_piramide.py`: arquitectura RNCF original; **sin cambios**.
- `programacion_genetica_gp_piramide.py`: código original; **sin cambios**. Se reutiliza su función `cargar_mnist` con verificación SHA256.

## Instalación y ejecución

Coloca los cinco `.py` en la **misma carpeta** y ejecuta:

```bash
python -m pip install -r requirements_RNCF_generativa.txt
python prueba_rncf_generativa.py    # Verifica las dimensiones de la RNCF
python gp_generacion_pseudodigitos.py
```

La primera ejecución descargará MNIST (si no existe `datos/mnist.npz`), entrenará una sola vez un `MLPClassifier` con dígitos del 1 al 9 y guardará su modelo en `datos/mlp_mnist_1_9.joblib`. Las siguientes ejecuciones **reutilizan el MLP** cuando coincide la configuración del entrenamiento. Durante el GP **no se reentrena el clasificador**. No se necesita TensorFlow ni una GPU.

Edita `Configuracion(...)` en la función `main()` de `gp_generacion_pseudodigitos.py` para seleccionar `digito_objetivo=4` (1 a 9), `poblacion`, `generaciones` y `semilla`.

## Fitness

Minimiza:

```text
1 - P(objetivo|imagen)
  + penalizacion_nodos * nodos_totales
  + peso_bordes * luminosidad_media_del_borde
  + peso_colapso * penalizacion_de_imagen_degenerada
  + peso_referencia * distancia_a_MNIST
```

- Por defecto, `peso_referencia=0.0`: ningún ejemplo real participa en la **evaluación de GP**.
- Puedes activar `peso_referencia=0.15` para favorecer la semejanza con ejemplos reales del dígito objetivo y limitar imágenes adversarias al clasificador.
- Para evaluar estrictamente probabilidad y tamaño de árboles, fija `peso_bordes=0.0`, `peso_colapso=0.0` y `peso_referencia=0.0`.
- **Una probabilidad de clase alta no es una exactitud alta**: la exactitud informada por el MLP procede de imágenes MNIST reales de validación. Las salidas GP están fuera de distribución y pueden engañar al clasificador.
- La normalización min-max de la salida se realiza por candidato; una salida constante se convierte en una imagen negra.

## Resultados

El programa guarda en `resultados_pseudodigitos/`:

- `superficie_trigonometrica.png`: entrada fija.
- `pseudodigito_final.png` y `.npy`: mejor imagen sintética.
- `referencia_real_mnist.png`: ejemplo real, **solo para comparación visual**.
- `evolucion_imagenes.png`: mejor imagen de cada generación.
- `curva_probabilidad.png`: evolución de `P(objetivo)`.
- `probabilidades_finales.png`: nueve probabilidades finales.
- `historial.csv`: evolución por generación.
- `mejor_individuo.json`: parámetros, 14 árboles, métricas y tamaños de los mapas.

## Alcance de las pruebas realizadas

El módulo de la red se probó con funciones de ejemplo: sus tamaños fueron `DoG (2,28,28)`, capa `h (6,28,28)`, MaxPool final `(1,14,14)` y Flatten `(196,)`, con imagen final `(28,28)`. También se comprobó la variante sin MaxPool final `(784,)`. Se verificó la sintaxis de los programas. El ciclo completo DEAP+MLP requiere instalar `deap` y cargar MNIST en el equipo de ejecución.
