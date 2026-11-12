"""Generación evolutiva de pseudodígitos con RNCF (14 árboles GP) + MLP MNIST.

Esquema:
   imagen trigonométrica 28x28
     -> f1..f5 -> g1,g2 -> pirámides Gaussianas y DoG (sin MaxPool aquí)
     -> h11..h23 -> i -> MaxPool FINAL 2x2 -> Flatten (196)
     -> reshape 14x14 -> resize 28x28 -> min-max
     -> MLP previamente entrenado, congelado -> probabilidades [1,...,9]

Aptitud a minimizar:
    (1 - P(dígito_objetivo | imagen))
      + penalización por cantidad de nodos
      + penalizaciones suaves contra salidas degeneradas
      + peso_referencia * distancia opcional a ejemplos MNIST reales.

La probabilidad de una sola imagen NO es exactitud de clasificación.
Se reporta la exactitud del MLP sobre MNIST por separado.

Requiere los archivos originales en el mismo directorio:
  - funciones_terminales_gp_multicapa.py
  - red_convolucional_funcional_gp_piramide.py
  - programacion_genetica_gp_piramide.py (solo se reutiliza cargar_mnist)
Además de rncf_generativa.py.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import csv
import json
import random
import time

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deap import base, creator, gp, tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from funciones_terminales_gp_multicapa import construir_pset, generar_expresion
from programacion_genetica_gp_piramide import cargar_mnist
from rncf_generativa import RNCFGenerativa


@dataclass
class Configuracion:
    # --- Experimento de generación ---
    digito_objetivo: int = 0            # Elige un dígito de 0 a 9.
    a_superficie: float = 3.0
    b_superficie: float = 60.0
    ancho_imagen: int = 28
    conservar_maxpool_final: bool = True

    # --- Programación genética multiárbol ---
    poblacion: int = 24
    generaciones: int = 25
    elitismo: int = 2
    torneo: int = 3
    prob_cruce: float = 0.80
    prob_mutacion: float = 0.35
    profundidad_inicial: int = 3
    profundidad_mutacion: int = 2
    altura_maxima: int = 6
    nodos_max_arbol: int = 60
    nodos_max_individuo: int = 500
    penalizacion_nodos: float = 1e-4
    cache_evaluaciones: int = 512

    # --- Regularización ligera de las imágenes (modificable a cero) ---
    peso_bordes: float = 0.05           # Evita bordes claros no-MNIST.
    peso_colapso: float = 0.20          # Evita imágenes casi constantes.
    peso_referencia: float = 0.0        # 0 -> fitness sin ejemplo MNIST.
    numero_referencias: int = 32       # Para usar si peso_referencia > 0.

    # --- Clasificador MLP (se entrena UNA SOLA VEZ y se guarda) ---
    ejemplos_entrenamiento_mlp: int = 12000
    ejemplos_validacion_mlp: int = 2000
    capas_mlp: tuple = (128, 64)
    iteraciones_mlp: int = 45
    ruta_mnist: str = "datos/mnist.npz"
    ruta_modelo_mlp: str = "datos/mlp_mnist_1_9.joblib"
    carpeta_resultados: str = "resultados_pseudodigitos"
    semilla: int | None = 42


def construir_superficie_trigonometrica(n=28, a=3.0, b=60.0):
    """Versión 28x28 de la 'tapa de huevo' del usuario, normalizada a [0,1].

    El ejemplo original usa arange(1,28), que genera 27x27.
    Aquí se usa arange(1,29) para hacerla compatible con MNIST (28x28).
    """
    if n < 8 or a == 0 or b == 0:
        raise ValueError("Se requiere n>=8 y a,b distintos de cero.")
    x, y = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    z = np.cos(x / a) * np.sin(y / a) + np.cos(x / b) * np.sin(y / b)
    z = z.astype(np.float32)
    return (z - z.min()) / max(float(np.ptp(z)), 1e-8)


def obtener_clasificador_mlp(cfg, x_train, y_train):
    """Carga o entrena el MLP MNIST 0..9 una vez. Nunca se ajusta en GP."""
    ruta = Path(cfg.ruta_modelo_mlp)
    clave = {
        "version": 1,
        "ejemplos_entrenamiento": cfg.ejemplos_entrenamiento_mlp,
        "ejemplos_validacion": cfg.ejemplos_validacion_mlp,
        "capas": list(cfg.capas_mlp),
        "max_iter": cfg.iteraciones_mlp,
        "semilla": cfg.semilla,
    }
    if ruta.exists():
        # Cargar SOLO modelos joblib de origen confiable.
        paquete = joblib.load(ruta)
        if (isinstance(paquete, dict)
                and paquete.get("config") == clave
                and np.array_equal(paquete["modelo"].classes_, np.arange(0, 10))):
            print(f"MLP cargado de {ruta} (no se reentrena).", flush=True)
            return paquete["modelo"], paquete.get("validacion", None)
        print("La configuración del MLP cambió: se volverá a entrenar.")

    filtro = (y_train >= 0) & (y_train <= 9)
    x = x_train[filtro].astype(np.float32) / 255.0
    y = y_train[filtro].astype(np.int64)

    total = cfg.ejemplos_entrenamiento_mlp + cfg.ejemplos_validacion_mlp
    if total > len(y):
        raise ValueError("Se solicitaron más ejemplos de los disponibles en MNIST 0..9.")

    indices, _ = train_test_split(
        np.arange(len(y)), train_size=total,
        stratify=y, random_state=cfg.semilla,
    )
    ind_train, ind_valid = train_test_split(
        indices, train_size=cfg.ejemplos_entrenamiento_mlp,
        test_size=cfg.ejemplos_validacion_mlp,
        stratify=y[indices], random_state=cfg.semilla,
    )

    modelo = MLPClassifier(
        hidden_layer_sizes=cfg.capas_mlp,
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=cfg.iteraciones_mlp,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=cfg.semilla,
    )
    print(f"Entrenando MLP (solo una vez) con {len(ind_train)} ejemplos...", flush=True)
    modelo.fit(x[ind_train].reshape(-1, 784), y[ind_train])

    pred = modelo.predict(x[ind_valid].reshape(-1, 784))
    exactitud_valid = float(accuracy_score(y[ind_valid], pred))
    assert np.array_equal(modelo.classes_, np.arange(0, 10))
    print(f"Exactitud MLP en validación real: {exactitud_valid:.4f}", flush=True)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"modelo": modelo, "config": clave, "validacion": exactitud_valid},
        ruta, compress=3,
    )
    return modelo, exactitud_valid


def contar_nodos(individuo):
    return sum(len(arbol) for arbol in individuo)


def clave_individuo(individuo):
    return tuple(str(arbol) for arbol in individuo)


# Nombre específico para poder importar este módulo junto al clasificador GP.
if not hasattr(creator, "FitnessPseudoDigitoMin"):
    creator.create("FitnessPseudoDigitoMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividuoPseudoDigito"):
    creator.create(
        "IndividuoPseudoDigito", list, fitness=creator.FitnessPseudoDigitoMin
    )


class EvolucionPseudodigito:
    NOMBRES_ARB = (
        [f"f{n}(I)" for n in range(1, 6)]
        + ["g1(F1,...,F5)", "g2(F1,...,F5)"]
        + ["h11(D1)", "h12(D1)", "h13(D1)",
           "h21(D2)", "h22(D2)", "h23(D2)"]
        + ["i(H11,...,H23)"]
    )

    def __init__(self, cfg, clasificador, imagen_entrada, referencias=None):
        self.cfg = cfg
        self.clasificador = clasificador
        self.imagen_entrada = np.asarray(imagen_entrada, dtype=np.float32)
        self.referencias = None if referencias is None else np.asarray(
            referencias, dtype=np.float32
        )
        self._validar()
        random.seed(cfg.semilla)
        np.random.seed(cfg.semilla)

        self.pset_unario = construir_pset(1, "GEN_FH", ["I"])
        self.pset_g = construir_pset(
            5, "GEN_G", [f"F{i}" for i in range(1, 6)]
        )
        self.pset_i = construir_pset(
            6, "GEN_I", ["H11", "H12", "H13", "H21", "H22", "H23"]
        )
        self.psets_por_indice = (
            [self.pset_unario] * 5 + [self.pset_g] * 2
            + [self.pset_unario] * 6 + [self.pset_i]
        )

        self.toolbox = base.Toolbox()
        self.toolbox.register("clone", deepcopy)
        self.toolbox.register("individual", self.crear_individuo)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("select", tools.selTournament,
                              tournsize=cfg.torneo)
        self.toolbox.register("mate", self.cruzar)
        self.toolbox.register("mutate", self.mutar)
        self.toolbox.register("evaluate", self.evaluar)

        self.cache = OrderedDict()
        self.historial = []
        self.imagenes_historial = []
        self.mejor = None
        self.indice_objetivo = int(np.flatnonzero(
            self.clasificador.classes_ == self.cfg.digito_objetivo
        )[0])

    # Validar digitos entre 0 y 9, imagen de 28x28, elitismo < poblacion, probabilidades [0,1],
    def _validar(self):
        c = self.cfg
        if not 0 <= c.digito_objetivo <= 9:
            raise ValueError("digito_objetivo debe estar entre 0 y 9.")
        if c.ancho_imagen != 28 or self.imagen_entrada.shape != (28, 28):
            raise ValueError("Este experimento requiere una imagen de 28x28.")
        if not 1 <= c.elitismo < c.poblacion:
            raise ValueError("Se requiere 1 <= elitismo < poblacion.")
        if c.generaciones < 0 or c.torneo < 2:
            raise ValueError("Generaciones>=0 y torneo>=2.")
        if not (0 <= c.prob_cruce <= 1 and 0 <= c.prob_mutacion <= 1):
            raise ValueError("Las probabilidades deben estar entre 0 y 1.")
        if not 1 <= c.profundidad_inicial <= c.altura_maxima:
            raise ValueError("profundidad_inicial debe ser <= altura_maxima.")
        if (c.profundidad_mutacion < 0 or c.nodos_max_arbol < 2
                or c.nodos_max_individuo < 14 or c.cache_evaluaciones < 1):
            raise ValueError("Límites de complejidad incorrectos.")
        if any(x < 0 for x in [c.penalizacion_nodos, c.peso_bordes,
                                c.peso_colapso, c.peso_referencia]):
            raise ValueError("Los pesos de penalización deben ser no negativos.")
        if not np.array_equal(self.clasificador.classes_, np.arange(0, 10)):
            raise ValueError("El clasificador debe reconocer exactamente 0,...,9.")
        if c.peso_referencia > 0 and (
            self.referencias is None or self.referencias.ndim != 3
            or self.referencias.shape[1:] != (28, 28) or len(self.referencias) == 0
        ):
            raise ValueError("Para peso_referencia>0 se requieren referencias (K,28,28).")

    def dentro_limites(self, individuo):
        c = self.cfg
        return (
            len(individuo) == 14
            and contar_nodos(individuo) <= c.nodos_max_individuo
            and all(arbol.height <= c.altura_maxima
                    and len(arbol) <= c.nodos_max_arbol
                    for arbol in individuo)
        )

    def crear_individuo(self):
        for _ in range(200):
            arboles = [gp.PrimitiveTree(generar_expresion(
                pset, min_=1, max_=self.cfg.profundidad_inicial
            )) for pset in self.psets_por_indice]
            individuo = creator.IndividuoPseudoDigito(arboles)
            if self.dentro_limites(individuo):
                return individuo
        raise RuntimeError("No se pudo crear un individuo dentro de los límites.")

    def individuo_referencia(self):
        """Control funcional sin filtros: identidades en f/h y sumas g/i."""
        f = [gp.PrimitiveTree.from_string("I", self.pset_unario)
             for _ in range(5)]
        suma5 = "suma(suma(suma(suma(F1, F2), F3), F4), F5)"
        g = [gp.PrimitiveTree.from_string(suma5, self.pset_g)
             for _ in range(2)]
        h = [gp.PrimitiveTree.from_string("I", self.pset_unario)
             for _ in range(6)]
        suma6 = "suma(suma(suma(suma(suma(H11, H12), H13), H21), H22), H23)"
        i = [gp.PrimitiveTree.from_string(suma6, self.pset_i)]
        individuo = creator.IndividuoPseudoDigito(f + g + h + i)
        if not self.dentro_limites(individuo):
            raise ValueError("El individuo de referencia excede los límites.")
        return individuo

    def compilar(self, individuo):
        if len(individuo) != 14:
            raise ValueError("Se esperan catorce árboles.")
        return [gp.compile(expr=arbol, pset=pset)
                for arbol, pset in zip(individuo, self.psets_por_indice)]

    def red(self, individuo):
        return RNCFGenerativa(
            self.compilar(individuo),
            conservar_maxpool_final=self.cfg.conservar_maxpool_final,
        )

    def cruzar(self, a, b):
        # Solo se cruzan árboles de la misma función/posición.
        indice = random.randrange(14)
        anterior_a, anterior_b = deepcopy(a[indice]), deepcopy(b[indice])
        if random.random() < 0.25:
            a[indice], b[indice] = b[indice], a[indice]
        else:
            gp.cxOnePoint(a[indice], b[indice])
        if not self.dentro_limites(a):
            a[indice] = anterior_a
        if not self.dentro_limites(b):
            b[indice] = anterior_b
        return a, b

    def mutar(self, individuo):
        indice = random.randrange(14)
        arbol_previo = deepcopy(individuo[indice])
        gp.mutUniform(
            individuo[indice],
            expr=partial(generar_expresion,
                         min_=0, max_=self.cfg.profundidad_mutacion),
            pset=self.psets_por_indice[indice],
        )
        if not self.dentro_limites(individuo):
            individuo[indice] = arbol_previo
        return (individuo,)

    def imagen_generada(self, individuo):
        return self.red(individuo).generar_imagen(self.imagen_entrada)

    def _metricas(self, imagen, individuo):
        prob = self.clasificador.predict_proba(
            imagen.reshape(1, -1)
        )[0].astype(float)
        prob_obj = float(prob[self.indice_objetivo])
        if not np.isfinite(prob).all():
            raise ValueError("El clasificador devolvió probabilidades no finitas.")

        # Regularizaciones interpretables, en [0,+inf).
        std = float(np.std(imagen))
        perdida_colapso = max(0.0, 0.08 - std) / 0.08
        tasa_activados = float(np.mean(imagen > 0.2))
        if tasa_activados < 0.035:
            perdida_colapso += (0.035 - tasa_activados) / 0.035
        elif tasa_activados > 0.65:
            perdida_colapso += (tasa_activados - 0.65) / 0.35

        bordes = np.concatenate([
            imagen[0], imagen[-1], imagen[1:-1, 0], imagen[1:-1, -1]
        ])
        media_bordes = float(bordes.mean())

        if self.cfg.peso_referencia > 0:
            errores = np.mean(
                (self.referencias - imagen[None, :, :]) ** 2,
                axis=(1, 2),
            )
            distancia_referencia = float(np.sqrt(np.min(errores)))
        else:
            distancia_referencia = 0.0

        nodos = contar_nodos(individuo)
        aptitud = (
            1.0 - prob_obj
            + self.cfg.penalizacion_nodos * nodos
            + self.cfg.peso_bordes * media_bordes
            + self.cfg.peso_colapso * perdida_colapso
            + self.cfg.peso_referencia * distancia_referencia
        )
        return {
            "aptitud": float(aptitud),
            "prob_objetivo": prob_obj,
            "clase_predicha": int(self.clasificador.classes_[np.argmax(prob)]),
            "confianza_predicha": float(np.max(prob)),
            "probabilidades": prob.tolist(),
            "nodos": nodos,
            "bordes": media_bordes,
            "desviacion": std,
            "penalizacion_colapso": float(perdida_colapso),
            "distancia_referencia": distancia_referencia,
            "error": "",
        }

    def evaluar(self, individuo):
        clave = clave_individuo(individuo)
        if clave in self.cache:
            self.cache.move_to_end(clave)
            resultado = self.cache[clave]
        else:
            try:
                if not self.dentro_limites(individuo):
                    raise ValueError("Individuo fuera de límites.")
                imagen = self.imagen_generada(individuo)
                resultado = self._metricas(imagen, individuo)
            except Exception as exc:
                resultado = {
                    "aptitud": 10.0 + self.cfg.penalizacion_nodos * contar_nodos(individuo),
                    "prob_objetivo": 0.0,
                    "clase_predicha": -1,
                    "confianza_predicha": 0.0,
                    "probabilidades": [0.0] * 9,
                    "nodos": contar_nodos(individuo),
                    "bordes": 0.0,
                    "desviacion": 0.0,
                    "penalizacion_colapso": 0.0,
                    "distancia_referencia": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self.cache[clave] = resultado
            if len(self.cache) > self.cfg.cache_evaluaciones:
                self.cache.popitem(last=False)
        individuo.metricas = dict(resultado)
        return (resultado["aptitud"],)

    def evaluar_invalidos(self, individuos):
        for individuo in individuos:
            if not individuo.fitness.valid:
                individuo.fitness.values = self.toolbox.evaluate(individuo)

    def registrar(self, poblacion, generacion):
        mejor = tools.selBest(poblacion, 1)[0]
        registro = {
            "generacion": generacion,
            "fitness_min": float(mejor.fitness.values[0]),
            "fitness_media": float(np.mean([
                ind.fitness.values[0] for ind in poblacion
            ])),
            "prob_objetivo": mejor.metricas["prob_objetivo"],
            "clase_predicha": mejor.metricas["clase_predicha"],
            "nodos_mejor": contar_nodos(mejor),
            "evaluaciones_fallidas": sum(bool(ind.metricas["error"])
                                        for ind in poblacion),
        }
        self.historial.append(registro)
        self.imagenes_historial.append(self.imagen_generada(mejor))
        print(
            f"G{generacion:03d} | fitness={registro['fitness_min']:.6f} "
            f"| P({self.cfg.digito_objetivo})={registro['prob_objetivo']:.5f} "
            f"| pred={registro['clase_predicha']} "
            f"| nodos={registro['nodos_mejor']} "
            f"| fallos={registro['evaluaciones_fallidas']}",
            flush=True,
        )

    def evolucionar(self):
        self.historial = []
        self.imagenes_historial = []
        self.cache.clear()
        poblacion = self.toolbox.population(n=self.cfg.poblacion)
        poblacion[0] = self.individuo_referencia()
        self.evaluar_invalidos(poblacion)

        if all(ind.metricas["error"] for ind in poblacion):
            raise RuntimeError("Fallaron todos los individuos: "
                               + poblacion[0].metricas["error"])
        salon_fama = tools.HallOfFame(1)
        salon_fama.update(poblacion)
        self.registrar(poblacion, 0)

        for generacion in range(1, self.cfg.generaciones + 1):
            elites = [self.toolbox.clone(ind) for ind in
                      tools.selBest(poblacion, self.cfg.elitismo)]
            elegidos = self.toolbox.select(
                poblacion, self.cfg.poblacion - self.cfg.elitismo
            )
            hijos = [self.toolbox.clone(ind) for ind in elegidos]

            for a, b in zip(hijos[::2], hijos[1::2]):
                if random.random() < self.cfg.prob_cruce:
                    self.toolbox.mate(a, b)
                    del a.fitness.values
                    del b.fitness.values

            for hijo in hijos:
                if random.random() < self.cfg.prob_mutacion:
                    self.toolbox.mutate(hijo)
                    if hijo.fitness.valid:
                        del hijo.fitness.values

            self.evaluar_invalidos(hijos)
            poblacion = elites + hijos
            salon_fama.update(poblacion)
            self.registrar(poblacion, generacion)

        self.mejor = self.toolbox.clone(salon_fama[0])
        return self.mejor

    def guardar(self, exactitud_mlp):
        if self.mejor is None:
            raise RuntimeError("Evolucione primero la población.")
        carpeta = Path(self.cfg.carpeta_resultados)
        carpeta.mkdir(parents=True, exist_ok=True)
        imagen_final = self.imagen_generada(self.mejor)
        cv2.imwrite(str(carpeta / "superficie_trigonometrica.png"),
                    np.rint(self.imagen_entrada * 255).astype(np.uint8))
        cv2.imwrite(str(carpeta / "pseudodigito_final.png"),
                    np.rint(imagen_final * 255).astype(np.uint8))
        np.save(carpeta / "pseudodigito_final.npy", imagen_final)

        documento = {
            "configuracion": asdict(self.cfg),
            "exactitud_validacion_MLP_real": exactitud_mlp,
            "metricas_mejor_individuo": self.mejor.metricas,
            "arboles_evolucionados": dict(zip(
                self.NOMBRES_ARB, clave_individuo(self.mejor)
            )),
            "nodos_por_arbol": [len(a) for a in self.mejor],
        }
        # ejecutar primero para recopilar tamaños reales
        red_final = self.red(self.mejor)
        red_final.generar_imagen(self.imagen_entrada)
        documento["formas_intermedias"] = {
            clave: str(valor) for clave, valor in
            red_final.formas_intermedias.items()
        }
        (carpeta / "mejor_individuo.json").write_text(
            json.dumps(documento, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (carpeta / "historial.csv").open(
            "w", newline="", encoding="utf-8"
        ) as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=self.historial[0].keys())
            escritor.writeheader()
            escritor.writerows(self.historial)

        fig = plt.figure(figsize=(8, 3))
        plt.plot([r["generacion"] for r in self.historial],
                 [r["prob_objetivo"] for r in self.historial], marker="o", ms=3)
        plt.title(f"Probabilidad del dígito objetivo {self.cfg.digito_objetivo}")
        plt.xlabel("Generación")
        plt.ylabel("P(objetivo)")
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(carpeta / "curva_probabilidad.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 3))
        clases = self.clasificador.classes_.astype(int)
        probabilidades = self.mejor.metricas["probabilidades"]
        plt.bar(clases, probabilidades)
        plt.xticks(clases)
        plt.xlabel("Dígito")
        plt.ylabel("Probabilidad MLP")
        plt.ylim(0, 1.02)
        plt.title("Predicción sobre el pseudodígito final")
        fig.tight_layout()
        fig.savefig(carpeta / "probabilidades_finales.png", dpi=180)
        plt.close(fig)

        # Montaje: la imagen de entrada y el mejor resultado de cada generación.
        fotos = [self.imagen_entrada] + self.imagenes_historial
        etiquetas = ["Entrada"] + [f"G{k}" for k in range(len(self.historial))]
        columnas = 6
        filas = int(np.ceil(len(fotos) / columnas))
        fig, ejes = plt.subplots(filas, columnas, figsize=(columnas * 2, filas * 2))
        ejes = np.asarray(ejes).reshape(-1)
        for eje, foto, etiqueta in zip(ejes, fotos, etiquetas):
            eje.imshow(foto, cmap="gray", vmin=0, vmax=1)
            eje.set_title(etiqueta)
            eje.axis("off")
        for eje in ejes[len(fotos):]:
            eje.axis("off")
        fig.tight_layout()
        fig.savefig(carpeta / "evolucion_imagenes.png", dpi=150)
        plt.close(fig)
        return carpeta


def elegir_referencias(x_train, y_train, cfg):
    if cfg.peso_referencia == 0.0:
        return None
    x = x_train[y_train == cfg.digito_objetivo].astype(np.float32) / 255.0
    rng = np.random.default_rng(cfg.semilla)
    indices = rng.choice(len(x), size=min(cfg.numero_referencias, len(x)),
                         replace=False)
    return x[indices]


def main():
    configuracion = Configuracion(
        digito_objetivo=0,               # Cambiar por 0,...,9
        poblacion=24,
        generaciones=25,
        elitismo=2,
        semilla=41,
        # peso_referencia=0.15,         # Opcional: favorece similitud visual MNIST.
        # conservar_maxpool_final=False, # Opcional: salida nativa 28x28 sin pool final.
    )
    inicio = time.perf_counter()
    entrada = construir_superficie_trigonometrica(
        configuracion.ancho_imagen,
        configuracion.a_superficie,
        configuracion.b_superficie,
    )
    print("Entrada trigonométrica:", entrada.shape)
    x_train, y_train, _, _ = cargar_mnist(configuracion.ruta_mnist)
    clasificador, exactitud_mlp = obtener_clasificador_mlp(
        configuracion, x_train, y_train
    )
    motor = EvolucionPseudodigito(
        configuracion,
        clasificador,
        entrada,
        elegir_referencias(x_train, y_train, configuracion),
    )
    mejor = motor.evolucionar()
    carpeta = motor.guardar(exactitud_mlp)
    # Referencia visual únicamente; no interviene en fitness si peso_referencia=0.
    muestra_real = x_train[y_train == configuracion.digito_objetivo][0]
    cv2.imwrite(str(carpeta / "referencia_real_mnist.png"), muestra_real)

    print("\nMejor pseudodígito")
    print("Dígito objetivo:", configuracion.digito_objetivo)
    print("Clase predicha:", mejor.metricas["clase_predicha"])
    print("P(dígito objetivo):", mejor.metricas["prob_objetivo"])
    print("Fitness (menor es mejor):", mejor.fitness.values[0])
    print("Nodos totales:", contar_nodos(mejor))
    print("Distribución de probabilidades:")
    for clase, prob in zip(clasificador.classes_, mejor.metricas["probabilidades"]):
        print(f"  P({clase}) = {prob:.6f}")
    print("\nLas 14 funciones evolucionadas:")
    for nombre, arbol in zip(motor.NOMBRES_ARB, mejor):
        print(f"  {nombre} = {arbol}")
    print("\nResultados en:", carpeta.resolve())
    print("Tiempo total:", round(time.perf_counter() - inicio, 2), "s")


if __name__ == "__main__":
    main()
