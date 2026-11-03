# -*- coding: utf-8 -*-
"""
Created on Tue Oct  12 13:45:44 2026

@author: Ricar

EXPERIMENTO MONTE CARLO: RANGO REESCALADO (R/S) PARA fGn

1. Genera directamente realizaciones de fGn.
2. Aplica el análisis R/S sobre cada realización.
3. Estima H mediante el ajuste log(R/S) vs log(n).
4. Repite 300 realizaciones por combinación H-N.
5. Calcula sesgo, MAE, RMSE, desviación estándar y R².
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fGn



Hs = np.round(np.arange(0.1, 1.0, 0.1), 1)

Ns = [1024, 4096, 16384, 65536]

M = 300

SEMILLA_BASE = 1337

CARPETA_SALIDA = Path("resultados_RS_fGn")
CARPETA_GRAFICAS = CARPETA_SALIDA / "graficas"

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

N_ESCALAS = 30 # Número de escalas para el ajuste
N_MIN = 8 # Escala mínima
FACTOR_ESCALA_MAX = 10 # Escala máxima: N/10


def generar_fgn(N, H, semilla):

    X = np.asarray(
        fGn.fGn(N, H, seed=int(semilla)),
        dtype=float
    ).ravel()

    if len(X) == N + 1:
        X = X[:N]

    if len(X) != N:
        raise ValueError(
            f"Longitud inesperada: {len(X)}. "
            f"Se esperaba {N}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "La serie contiene valores no finitos."
        )

    return X

def obtener_escalas(N):

    n_max = N // FACTOR_ESCALA_MAX

    if n_max <= N_MIN:
        raise ValueError(
            "La serie es demasiado corta."
        )

    escalas = np.unique(
        np.logspace(
            np.log10(N_MIN),
            np.log10(n_max),
            N_ESCALAS
        ).astype(int)
    )

    return escalas

def calcular_RS(X, n):

    """
    Divide la serie en segmentos no superpuestos.

    Para cada segmento:

    1. Calcula su media.
    2. Resta la media a cada observación.
    3. Calcula las sumas acumuladas.
    4. Obtiene el rango de las sumas acumuladas.
    5. Divide el rango entre la desviación estándar.

    Devuelve el promedio de R/S entre segmentos.
    """

    X = np.asarray(X, dtype=float)

    K = len(X) // n

    if K < 1:
        return np.nan

    # Descartar fragmento final incompleto
    X = X[:K * n]

    segmentos = X.reshape(K, n)

    medias = np.mean(
        segmentos,
        axis=1,
        keepdims=True
    )

    desviaciones = segmentos - medias

    acumuladas = np.cumsum(
        desviaciones,
        axis=1
    )

    R = (
        np.max(acumuladas, axis=1)
        - np.min(acumuladas, axis=1)
    )

    S = np.std(
        segmentos,
        axis=1,
        ddof=1
    )

    validos = (
        np.isfinite(R)
        & np.isfinite(S)
        & (S > 0)
    )

    if not np.any(validos):
        return np.nan

    valores_RS = R[validos] / S[validos]

    return np.mean(valores_RS)


#ESTIMACIÓN DE H MEDIANTE R/S
def estimar_h_RS(X):

    escalas = obtener_escalas(len(X))

    valores_RS = np.array([
        calcular_RS(X, n)
        for n in escalas
    ])

    validos = (
        np.isfinite(valores_RS)
        & (valores_RS > 0)
    )

    escalas_validas = escalas[validos]
    RS_validos = valores_RS[validos]

    if len(escalas_validas) < 5:
        raise ValueError(
            "No hay suficientes escalas válidas."
        )

    log_n = np.log(escalas_validas)
    log_RS = np.log(RS_validos)

    pendiente, intercepto = np.polyfit(
        log_n,
        log_RS,
        1
    )

    log_RS_ajuste = (
        pendiente * log_n + intercepto
    )

    # Coeficiente de determinación
    ss_res = np.sum(
        (log_RS - log_RS_ajuste) ** 2
    )

    ss_tot = np.sum(
        (log_RS - np.mean(log_RS)) ** 2
    )

    if ss_tot > 0:
        R2 = 1 - ss_res / ss_tot
    else:
        R2 = np.nan

    RS_ajuste = np.exp(log_RS_ajuste)

    return {
        "H_estimado": pendiente,
        "pendiente": pendiente,
        "intercepto": intercepto,
        "R2": R2,
        "escalas": escalas_validas,
        "RS": RS_validos,
        "RS_ajuste": RS_ajuste
    }


#EXPERIMENTO MONTE CARLO
rng = np.random.default_rng(SEMILLA_BASE)

resultados = []

ejemplos = {}

total = len(Hs) * len(Ns) * M
contador = 0

print("=" * 75)
print("EXPERIMENTO MONTE CARLO: R/S DIRECTAMENTE SOBRE fGn")
print("=" * 75)

for H in Hs:

    for N in Ns:

        print("\n" + "-" * 65)
        print(f"H = {H:.1f} | N = {N}")
        print("-" * 65)

        estimaciones = []
        R2_valores = []

        for m in range(M):

            contador += 1

            semilla = int(
                rng.integers(0, 2**31 - 1)
            )

            try:

                # Generar directamente fGn
                X = generar_fgn(
                    N,
                    H,
                    semilla
                )

                # Estimar H mediante R/S
                resultado = estimar_h_RS(X)

                H_estimado = resultado["H_estimado"]
                R2 = resultado["R2"]

                estimaciones.append(H_estimado)
                R2_valores.append(R2)

                resultados.append({

                    "H_teorico": H,
                    "N": N,
                    "realizacion": m + 1,
                    "semilla": semilla,
                    "H_estimado": H_estimado,
                    "pendiente": resultado["pendiente"],
                    "R2": R2,
                    "error": H_estimado - H,
                    "error_absoluto": abs(
                        H_estimado - H
                    )

                })

                # Guardar una realización representativa
                if N == max(Ns) and m == 0:

                    ejemplos[H] = {

                        "escalas": resultado["escalas"],
                        "RS": resultado["RS"],
                        "RS_ajuste": resultado["RS_ajuste"],
                        "H_estimado": H_estimado,
                        "R2": R2

                    }

            except Exception as e:

                print(
                    f"Error en H={H}, N={N}, "
                    f"realización={m + 1}: {e}"
                )

        estimaciones = np.asarray(estimaciones)
        R2_valores = np.asarray(R2_valores)

        n_validos = len(estimaciones)

        if n_validos == 0:
            continue

        H_media = np.mean(estimaciones)

        H_std = np.std(
            estimaciones,
            ddof=1
        )

        sesgo = H_media - H

        MAE = np.mean(
            np.abs(estimaciones - H)
        )

        RMSE = np.sqrt(
            np.mean((estimaciones - H) ** 2)
        )

        error_estandar = H_std / np.sqrt(n_validos)

        IC95 = 1.96 * error_estandar

        R2_medio = np.nanmean(R2_valores)

        print(f"H teórico:             {H:.4f}")
        print(f"H medio estimado:      {H_media:.6f}")
        print(f"Desviación estándar:   {H_std:.6f}")
        print(f"Sesgo:                 {sesgo:.6f}")
        print(f"MAE:                   {MAE:.6f}")
        print(f"RMSE:                  {RMSE:.6f}")
        print(f"R2 medio:              {R2_medio:.6f}")
        print(f"IC 95% de la media:    +/- {IC95:.6f}")
        print(f"Realizaciones válidas: {n_validos}")


#GUARDAR RESULTADOS INDIVIDUALES
df_individual = pd.DataFrame(resultados)

ruta_individual = (
    CARPETA_SALIDA /
    "resultados_individuales_RS.csv"
)

df_individual.to_csv(
    ruta_individual,
    index=False
)


#RESUMEN ESTADÍSTICO
resumen = []

for H in Hs:

    for N in Ns:

        grupo = df_individual[
            (df_individual["H_teorico"] == H)
            & (df_individual["N"] == N)
        ]

        estimaciones = grupo["H_estimado"].values

        n_validos = len(estimaciones)

        if n_validos == 0:
            continue

        H_media = np.mean(estimaciones)

        H_std = np.std(
            estimaciones,
            ddof=1
        )

        sesgo = H_media - H

        MAE = np.mean(
            np.abs(estimaciones - H)
        )

        RMSE = np.sqrt(
            np.mean((estimaciones - H) ** 2)
        )

        error_estandar = H_std / np.sqrt(n_validos)

        IC95 = 1.96 * error_estandar

        R2_medio = grupo["R2"].mean()

        resumen.append({

            "H": H,
            "N": N,
            "H_media": H_media,
            "H_std": H_std,
            "sesgo": sesgo,
            "MAE": MAE,
            "RMSE": RMSE,
            "R2_medio": R2_medio,
            "realizaciones_validas": n_validos,
            "error_estandar": error_estandar,
            "IC95_inferior": H_media - IC95,
            "IC95_superior": H_media + IC95

        })

df_resumen = pd.DataFrame(resumen)

ruta_resumen = (
    CARPETA_SALIDA /
    "resumen_RS.csv"
)

df_resumen.to_csv(
    ruta_resumen,
    index=False
)


# PLOT H ESTIMADO VS H TEÓRICO
plt.figure(figsize=(8, 6))

for N in Ns:

    grupo = df_resumen[
        df_resumen["N"] == N
    ]

    plt.errorbar(
        grupo["H"],
        grupo["H_media"],
        yerr=1.96 * grupo["error_estandar"],
        marker="o",
        capsize=3,
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label=r"$\widehat{H}=H$"
)

plt.xlabel("H teórico")
plt.ylabel("H estimado")

plt.title("Estimación de H mediante R/S sobre fGn")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "H_estimado_vs_H_teorico.png",
    dpi=300
)

plt.close()


#SESGO
plt.figure(figsize=(8, 6))

for N in Ns:

    grupo = df_resumen[
        df_resumen["N"] == N
    ]

    plt.plot(
        grupo["H"],
        grupo["sesgo"],
        marker="o",
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("H teórico")
plt.ylabel("Sesgo")

plt.title("Sesgo de las estimaciones R/S")

plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "sesgo_vs_H.png",
    dpi=300
)

plt.close()


#DESVIACIÓN ESTÁNDAR
plt.figure(figsize=(8, 6))

for N in Ns:

    grupo = df_resumen[
        df_resumen["N"] == N
    ]

    plt.plot(
        grupo["H"],
        grupo["H_std"],
        marker="o",
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.xlabel("H teórico")
plt.ylabel("Desviación estándar")

plt.title("Variabilidad de las estimaciones R/S")

plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "desviacion_estandar_vs_H.png",
    dpi=300
)

plt.close()


#BOXPLOTS PARA CADA N
for N in Ns:

    datos = []

    for H in Hs:

        grupo = df_individual[
            (df_individual["H_teorico"] == H)
            & (df_individual["N"] == N)
        ]

        datos.append(
            grupo["H_estimado"].values
        )

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        datos,
        tick_labels=[f"{H:.1f}" for H in Hs],
        showfliers=False
    )

    plt.plot(
        range(1, len(Hs) + 1),
        Hs,
        "r--",
        linewidth=1.5,
        label="H teórico"
    )

    plt.xlabel("H teórico")
    plt.ylabel("H estimado")

    plt.title(
        f"Estimaciones R/S para N = {N}"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS /
        f"boxplot_RS_N{N}.png",
        dpi=300
    )

    plt.close()


#EJEMPLOS DEL AJUSTE LOG-LOG

plt.figure(figsize=(10, 7))

for H in Hs:

    if H not in ejemplos:
        continue

    datos = ejemplos[H]

    plt.plot(
        np.log10(datos["escalas"]),
        np.log10(datos["RS"]),
        "o",
        markersize=3,
        label=f"H={H:.1f}"
    )

    plt.plot(
        np.log10(datos["escalas"]),
        np.log10(datos["RS_ajuste"]),
        "--",
        linewidth=1
    )

plt.xlabel(r"$\log_{10}(n)$")
plt.ylabel(r"$\log_{10}(R/S)$")

plt.title(
    "Ajustes R/S para una realización de cada H "
    f"(N={max(Ns)})"
)

plt.grid(alpha=0.25)
plt.legend(ncol=3)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "ejemplos_RS_loglog.png",
    dpi=300
)

plt.close()


#GRÁFICA COMPARATIVA 2x2
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9)
)

for ax, N in zip(axes.flat, Ns):

    grupo = df_resumen[
        df_resumen["N"] == N
    ]

    ax.errorbar(
        grupo["H"],
        grupo["H_media"],
        yerr=1.96 * grupo["error_estandar"],
        marker="o",
        capsize=3,
        label="Estimación"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Teórico"
    )

    ax.set_title(f"N = {N}")

    ax.set_xlabel("H teórico")
    ax.set_ylabel("H estimado")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(alpha=0.25)
    ax.legend()

fig.suptitle(
    "Comparación entre H teórico y estimado mediante R/S",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS /
    "comparacion_H_teorico_estimado.png",
    dpi=300
)

plt.close()