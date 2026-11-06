# -*- coding: utf-8 -*-
"""
Metodo:
    - DWT con wavelet Daubechies 4 (db4).
    - Extension simetrica de los extremos.
    - Exclusion de coeficientes afectados por los bordes.
    - Varianza de coeficientes de detalle.
    - Regresion lineal de log2(Var(d_j)) respecto a j.

Relaciones teoricas:

    fBm:
        Var(d_j) ~ 2**(j*(2H + 1))
        H_estimado = (m - 1)/2

    fGn:
        Var(d_j) ~ 2**(j*(2H - 1))
        H_estimado = (m + 1)/2
@author: Ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import fGn

from pathlib import Path


Hs = np.round(np.arange(0.1, 1.0, 0.1), 1)

Ns = [1024, 4096, 16384, 65536]

M = 300

WAVELET = "db4"


J_MIN = 2
J_MAX = 5

# Extension de los extremos.
MODO_WAVELET = "symmetric"

SEMILLA_BASE = 1337

# Inferencia estadistica

B_BOOT = 10000
ALPHA = 0.05

rng_bootstrap = np.random.default_rng(2026)

CARPETA_SALIDA = Path("wavelet")

CARPETA_GRAFICAS = CARPETA_SALIDA / "graficas"

CARPETA_GRAFICAS.mkdir(
    parents=True,
    exist_ok=True
)

# Ejemplo individual

H_EJEMPLO = 0.75
N_EJEMPLO = 65536
SEED_EJEMPLO = 30


def recortar_bordes(
    coeficientes,
    wavelet,
    escala
):
    """
    Retorna:
        Coeficientes interiores.
    """

    coeficientes = np.asarray(
        coeficientes,
        dtype=float
    )

    L = wavelet.dec_len

    soporte_efectivo = (
        (L - 1) * (2**escala - 1) + 1
    )

    n_recorte = int(
        np.ceil(
            (soporte_efectivo - 1) / (2**escala)
        )
    ) + 1

    if len(coeficientes) <= 2 * n_recorte:

        return np.array([], dtype=float)

    return coeficientes[
        n_recorte:-n_recorte
    ]


def analizar_wavelet(
    senal,
    wavelet="db4",
    j_min=2,
    j_max=5,
    tipo="fBm"
):
    """
    Retorna:
        - Pendiente.
        - Intercepto.
        - R2.
        - H estimado.
        - Varianzas por escala.
        - Residuos de la regresion.
        - RMSE de los residuos.
        - Residuo maximo absoluto.
    """

    senal = np.asarray(
        senal,
        dtype=float
    )

    wavelet_obj = pywt.Wavelet(wavelet)


    nivel_maximo = pywt.dwt_max_level(
        len(senal),
        wavelet_obj.dec_len
    )

    if j_max > nivel_maximo:

        raise ValueError(
            f"El nivel solicitado ({j_max}) supera "
            f"el nivel maximo recomendado ({nivel_maximo})."
        )

    descomposicion = pywt.wavedec(
        senal,
        wavelet=wavelet_obj,
        mode=MODO_WAVELET,
        level=j_max
    )


    detalles = descomposicion[1:][::-1]

    escalas = np.arange(
        1,
        len(detalles) + 1
    )

    varianzas = []
    numero_coeficientes = []

    for j, detalle in enumerate(
        detalles,
        start=1
    ):

        detalle_interior = recortar_bordes(
            detalle,
            wavelet_obj,
            j
        )

        numero_coeficientes.append(
            len(detalle_interior)
        )

        if len(detalle_interior) > 1:

            varianza = np.var(
                detalle_interior,
                ddof=1
            )

        else:

            varianza = np.nan

        varianzas.append(varianza)

    varianzas = np.asarray(
        varianzas,
        dtype=float
    )

    numero_coeficientes = np.asarray(
        numero_coeficientes,
        dtype=int
    )


    mascara = (
        (escalas >= j_min)
        & (escalas <= j_max)
        & np.isfinite(varianzas)
        & (varianzas > 0)
    )

    j_fit = escalas[mascara]

    log_var = np.log2(
        varianzas[mascara]
    )

    if len(j_fit) < 2:

        raise ValueError(
            "No hay suficientes escalas validas para "
            "la regresion."
        )

    m, b = np.polyfit(
        j_fit,
        log_var,
        1
    )

    log_var_pred = (
        m * j_fit + b
    )

    ss_res = np.sum(
        (log_var - log_var_pred)**2
    )

    ss_tot = np.sum(
        (log_var - np.mean(log_var))**2
    )

    if ss_tot > 0:

        R2 = 1 - ss_res / ss_tot

    else:

        R2 = np.nan

 
    residuos = (
        log_var - log_var_pred
    )

    RMSE_residuos = np.sqrt(
        np.mean(residuos**2)
    )

    residuo_max_abs = np.max(
        np.abs(residuos)
    )

    if tipo == "fBm":

        H_estimado = (m - 1) / 2

    elif tipo == "fGn":

        H_estimado = (m + 1) / 2

    else:

        raise ValueError(
            "tipo debe ser 'fBm' o 'fGn'."
        )

    return {

        "pendiente": m,

        "intercepto": b,

        "R2": R2,

        "H_estimado": H_estimado,

        "escalas": escalas,

        "varianzas": varianzas,

        "numero_coeficientes": numero_coeficientes,

        "j_fit": j_fit,

        "log_var_fit": log_var,

        "log_var_pred": log_var_pred,

        "residuos": residuos,

        "RMSE_residuos": RMSE_residuos,

        "residuo_max_abs": residuo_max_abs

    }


resultados = []

errores = []

total_experimentos = (
    len(Hs) * len(Ns) * M
)

contador = 0

print("=" * 75)
print("EXPERIMENTO MONTE CARLO: ESTIMACION WAVELET")
print("=" * 75)

print(f"Wavelet: {WAVELET}")
print(f"Modo de extension: {MODO_WAVELET}")
print(f"Valores de H: {len(Hs)}")
print(f"Longitudes N: {Ns}")
print(f"Realizaciones por combinacion: {M}")
print(f"Escalas de ajuste: {J_MIN} a {J_MAX}")
print(f"Experimentos por proceso: {total_experimentos}")
print(f"Total de estimaciones esperadas: {2 * total_experimentos}")


for H in Hs:

    for N in Ns:

        print("\n" + "-" * 65)
        print(f"H = {H:.1f} | N = {N}")
        print("-" * 65)

        for r in range(M):

            contador += 1

            seed = (
                SEMILLA_BASE
                + int(H * 1000) * 100000
                + N * 10
                + r
            )

            try:

               
                B = fGn.fBm(
                    N,
                    H,
                    seed=seed
                )

                B = np.asarray(
                    B,
                    dtype=float
                )

                # Incrementos del fBm: fGn

                X = np.diff(B)

                # ------------------------------------------------
                # Analizar fBm
                # ------------------------------------------------

                resultado_bm = analizar_wavelet(
                    B,
                    wavelet=WAVELET,
                    j_min=J_MIN,
                    j_max=J_MAX,
                    tipo="fBm"
                )

                resultados.append({

                    "proceso": "fBm",

                    "H_teorico": H,

                    "N": N,

                    "realizacion": r + 1,

                    "pendiente": resultado_bm["pendiente"],

                    "H_estimado": resultado_bm["H_estimado"],

                    "R2": resultado_bm["R2"],

                    "RMSE_residuos": resultado_bm["RMSE_residuos"],

                    "residuo_max_abs": resultado_bm["residuo_max_abs"]

                })

                # ------------------------------------------------
                # Analizar fGn
                # ------------------------------------------------

                resultado_gn = analizar_wavelet(
                    X,
                    wavelet=WAVELET,
                    j_min=J_MIN,
                    j_max=J_MAX,
                    tipo="fGn"
                )

                resultados.append({

                    "proceso": "fGn",

                    "H_teorico": H,

                    "N": N,

                    "realizacion": r + 1,

                    "pendiente": resultado_gn["pendiente"],

                    "H_estimado": resultado_gn["H_estimado"],

                    "R2": resultado_gn["R2"],

                    "RMSE_residuos": resultado_gn["RMSE_residuos"],

                    "residuo_max_abs": resultado_gn["residuo_max_abs"]

                })

            except Exception as e:

                errores.append({

                    "H": H,

                    "N": N,

                    "realizacion": r + 1,

                    "error": str(e)

                })

            if (r + 1) % 50 == 0:

                print(
                    f"Realizaciones: {r + 1}/{M} | "
                    f"Progreso global: "
                    f"{100 * contador / total_experimentos:.1f}%"
                )


df = pd.DataFrame(resultados)

if df.empty:

    raise RuntimeError(
        "No se obtuvieron resultados. Revisa la generacion "
        "de señales y los parametros wavelet."
    )

archivo_individual = (
    CARPETA_SALIDA / "resultados_individuales_wavelet.csv"
)

df.to_csv(
    archivo_individual,
    index=False,
    encoding="utf-8-sig"
)

if errores:

    pd.DataFrame(errores).to_csv(
        CARPETA_SALIDA / "errores_wavelet.csv",
        index=False,
        encoding="utf-8-sig"
    )


def bootstrap_inferencia(
    valores,
    valor_nulo,
    B=10000,
    alpha=0.05,
    rng=None
):
    """
    Calcula:

    1. Intervalo bootstrap percentil para la media.
    2. Prueba bootstrap bilateral centrada bajo H0.

    Hipotesis:

        H0: E[H_estimado] = H
        H1: E[H_estimado] != H
    """

    valores = np.asarray(
        valores,
        dtype=float
    )

    valores = valores[
        np.isfinite(valores)
    ]

    n = len(valores)

    if n < 2:

        raise ValueError(
            "Se requieren al menos dos estimaciones validas."
        )

    if rng is None:

        rng = np.random.default_rng()

    media_observada = np.mean(
        valores
    )

    # --------------------------------------------------------
    # Intervalo bootstrap percentil para la media
    # --------------------------------------------------------

    indices = rng.integers(
        0,
        n,
        size=(B, n)
    )

    medias_boot = valores[
        indices
    ].mean(axis=1)

    ic_inf, ic_sup = np.quantile(
        medias_boot,
        [alpha / 2, 1 - alpha / 2]
    )

    # --------------------------------------------------------
    # Prueba bootstrap bilateral bajo H0
    # --------------------------------------------------------

    muestra_nula = (
        valores - media_observada + valor_nulo
    )

    indices_nulos = rng.integers(
        0,
        n,
        size=(B, n)
    )

    medias_nulas = muestra_nula[
        indices_nulos
    ].mean(axis=1)

    estadistico_observado = abs(
        media_observada - valor_nulo
    )

    estadisticos_boot = abs(
        medias_nulas - valor_nulo
    )

    p_valor = (
        1 + np.sum(
            estadisticos_boot >= estadistico_observado
        )
    ) / (B + 1)

    return {

        "IC95_inf": ic_inf,

        "IC95_sup": ic_sup,

        "p_valor": p_valor

    }


def correccion_holm(
    p_valores,
    alpha=0.05
):
    """
    Correccion de Holm-Bonferroni.
    """

    p_valores = np.asarray(
        p_valores,
        dtype=float
    )

    m = len(p_valores)

    orden = np.argsort(
        p_valores
    )

    p_ordenados = p_valores[
        orden
    ]

    ajustados = np.maximum.accumulate(
        (m - np.arange(m)) * p_ordenados
    )

    ajustados = np.minimum(
        ajustados,
        1.0
    )

    p_holm = np.empty(m)

    p_holm[orden] = ajustados

    rechazar = (
        p_holm <= alpha
    )

    return p_holm, rechazar


def calcular_rmse(
    valores,
    H
):

    return np.sqrt(
        np.mean(
            (valores - H)**2
        )
    )


resumen = []

for (proceso, H, N), grupo in df.groupby(
    ["proceso", "H_teorico", "N"]
):

    estimaciones = grupo[
        "H_estimado"
    ].values

    media = np.mean(
        estimaciones
    )

    desviacion = np.std(
        estimaciones,
        ddof=1
    )

    sesgo = media - H

    rmse = calcular_rmse(
        estimaciones,
        H
    )

    resumen.append({

        "proceso": proceso,

        "H_teorico": H,

        "N": N,

        "n_realizaciones": len(estimaciones),

        "H_media": media,

        "desviacion_estandar": desviacion,

        "sesgo": sesgo,

        "RMSE": rmse,

        "R2_medio": grupo["R2"].mean(),

        "pendiente_media": grupo["pendiente"].mean(),

        "RMSE_residuos_medio": grupo["RMSE_residuos"].mean(),

        "residuo_max_abs_medio": grupo["residuo_max_abs"].mean()

    })

df_resumen = pd.DataFrame(
    resumen
)


resultados_inferencia = []

for (proceso, H, N), grupo in df.groupby(
    ["proceso", "H_teorico", "N"]
):

    estimaciones = grupo[
        "H_estimado"
    ].values

    inferencia = bootstrap_inferencia(
        estimaciones,
        valor_nulo=H,
        B=B_BOOT,
        alpha=ALPHA,
        rng=rng_bootstrap
    )

    resultados_inferencia.append({

        "proceso": proceso,

        "H_teorico": H,

        "N": N,

        **inferencia

    })

df_inferencia = pd.DataFrame(
    resultados_inferencia
)

df_resumen = df_resumen.merge(
    df_inferencia,
    on=["proceso", "H_teorico", "N"],
    how="left"
)



# Se aplica conjuntamente a las 72 pruebas:
# 36 combinaciones de fBm y 36 de fGn.

p_holm, rechazar = correccion_holm(
    df_resumen["p_valor"].values,
    alpha=ALPHA
)

df_resumen["p_valor_Holm"] = p_holm

df_resumen["rechazar_H0"] = rechazar



archivo_resumen = (
    CARPETA_SALIDA / "resumen_wavelet.csv"
)

df_resumen.to_csv(
    archivo_resumen,
    index=False,
    encoding="utf-8-sig"
)

archivo_inferencia = (
    CARPETA_SALIDA / "inferencia_wavelet.csv"
)

df_inferencia_final = df_resumen[

    [
        "proceso",
        "H_teorico",
        "N",
        "IC95_inf",
        "IC95_sup",
        "p_valor",
        "p_valor_Holm",
        "rechazar_H0"
    ]

].copy()

df_inferencia_final.to_csv(
    archivo_inferencia,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 75)
print("RESUMEN DEL EXPERIMENTO")
print("=" * 75)

print(
    df_resumen.groupby("proceso")[
        [
            "H_media",
            "desviacion_estandar",
            "sesgo",
            "RMSE",
            "R2_medio",
            "RMSE_residuos_medio"
        ]
    ].mean().round(4)
)

print("\nRESULTADOS DE INFERENCIA")

for proceso in ["fBm", "fGn"]:

    datos_proceso = df_resumen[
        df_resumen["proceso"] == proceso
    ]

    rechazadas = datos_proceso[
        "rechazar_H0"
    ].sum()

    total = len(
        datos_proceso
    )

    print(
        f"{proceso}: {rechazadas} de {total} "
        "hipotesis rechazadas."
    )

print(
    "\nTotal de hipotesis rechazadas:",
    df_resumen["rechazar_H0"].sum(),
    "de",
    len(df_resumen)
)

print("\nArchivos guardados:")

print(archivo_individual)
print(archivo_resumen)
print(archivo_inferencia)

print(f"\nEstimaciones obtenidas: {len(df)}")

if errores:

    print(
        f"Realizaciones con error: {len(errores)}"
    )

else:

    print("No se registraron errores.")


def generar_graficas_proceso(
    df,
    df_resumen,
    proceso,
    carpeta
):

    datos = df[
        df["proceso"] == proceso
    ].copy()

    resumen_proceso = df_resumen[
        df_resumen["proceso"] == proceso
    ].copy()

    if datos.empty:
        return
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for N in Ns:

        grupo = (
            resumen_proceso[
                resumen_proceso["N"] == N
            ]
            .sort_values("H_teorico")
        )

        ax.plot(
            grupo["H_teorico"],
            grupo["H_media"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"N = {N}"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.5,
        label="Relacion ideal"
    )

    ax.set_xlabel(r"$H$ teorico")
    ax.set_ylabel(r"$\widehat{H}$ estimado")

    ax.set_title(
        f"Estimacion wavelet de H - {proceso}"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        carpeta / f"H_estimado_vs_H_{proceso}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for N in Ns:

        grupo = (
            resumen_proceso[
                resumen_proceso["N"] == N
            ]
            .sort_values("H_teorico")
        )

        ax.plot(
            grupo["H_teorico"],
            grupo["desviacion_estandar"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"N = {N}"
        )

    ax.set_xlabel(r"$H$ teorico")
    ax.set_ylabel("Desviacion estandar")

    ax.set_title(
        f"Dispersion de las estimaciones wavelet - {proceso}"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        carpeta / f"desviacion_estandar_vs_H_{proceso}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for N in Ns:

        grupo = (
            resumen_proceso[
                resumen_proceso["N"] == N
            ]
            .sort_values("H_teorico")
        )

        ax.plot(
            grupo["H_teorico"],
            grupo["RMSE"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"N = {N}"
        )

    ax.set_xlabel(r"$H$ teorico")
    ax.set_ylabel("RMSE")

    ax.set_title(
        f"Error de estimacion wavelet - {proceso}"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        carpeta / f"RMSE_vs_H_{proceso}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, N in zip(axes, Ns):

        datos_N = datos[
            datos["N"] == N
        ]

        datos_boxplot = [

            datos_N[
                datos_N["H_teorico"] == H
            ]["H_estimado"].values

            for H in Hs

        ]

        ax.boxplot(
            datos_boxplot,
            tick_labels=[
                f"{H:.1f}" for H in Hs
            ],
            showfliers=False
        )

        ax.plot(
            np.arange(1, len(Hs) + 1),
            Hs,
            "r--o",
            markersize=4,
            linewidth=1.5,
            label="H teorico"
        )

        ax.set_title(
            f"N = {N}"
        )

        ax.set_xlabel(
            r"$H$ teorico"
        )

        ax.set_ylabel(
            r"$\widehat{H}$ estimado"
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        ax.legend()

    fig.suptitle(
        f"Distribucion de las estimaciones wavelet - {proceso}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        carpeta / f"boxplot_wavelet_2x2_{proceso}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for N in Ns:

        grupo = (
            resumen_proceso[
                resumen_proceso["N"] == N
            ]
            .sort_values("H_teorico")
        )

        ax.plot(
            grupo["H_teorico"],
            grupo["RMSE_residuos_medio"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"N = {N}"
        )

    ax.set_xlabel(
        r"$H$ teorico"
    )

    ax.set_ylabel(
        "RMSE de los residuos"
    )

    ax.set_title(
        f"Residuos del ajuste wavelet - {proceso}"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        carpeta / f"RMSE_residuos_vs_H_{proceso}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

for proceso in ["fBm", "fGn"]:

    generar_graficas_proceso(
        df,
        df_resumen,
        proceso,
        CARPETA_GRAFICAS
    )


B = np.asarray(
    fGn.fBm(
        N_EJEMPLO,
        H_EJEMPLO,
        seed=SEED_EJEMPLO
    ),
    dtype=float
)

X = np.diff(B)

# Se utiliza el mismo intervalo de escalas del Monte Carlo.

resultado_fBm = analizar_wavelet(
    B,
    wavelet=WAVELET,
    j_min=J_MIN,
    j_max=J_MAX,
    tipo="fBm"
)

resultado_fGn = analizar_wavelet(
    X,
    wavelet=WAVELET,
    j_min=J_MIN,
    j_max=J_MAX,
    tipo="fGn"
)


print("\n" + "=" * 65)
print("EJEMPLO INDIVIDUAL")
print("=" * 65)

print(f"H teorico: {H_EJEMPLO}")
print(f"N: {N_EJEMPLO}")
print(f"Semilla: {SEED_EJEMPLO}")
print(f"Wavelet: {WAVELET}")
print(f"Modo: {MODO_WAVELET}")
print(f"Escalas: {J_MIN} a {J_MAX}")

for proceso, resultado in [

    ("fBm", resultado_fBm),

    ("fGn", resultado_fGn)

]:

    print("\n", proceso)

    print(
        f"Pendiente: {resultado['pendiente']:.4f}"
    )

    print(
        f"H estimado: {resultado['H_estimado']:.4f}"
    )

    print(
        f"R2: {resultado['R2']:.4f}"
    )

    print(
        f"RMSE residuos: {resultado['RMSE_residuos']:.6f}"
    )

    print(
        f"Residuo maximo absoluto: "
        f"{resultado['residuo_max_abs']:.6f}"
    )

    print(
        "Coeficientes interiores por escala:",
        resultado["numero_coeficientes"]
    )



fig, ax = plt.subplots(
    2,
    2,
    figsize=(14, 9)
)

#Trayectoria fBm
ax[0, 0].plot(
    B,
    linewidth=0.8
)

ax[0, 0].set_title(
    f"fBm (H = {H_EJEMPLO})"
)

ax[0, 0].set_xlabel("Tiempo")
ax[0, 0].set_ylabel(r"$B_H(t)$")
ax[0, 0].grid(alpha=0.3)


#Trayectoria fGn
ax[0, 1].plot(
    X,
    linewidth=0.6
)

ax[0, 1].set_title(
    f"fGn (H = {H_EJEMPLO})"
)

ax[0, 1].set_xlabel("Tiempo")
ax[0, 1].set_ylabel(r"$X(t)$")
ax[0, 1].grid(alpha=0.3)


#  Escalamiento wavelet fBm

r = resultado_fBm

ax[1, 0].plot(
    r["escalas"],
    np.log2(r["varianzas"]),
    "o-",
    label="Varianza wavelet"
)

ax[1, 0].plot(
    r["j_fit"],
    r["log_var_pred"],
    "--",
    label=(
        f"Pendiente = {r['pendiente']:.3f}\n"
        f"H estimado = {r['H_estimado']:.3f}"
    )
)

ax[1, 0].axvspan(
    J_MIN,
    J_MAX,
    alpha=0.10,
    label="Intervalo de ajuste"
)

ax[1, 0].set_title(
    "Escalamiento wavelet del fBm"
)

ax[1, 0].set_xlabel(
    r"Escala $j$"
)

ax[1, 0].set_ylabel(
    r"$\log_2[\operatorname{Var}(d_j)]$"
)

ax[1, 0].legend()
ax[1, 0].grid(alpha=0.3)


# Escalamiento wavelet fGn
r = resultado_fGn

ax[1, 1].plot(
    r["escalas"],
    np.log2(r["varianzas"]),
    "o-",
    label="Varianza wavelet"
)

ax[1, 1].plot(
    r["j_fit"],
    r["log_var_pred"],
    "--",
    label=(
        f"Pendiente = {r['pendiente']:.3f}\n"
        f"H estimado = {r['H_estimado']:.3f}"
    )
)

ax[1, 1].axvspan(
    J_MIN,
    J_MAX,
    alpha=0.10,
    label="Intervalo de ajuste"
)

ax[1, 1].set_title(
    "Escalamiento wavelet del fGn"
)

ax[1, 1].set_xlabel(
    r"Escala $j$"
)

ax[1, 1].set_ylabel(
    r"$\log_2[\operatorname{Var}(d_j)]$"
)

ax[1, 1].legend()
ax[1, 1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "ejemplo_escalamiento_wavelet.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#ANALISIS DE RESIDUOS DEL EJEMPLO INDIVIDUAL
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4)
)

for ax, resultado, proceso in [

    (axes[0], resultado_fBm, "fBm"),

    (axes[1], resultado_fGn, "fGn")

]:

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.plot(
        resultado["j_fit"],
        resultado["residuos"],
        "o-",
        linewidth=1.2,
        markersize=5
    )

    ax.set_xlabel(
        r"Escala $j$"
    )

    ax.set_ylabel(
        "Residuo"
    )

    ax.set_title(
        f"Residuos del ajuste wavelet - {proceso}"
    )

    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "residuos_ejemplo_wavelet.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

