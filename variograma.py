# -*- coding: utf-8 -*-
"""
Experimento Monte Carlo:
Estimacion del exponente de Hurst mediante variograma
(funcion de estructura de segundo orden) aplicado a fBm.

Configuracion:
    H = 0.1, 0.2, ..., 0.9
    N = 1024, 4096, 16384, 65536
    M = 300 realizaciones por combinacion

Analisis estadistico:
    - Media y desviacion estandar
    - Sesgo, MAE y RMSE
    - Intervalos de confianza bootstrap del 95 %
    - Prueba de hipotesis bootstrap bilateral
    - Correccion de Holm para comparaciones multiples
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fGn


Hs = np.round(np.arange(0.1, 1.0, 0.1), 1)

Ns = [2**10, 2**12, 2**14, 2**16]

M = 300

SEMILLA_BASE = 1337

# Numero de retardos utilizados
N_TAUS = 50

# Analisis estadistico
B_BOOT = 10000
ALPHA = 0.05

# Carpeta de resultados
CARPETA_SALIDA = Path("resultados_variograma_fBm")
CARPETA_GRAFICAS = CARPETA_SALIDA / "graficas"

CARPETA_SALIDA.mkdir(exist_ok=True)
CARPETA_GRAFICAS.mkdir(exist_ok=True)

rng = np.random.default_rng(SEMILLA_BASE)

total_realizaciones = len(Hs) * len(Ns) * M

semillas = rng.choice(
    np.arange(1, 2**31 - 1),
    size=total_realizaciones,
    replace=False
)

contador_semillas = 0


def generar_fbm(N, H, semilla):

    X = np.asarray(
        fGn.fBm(N, H, int(semilla)),
        dtype=float
    ).flatten()

    # Validar longitud
    if len(X) == N + 1:
        X = X[:N]

    if len(X) != N:
        raise ValueError(
            f"Longitud incorrecta: se esperaba {N}, "
            f"se obtuvo {len(X)}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "La trayectoria contiene valores no finitos."
        )

    return X


def obtener_taus(N, n_taus=50):

    taus = np.unique(
        np.logspace(
            np.log10(1),
            np.log10(N // 10),
            n_taus
        ).astype(int)
    )

    return taus


def funcion_estructura(X, taus):

    X = np.asarray(X, dtype=float)

    S2 = []

    for tau in taus:

        diferencias = X[tau:] - X[:-tau]

        valor = np.mean(diferencias**2)

        S2.append(valor)

    return np.asarray(S2)


def estimar_h_variograma(X, taus):

    S2 = funcion_estructura(X, taus)

    # Evitar valores nulos o no finitos
    mascara = (
        np.isfinite(S2) &
        (S2 > 0) &
        (taus > 0)
    )

    taus_validos = taus[mascara]
    S2_validos = S2[mascara]

    if len(taus_validos) < 3:
        return None

    log_tau = np.log(taus_validos)
    log_S2 = np.log(S2_validos)

    # Ajuste lineal
    pendiente, intercepto = np.polyfit(
        log_tau,
        log_S2,
        1
    )

    # H = pendiente / 2
    H_estimado = pendiente / 2.0

    # Coeficiente de determinacion R2
    predicciones = pendiente * log_tau + intercepto

    ss_res = np.sum(
        (log_S2 - predicciones)**2
    )

    ss_tot = np.sum(
        (log_S2 - np.mean(log_S2))**2
    )

    if ss_tot > 0:
        R2 = 1 - ss_res / ss_tot
    else:
        R2 = np.nan

    S2_ajuste = (
        np.exp(intercepto) *
        taus_validos**pendiente
    )

    return {
        "H_estimado": H_estimado,
        "pendiente": pendiente,
        "intercepto": intercepto,
        "R2": R2,
        "taus": taus_validos,
        "S2": S2_validos,
        "S2_ajuste": S2_ajuste
    }


def intervalo_bootstrap_media(
    datos,
    B=10000,
    alpha=0.05,
    rng=None
):
    """
    Intervalo de confianza percentil bootstrap para la media.
    """

    datos = np.asarray(datos, dtype=float)

    n = len(datos)

    if n == 0:
        return np.nan, np.nan

    if rng is None:
        rng = np.random.default_rng()

    indices = rng.integers(
        0,
        n,
        size=(B, n)
    )

    medias_bootstrap = np.mean(
        datos[indices],
        axis=1
    )

    inferior = np.quantile(
        medias_bootstrap,
        alpha / 2
    )

    superior = np.quantile(
        medias_bootstrap,
        1 - alpha / 2
    )

    return inferior, superior


def prueba_bootstrap_media(
    datos,
    valor_nulo,
    B=10000,
    rng=None
):
    """
    Prueba bootstrap bilateral:

        H0: E[H_estimado] = valor_nulo
        H1: E[H_estimado] != valor_nulo

    Se centra la muestra bajo H0.
    """

    datos = np.asarray(datos, dtype=float)

    n = len(datos)

    if n == 0:
        return np.nan

    if rng is None:
        rng = np.random.default_rng()

    media_observada = np.mean(datos)

    estadistico_observado = (
        media_observada - valor_nulo
    )

    # Construir muestra bajo H0
    muestra_nula = (
        datos - media_observada + valor_nulo
    )

    indices = rng.integers(
        0,
        n,
        size=(B, n)
    )

    medias_bootstrap = np.mean(
        muestra_nula[indices],
        axis=1
    )

    estadisticos_bootstrap = (
        medias_bootstrap - valor_nulo
    )

    # Prueba bilateral
    p_valor = (
        1 + np.sum(
            np.abs(estadisticos_bootstrap) >=
            abs(estadistico_observado)
        )
    ) / (B + 1)

    return p_valor


def correccion_holm(p_valores, alpha=0.05):
    """
    Correccion de Holm-Bonferroni.

    Devuelve:
        p_valores ajustados
        decisiones de rechazo
    """

    p_valores = np.asarray(
        p_valores,
        dtype=float
    )

    m = len(p_valores)

    orden = np.argsort(p_valores)

    p_ordenados = p_valores[orden]

    ajustados_ordenados = np.zeros(m)

    for i in range(m):

        ajustados_ordenados[i] = (
            (m - i) * p_ordenados[i]
        )

    # Garantizar monotonicidad
    ajustados_ordenados = np.maximum.accumulate(
        ajustados_ordenados
    )

    ajustados_ordenados = np.minimum(
        ajustados_ordenados,
        1.0
    )

    # Recuperar orden original
    p_ajustados = np.zeros(m)

    p_ajustados[orden] = ajustados_ordenados

    rechazar = p_ajustados < alpha

    return p_ajustados, rechazar


#EXPERIMENTO MONTE CARLO

resultados = []

# Guardar ejemplos para las graficas
ejemplos = {}

print("=" * 75)
print("EXPERIMENTO MONTE CARLO: VARIOGRAMA")
print("=" * 75)

print(f"Valores de H: {Hs}")
print(f"Tamanos N: {Ns}")
print(f"Realizaciones por combinacion: {M}")
print(f"Total de realizaciones: {total_realizaciones}")
print(f"Bootstrap: {B_BOOT} remuestreos")
print(f"Nivel de significancia: {ALPHA}")

for H in Hs:

    for N in Ns:

        taus = obtener_taus(
            N,
            N_TAUS
        )

        Hs_estimados = []
        pendientes = []
        R2s = []

        ejemplo_guardado = False

        for m in range(M):

            semilla = semillas[contador_semillas]
            contador_semillas += 1

            try:

                X = generar_fbm(
                    N,
                    H,
                    semilla
                )

                resultado = estimar_h_variograma(
                    X,
                    taus
                )

                if resultado is None:
                    continue

                H_est = resultado["H_estimado"]
                beta = resultado["pendiente"]
                R2 = resultado["R2"]

                if not np.isfinite(H_est):
                    continue

                Hs_estimados.append(H_est)
                pendientes.append(beta)
                R2s.append(R2)

                resultados.append({
                    "H_teorico": H,
                    "N": N,
                    "realizacion": m + 1,
                    "semilla": int(semilla),
                    "H_estimado": H_est,
                    "pendiente": beta,
                    "R2": R2,
                    "error": H_est - H,
                    "error_absoluto": abs(H_est - H)
                })

                # Guardar primera realizacion para las graficas
                if not ejemplo_guardado:

                    ejemplos[(H, N)] = {
                        "X": X.copy(),
                        "taus": resultado["taus"].copy(),
                        "S2": resultado["S2"].copy(),
                        "S2_ajuste": resultado["S2_ajuste"].copy(),
                        "pendiente": beta,
                        "H_estimado": H_est,
                        "R2": R2
                    }

                    ejemplo_guardado = True

            except Exception as error:

                print(
                    f"Error en H={H}, N={N}, "
                    f"realizacion={m + 1}: {error}"
                )

        #Resumen descriptivo de la combinacion

        Hs_estimados = np.asarray(
            Hs_estimados
        )

        if len(Hs_estimados) == 0:

            print(
                f"Sin resultados validos: H={H}, N={N}"
            )

            continue

        error = Hs_estimados - H

        media = np.mean(Hs_estimados)

        desviacion = np.std(
            Hs_estimados,
            ddof=1
        )

        error_estandar = (
            desviacion /
            np.sqrt(len(Hs_estimados))
        )

        sesgo = np.mean(error)

        MAE = np.mean(
            np.abs(error)
        )

        RMSE = np.sqrt(
            np.mean(error**2)
        )

        R2_medio = np.nanmean(R2s)

        #Intervalo de confianza bootstrap

        IC95_inferior, IC95_superior = (
            intervalo_bootstrap_media(
                Hs_estimados,
                B=B_BOOT,
                alpha=ALPHA,
                rng=rng
            )
        )

        # Prueba de hipotesis bootstrap

        p_valor = prueba_bootstrap_media(
            Hs_estimados,
            valor_nulo=H,
            B=B_BOOT,
            rng=rng
        )


        resumen_actual = {
            "H": H,
            "N": N,
            "H_media": media,
            "H_std": desviacion,
            "sesgo": sesgo,
            "MAE": MAE,
            "RMSE": RMSE,
            "R2_medio": R2_medio,
            "realizaciones_validas": len(Hs_estimados),
            "error_estandar": error_estandar,
            "IC95_inferior": IC95_inferior,
            "IC95_superior": IC95_superior,
            "p_valor": p_valor
        }


        print("\n" + "-" * 75)
        print(f"H = {H:.1f} | N = {N}")
        print("-" * 75)

        print(f"H teorico:             {H:.4f}")
        print(f"H medio estimado:      {media:.6f}")
        print(f"Desviacion estandar:   {desviacion:.6f}")
        print(f"Error estandar:        {error_estandar:.6f}")

        print(
            f"IC bootstrap 95%:      "
            f"[{IC95_inferior:.6f}, {IC95_superior:.6f}]"
        )

        print(f"Sesgo:                 {sesgo:.6f}")
        print(f"MAE:                   {MAE:.6f}")
        print(f"RMSE:                  {RMSE:.6f}")
        print(f"R2 medio:              {R2_medio:.6f}")
        print(f"p-valor bootstrap:     {p_valor:.6f}")

        print(
            f"Realizaciones validas: {len(Hs_estimados)}"
        )



df = pd.DataFrame(resultados)

if df.empty:
    raise RuntimeError(
        "No se obtuvieron resultados validos."
    )

archivo_individual = (
    CARPETA_SALIDA /
    "resultados_individuales_variograma.csv"
)

df.to_csv(
    archivo_individual,
    index=False,
    encoding="utf-8-sig"
)


resumen = (
    df.groupby(["H_teorico", "N"])
    .agg(
        H_media=("H_estimado", "mean"),
        H_std=("H_estimado", "std"),
        sesgo=("error", "mean"),
        MAE=("error_absoluto", "mean"),
        RMSE=(
            "error",
            lambda x: np.sqrt(np.mean(x**2))
        ),
        R2_medio=("R2", "mean"),
        realizaciones_validas=("H_estimado", "count")
    )
    .reset_index()
)

resumen.rename(
    columns={"H_teorico": "H"},
    inplace=True
)

resumen["error_estandar"] = (
    resumen["H_std"] /
    np.sqrt(resumen["realizaciones_validas"])
)

# Calcular intervalos bootstrap y p-valores
IC_inferiores = []
IC_superiores = []
p_valores = []

for _, fila in resumen.iterrows():

    H = fila["H"]
    N = fila["N"]

    datos = df[
        (df["H_teorico"] == H) &
        (df["N"] == N)
    ]["H_estimado"].values

    # Intervalo bootstrap de la media
    inferior, superior = intervalo_bootstrap_media(
        datos,
        B=B_BOOT,
        alpha=ALPHA,
        rng=rng
    )

    # Prueba bootstrap bilateral
    p = prueba_bootstrap_media(
        datos,
        valor_nulo=H,
        B=B_BOOT,
        rng=rng
    )

    IC_inferiores.append(inferior)
    IC_superiores.append(superior)
    p_valores.append(p)

resumen["IC95_inferior"] = IC_inferiores
resumen["IC95_superior"] = IC_superiores
resumen["p_valor"] = p_valores

#Correccion de Holm
p_ajustados, rechazar = correccion_holm(
    resumen["p_valor"].values,
    alpha=ALPHA
)

resumen["p_valor_Holm"] = p_ajustados
resumen["rechazar_H0"] = rechazar


archivo_resumen = (
    CARPETA_SALIDA /
    "resumen_variograma.csv"
)

resumen.to_csv(
    archivo_resumen,
    index=False,
    encoding="utf-8-sig"
)

#Mostrar resultados inferenciales
print("\n" + "=" * 85)
print("ANALISIS INFERENCIAL: BOOTSTRAP Y CORRECCION DE HOLM")
print("=" * 85)

for _, fila in resumen.iterrows():

    print(
        f"H={fila['H']:.1f} | "
        f"N={int(fila['N'])} | "
        f"IC95=[{fila['IC95_inferior']:.6f}, "
        f"{fila['IC95_superior']:.6f}] | "
        f"p={fila['p_valor']:.6f} | "
        f"p_Holm={fila['p_valor_Holm']:.6f} | "
        f"Rechazar H0={fila['rechazar_H0']}"
    )

total_pruebas = len(resumen)

total_rechazos = int(
    resumen["rechazar_H0"].sum()
)

total_no_rechazos = (
    total_pruebas - total_rechazos
)

print("\n" + "-" * 85)

print(f"Numero de pruebas:       {total_pruebas}")
print(f"Pruebas con rechazo H0:  {total_rechazos}")
print(f"Pruebas sin rechazo H0:  {total_no_rechazos}")
print(f"Nivel de significancia:  {ALPHA}")
print(f"Remuestreos bootstrap:   {B_BOOT}")

print("=" * 85)


#GRAFICA: H ESTIMADO VS H TEORICO
plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.errorbar(
        datos["H"],
        datos["H_media"],
        yerr=(
            datos["H_media"] - datos["IC95_inferior"],
            datos["IC95_superior"] - datos["H_media"]
        ),
        marker="o",
        capsize=3,
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.plot(
    Hs,
    Hs,
    "--",
    color="black",
    label="Valor teorico"
)

plt.xlabel(r"$H$ teorico")
plt.ylabel(r"$\widehat{H}$ estimado")
plt.title(
    "Recuperacion del exponente de Hurst mediante variograma"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "H_estimado_vs_H_teorico.png",
    dpi=300
)

plt.close()


#GRAFICA: SESGO

plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["sesgo"],
        marker="o",
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel(r"$H$ teorico")
plt.ylabel("Sesgo")
plt.title(
    "Sesgo de la estimacion del exponente de Hurst"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "sesgo_vs_H.png",
    dpi=300
)

plt.close()


#GRAFICA: DESVIACION ESTANDAR
plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["H_std"],
        marker="o",
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.xlabel(r"$H$ teorico")
plt.ylabel("Desviacion estandar")
plt.title("Variabilidad de las estimaciones")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "desviacion_estandar_vs_H.png",
    dpi=300
)

plt.close()


#BOXPLOTS PARA CADA N
for N in Ns:

    datos_N = df[df["N"] == N]

    datos_boxplot = [
        datos_N[
            datos_N["H_teorico"] == H
        ]["H_estimado"].values
        for H in Hs
    ]

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        datos_boxplot,
        tick_labels=[f"{H:.1f}" for H in Hs],
        showfliers=False
    )

    plt.plot(
        np.arange(1, len(Hs) + 1),
        Hs,
        "r--o",
        markersize=4,
        label="H teorico"
    )

    plt.xlabel(r"$H$ teorico")
    plt.ylabel(r"$\widehat{H}$ estimado")
    plt.title(
        f"Distribucion de estimaciones del variograma, N={N}"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / f"boxplot_variograma_N{N}.png",
        dpi=300
    )

    plt.close()


#GRAFICA: R2 MEDIO

plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["R2_medio"],
        marker="o",
        linewidth=1.5,
        label=f"N = {N}"
    )

plt.xlabel(r"$H$ teorico")
plt.ylabel(r"$R^2$ medio")
plt.title(
    "Coeficiente de determinacion del ajuste log-log"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "R2_vs_H.png",
    dpi=300
)

plt.close()


# EJEMPLOS DE FUNCION DE ESTRUCTURA Y AJUSTE

N_ejemplo = max(Ns)

fig, axes = plt.subplots(
    3, 3,
    figsize=(15, 12)
)

for ax, H in zip(axes.flatten(), Hs):

    ejemplo = ejemplos.get(
        (H, N_ejemplo)
    )

    if ejemplo is None:

        ax.set_visible(False)
        continue

    ax.loglog(
        ejemplo["taus"],
        ejemplo["S2"],
        "o",
        markersize=3,
        label="Funcion de estructura"
    )

    ax.loglog(
        ejemplo["taus"],
        ejemplo["S2_ajuste"],
        "--",
        linewidth=1.5,
        label="Ajuste lineal"
    )

    ax.set_title(
        f"H={H:.1f} | "
        f"H estimado={ejemplo['H_estimado']:.4f}"
    )

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$S_2(\tau)$")
    ax.grid(
        True,
        which="both",
        alpha=0.3
    )

axes[0, 0].legend(fontsize=8)

fig.suptitle(
    f"Funcion de estructura y ajustes log-log, N={N_ejemplo}",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "ejemplos_variograma_loglog.png",
    dpi=300
)

plt.close()


#COMPARACION DE ESTIMACIONES CON VALORES TEORICOS

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

for ax, N in zip(axes.flatten(), Ns):

    datos = resumen[resumen["N"] == N]

    ax.errorbar(
        datos["H"],
        datos["H_media"],
        yerr=(
            datos["H_media"] - datos["IC95_inferior"],
            datos["IC95_superior"] - datos["H_media"]
        ),
        fmt="o",
        capsize=3,
        label="Estimacion"
    )

    ax.plot(
        Hs,
        Hs,
        "--",
        color="black",
        label="Teorico"
    )

    ax.set_title(f"N = {N}")
    ax.set_xlabel(r"$H$ teorico")
    ax.set_ylabel(r"$\widehat{H}$")
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle(
    "Comparacion entre H estimado y H teorico",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "comparacion_H_teorico_estimado.png",
    dpi=300
)

plt.close()


# RESUMEN FINAL
print("\n" + "=" * 85)
print("RESUMEN FINAL DEL EXPERIMENTO")
print("=" * 85)

print(
    resumen.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

