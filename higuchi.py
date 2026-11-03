# -*- coding: utf-8 -*-
"""
Created on Tue Oct  17 18:42:40 2026

@author: Ricar

ESTIMACION DE DIMENSION FRACTAL MEDIANTE EL METODO DE HIGUCHI

Experimento Monte Carlo:
- H = 0.1, 0.2, ..., 0.9
- N = 2^10, 2^12, 2^14, 2^16
- 300 realizaciones por cada combinacion (H, N)

Se calcula:
- Dimension fractal de Higuchi por realizacion.
- Calidad del ajuste log-log (R2).
- Media y desviacion estandar.
- Error respecto a D = 2 - H.
- Intervalos de confianza del 95% de la media.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fGn import fBm


Hs = np.round(np.arange(0.1, 1.0, 0.1), 1)

Ns = [2**10, 2**12, 2**14, 2**16]

N_REALIZACIONES = 300

KMAX_HIGUCHI = 10

SEMILLA_EXPERIMENTO = 1337

CARPETA_SALIDA = Path("resultados_Higuchi_fBm")

CARPETA_GRAFICAS = CARPETA_SALIDA / "graficas"
CARPETA_EJEMPLOS = CARPETA_SALIDA / "ejemplos"

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
CARPETA_GRAFICAS.mkdir(exist_ok=True)
CARPETA_EJEMPLOS.mkdir(exist_ok=True)



def generar_fbm(N, H, seed):

    x = np.asarray(
        fBm(N, H, seed=seed),
        dtype=float
    ).ravel()

    if len(x) == N + 1:
        x = x[:N]

    elif len(x) != N:
        raise ValueError(
            f"N={N}: se esperaban {N} o {N+1} muestras, "
            f"pero fBm devolvio {len(x)}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"La realizacion seed={seed} contiene valores no finitos."
        )

    return x


def dimension_higuchi(x, kmax=10):
    """
    Estima la dimension fractal mediante Higuchi.

    Ajusta:

        log(L(k)) = D * log(1/k) + c

    Regresa:
        D       : dimension estimada
        R2      : coeficiente de determinacion
        ks      : escalas utilizadas
        Lk      : longitudes promedio
    """

    x = np.asarray(x, dtype=float)

    N = len(x)

    kmax = min(kmax, (N - 1) // 2)

    ks = np.arange(1, kmax + 1)

    Lk = []

    for k in ks:

        longitudes_m = []

        for m in range(k):

            indices = np.arange(m, N, k)

            if len(indices) < 2:
                continue

            diferencias = np.abs(np.diff(x[indices]))

            n_tramos = len(indices) - 1

            longitud = (
                np.sum(diferencias)
                * (N - 1)
                / (n_tramos * k**2)
            )

            longitudes_m.append(longitud)

        Lk.append(np.mean(longitudes_m))

    Lk = np.asarray(Lk, dtype=float)

    validos = np.isfinite(Lk) & (Lk > 0)

    if validos.sum() < 3:

        return np.nan, np.nan, ks, Lk

    log_escala = np.log(1.0 / ks[validos])
    log_longitud = np.log(Lk[validos])

    D, intercepto = np.polyfit(
        log_escala,
        log_longitud,
        1
    )

    ajuste = D * log_escala + intercepto

    ss_res = np.sum((log_longitud - ajuste)**2)

    ss_tot = np.sum(
        (log_longitud - np.mean(log_longitud))**2
    )

    R2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return D, R2, ks, Lk


def graficar_trayectoria(x, H, N):

    t = np.linspace(0, 1, len(x))

    plt.figure(figsize=(9, 4))

    plt.plot(
        t,
        x,
        linewidth=0.8
    )

    plt.xlabel("$t$")
    plt.ylabel("$X(t)$")

    plt.title(
        rf"Realización de fBm: $H={H:.1f}$, $N={N}$"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        CARPETA_EJEMPLOS / f"trayectoria_H{H:.1f}_N{N}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def graficar_ajuste_higuchi(H, N, D, R2, ks, Lk):

    validos = np.isfinite(Lk) & (Lk > 0)

    x_log = np.log(1.0 / ks[validos])
    y_log = np.log(Lk[validos])

    pendiente, intercepto = np.polyfit(
        x_log,
        y_log,
        1
    )

    ajuste = pendiente * x_log + intercepto

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_log,
        y_log,
        'o',
        label="Datos"
    )

    plt.plot(
        x_log,
        ajuste,
        '--',
        label=rf"Ajuste: $D={D:.4f}$"
    )

    plt.xlabel(r"$\log(1/k)$")
    plt.ylabel(r"$\log L(k)$")

    plt.title(
        rf"Higuchi: $H={H:.1f}$, $N={N}$, $R^2={R2:.4f}$"
    )

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CARPETA_EJEMPLOS /
        f"ajuste_H{H:.1f}_N{N}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



rng_semillas = np.random.default_rng(SEMILLA_EXPERIMENTO)

total_experimentos = (
    len(Hs) * len(Ns) * N_REALIZACIONES
)

semillas = rng_semillas.choice(
    np.arange(1, 2000000000, dtype=np.int64),
    size=total_experimentos,
    replace=False
)

print("=" * 75)
print("EXPERIMENTO DE DIMENSION FRACTAL MEDIANTE HIGUCHI")
print("=" * 75)

print("Valores de H:", Hs)
print("Valores de N:", Ns)
print("Realizaciones por combinacion:", N_REALIZACIONES)
print("Total de realizaciones:", total_experimentos)
print("KMAX:", KMAX_HIGUCHI)
print("Semillas unicas:", len(np.unique(semillas)))


resultados = []

indice_semilla = 0
contador = 0

# Se guardan ejemplos para las figuras.
# Una trayectoria para cada H con el mayor N.
ejemplos_guardados = set()

for H in Hs:

    for N in Ns:

        dimension_teorica = 2.0 - H

        dimensiones = []
        r2_valores = []

        print("\n" + "-" * 65)
        print(f"H = {H:.1f} | N = {N}")
        print(f"Dimension teorica: {dimension_teorica:.4f}")
        print("-" * 65)

        for r in range(N_REALIZACIONES):

            contador += 1

            seed = int(semillas[indice_semilla])
            indice_semilla += 1

            # Generar trayectoria
            x = generar_fbm(N, H, seed)

            # Estimar dimension de Higuchi
            D, R2, ks, Lk = dimension_higuchi(
                x,
                kmax=KMAX_HIGUCHI
            )

            error = D - dimension_teorica

            dimensiones.append(D)
            r2_valores.append(R2)

            resultados.append({

                "H": H,
                "N": N,
                "realizacion": r + 1,
                "seed": seed,

                "D_teorica": dimension_teorica,
                "D_Higuchi": D,

                "error_D": error,
                "R2": R2

            })

            
            if N == Ns[0] and r == 0:

                graficar_trayectoria(
                    x, H, N
                )

            # Ajuste log-log para N=max:
            # se guarda una trayectoria representativa por H.
            if N == Ns[-1] and r == 0:

                graficar_trayectoria(
                    x, H, N
                )

                graficar_ajuste_higuchi(
                    H, N, D, R2, ks, Lk
                )

        # Estadisticas de la combinacion
        dimensiones = np.asarray(dimensiones)
        r2_valores = np.asarray(r2_valores)

        validas = np.isfinite(dimensiones)

        dimensiones_validas = dimensiones[validas]

        media = np.mean(dimensiones_validas)

        desviacion = np.std(
            dimensiones_validas,
            ddof=1
        )

        error_medio = np.mean(
            dimensiones_validas - dimension_teorica
        )

        mae = np.mean(
            np.abs(dimensiones_validas - dimension_teorica)
        )

        rmse = np.sqrt(
            np.mean(
                (dimensiones_validas - dimension_teorica)**2
            )
        )

        n_validas = len(dimensiones_validas)

        sem = desviacion / np.sqrt(n_validas)

        ic95 = 1.96 * sem

        print(f"Dimension media: {media:.6f}")
        print(f"Desviacion estandar: {desviacion:.6f}")
        print(f"IC 95% de la media: +/- {ic95:.6f}")
        print(f"Sesgo: {error_medio:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R2 medio: {np.nanmean(r2_valores):.6f}")
        print(f"Realizaciones validas: {n_validas}")


df = pd.DataFrame(resultados)

df.to_csv(
    CARPETA_SALIDA / "resultados_individuales_Higuchi.csv",
    index=False
)


resumen = df.groupby(["H", "N"]).agg(

    D_media=("D_Higuchi", "mean"),
    D_std=("D_Higuchi", "std"),

    error_medio=("error_D", "mean"),
    MAE=("error_D", lambda x: np.mean(np.abs(x.dropna()))),

    RMSE=(
        "error_D",
        lambda x: np.sqrt(np.mean(x.dropna()**2))
    ),

    R2_medio=("R2", "mean"),

    realizaciones_validas=("D_Higuchi", "count")

).reset_index()

resumen["D_teorica"] = 2.0 - resumen["H"]

resumen["error_estandar"] = (
    resumen["D_std"]
    / np.sqrt(resumen["realizaciones_validas"])
)

resumen["IC95_inferior"] = (
    resumen["D_media"]
    - 1.96 * resumen["error_estandar"]
)

resumen["IC95_superior"] = (
    resumen["D_media"]
    + 1.96 * resumen["error_estandar"]
)

resumen.to_csv(
    CARPETA_SALIDA / "resumen_Higuchi.csv",
    index=False
)



#GRAFICA A: DIMENSION ESTIMADA VS H
plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.errorbar(

        datos["H"],
        datos["D_media"],

        yerr=datos["error_estandar"] * 1.96,

        marker="o",
        capsize=4,
        linestyle="-",

        label=f"N = {N}"

    )

plt.plot(
    Hs,
    2.0 - Hs,
    "k--",
    label=r"Teórica: $D=2-H$"
)

plt.xlabel("$H$")
plt.ylabel(r"Dimensión fractal $D_{\mathrm{Higuchi}}$")

plt.title("Dimensión fractal estimada mediante Higuchi")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "dimension_vs_H.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


#GRAFICA B: DIMENSION ESTIMADA VS TEORICA
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9),
    constrained_layout=True
)

axes = axes.ravel()

for ax, N in zip(axes, Ns):

    datos = resumen[resumen["N"] == N]

    ax.errorbar(

        datos["D_teorica"],
        datos["D_media"],

        yerr=datos["error_estandar"] * 1.96,

        fmt="o",
        capsize=4,
        label="Estimación Higuchi"

    )

    minimo = min(
        datos["D_teorica"].min(),
        datos["D_media"].min()
    )

    maximo = max(
        datos["D_teorica"].max(),
        datos["D_media"].max()
    )

    ax.plot(
        [minimo, maximo],
        [minimo, maximo],
        "k--",
        label="Identidad"
    )

    ax.set_xlabel("Dimensión teórica")
    ax.set_ylabel("Dimensión estimada")

    ax.set_title(f"N = {N}")

    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle(
    "Comparación entre dimensión estimada y teórica"
)

fig.savefig(
    CARPETA_GRAFICAS / "dimension_estimada_vs_teorica.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


#GRAFICA C: SESGO VS H
plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["error_medio"],
        marker="o",
        label=f"N = {N}"
    )

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("$H$")
plt.ylabel(r"Sesgo: $\langle D\rangle-(2-H)$")

plt.title("Sesgo de la estimación de Higuchi")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "sesgo_vs_H.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()



#GRAFICA D: DESVIACION ESTANDAR VS H
plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["D_std"],
        marker="o",
        label=f"N = {N}"
    )

plt.xlabel("$H$")
plt.ylabel(r"Desviación estándar de $D_{\mathrm{Higuchi}}$")

plt.title("Variabilidad de la dimensión estimada")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "desviacion_estandar_vs_H.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


#GRAFICA E: BOXPLOTS PARA CADA N
for N in Ns:

    datos_N = df[df["N"] == N]

    datos_cajas = [

        datos_N[datos_N["H"] == H]["D_Higuchi"].dropna().values

        for H in Hs

    ]

    plt.figure(figsize=(11, 6))

    plt.boxplot(
        datos_cajas,
        labels=[f"{H:.1f}" for H in Hs],
        showfliers=False
    )

    plt.plot(
        np.arange(1, len(Hs) + 1),
        2.0 - Hs,
        "o--",
        label="Dimensión teórica"
    )

    plt.xlabel("$H$")
    plt.ylabel(r"Dimensión $D_{\mathrm{Higuchi}}$")

    plt.title(
        f"Distribución de dimensiones estimadas: N = {N}"
    )

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / f"boxplot_Higuchi_N{N}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()



#GRAFICA F: CALIDAD DEL AJUSTE
plt.figure(figsize=(9, 6))

for N in Ns:

    datos = resumen[resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["R2_medio"],
        marker="o",
        label=f"N = {N}"
    )

plt.xlabel("$H$")
plt.ylabel(r"$R^2$ medio")

plt.title("Calidad promedio del ajuste log-log de Higuchi")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "R2_vs_H.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()