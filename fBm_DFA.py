# -*- coding: utf-8 -*-
"""
Experimentos Monte Carlo de DFA-1 aplicado directamente a fBm.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fGn

Hs = np.round(np.arange(0.1, 1.0, 0.1), 1)

Ns = [1024, 4096, 16384, 65536]

M = 300

ORDEN_DFA = 1

NUM_ESCALAS = 25

ESCALA_MIN = 10

SEMILLA_BASE = 1337

CARPETA_SALIDA = Path("resultados_DFA_fBm")

CARPETA_GRAFICAS = CARPETA_SALIDA / "graficas"

CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

def dfa(X, escalas, orden=1):

    X = np.asarray(X, dtype=float)

    n = len(X)

    # Perfil acumulado
    Y = np.cumsum(X - np.mean(X))

    fluctuaciones = []
    escalas_validas = []

    for s in escalas:

        s = int(s)

        if s <= orden + 1 or s > n // 2:
            continue

        n_ventanas = n // s

        if n_ventanas < 2:
            continue

        residuos_cuadrados = []

        t = np.arange(s)

        # Ventanas desde el principio
        for v in range(n_ventanas):

            inicio = v * s
            fin = inicio + s

            segmento = Y[inicio:fin]

            coef = np.polyfit(t, segmento, orden)

            tendencia = np.polyval(coef, t)

            residuos = segmento - tendencia

            residuos_cuadrados.extend(residuos ** 2)

        # Ventanas desde el final
        for v in range(n_ventanas):

            inicio = n - (v + 1) * s
            fin = inicio + s

            segmento = Y[inicio:fin]

            coef = np.polyfit(t, segmento, orden)

            tendencia = np.polyval(coef, t)

            residuos = segmento - tendencia

            residuos_cuadrados.extend(residuos ** 2)

        F_s = np.sqrt(np.mean(residuos_cuadrados))

        if np.isfinite(F_s) and F_s > 0:

            escalas_validas.append(s)
            fluctuaciones.append(F_s)

    return np.asarray(escalas_validas), np.asarray(fluctuaciones)

def estimar_exponente(X, orden=1):

    n = len(X)

    escalas = np.unique(
        np.logspace(
            np.log10(ESCALA_MIN),
            np.log10(n // 4),
            NUM_ESCALAS
        ).astype(int)
    )

    s, F = dfa(X, escalas, orden)

    if len(s) < 3:
        return np.nan, np.nan

    log_s = np.log(s)
    log_F = np.log(F)

    alpha, intercepto = np.polyfit(log_s, log_F, 1)

    F_ajuste = np.exp(intercepto) * s ** alpha

    # Coeficiente de determinación
    ss_res = np.sum((log_F - np.log(F_ajuste)) ** 2)

    ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)

    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan

    return alpha, r2


resultados = []

total_experimentos = len(Hs) * len(Ns) * M

contador = 0

print("=" * 75)
print("EXPERIMENTOS DFA-1 APLICADOS A fBm")
print("=" * 75)

print(f"Valores de H: {Hs}")
print(f"Longitudes N: {Ns}")
print(f"Realizaciones M: {M}")
print(f"Orden DFA: {ORDEN_DFA}")
print(f"Número de escalas: {NUM_ESCALAS}")
print("=" * 75)


for N in Ns:

    print(f"\nLongitud de serie N = {N}")

    for H in Hs:

        alpha_teorico = H + 1

        estimaciones = []
        valores_r2 = []

        for m in range(M):

            contador += 1

            semilla = (
                SEMILLA_BASE
                + int(H * 1000) * 100000
                + N * 10
                + m
            )

            # Generar movimiento browniano fraccionario
            X = fGn.fBm(N, H, seed=semilla)

            # Aplicar DFA
            alpha_estimado, r2 = estimar_exponente(
                X,
                orden=ORDEN_DFA
            )

            if np.isfinite(alpha_estimado):

                estimaciones.append(alpha_estimado)
                valores_r2.append(r2)

                resultados.append({
                    "H": H,
                    "N": N,
                    "realizacion": m + 1,
                    "alpha_teorico": alpha_teorico,
                    "alpha_estimado": alpha_estimado,
                    "error": alpha_estimado - alpha_teorico,
                    "R2": r2
                })

            if contador % 100 == 0:
                print(
                    f"Progreso: {contador}/{total_experimentos}"
                )

        estimaciones = np.asarray(estimaciones)
        valores_r2 = np.asarray(valores_r2)

        if len(estimaciones) == 0:
            continue

        media = np.mean(estimaciones)

        sesgo = media - alpha_teorico

        desviacion = np.std(estimaciones, ddof=1)

        mae = np.mean(
            np.abs(estimaciones - alpha_teorico)
        )

        rmse = np.sqrt(
            np.mean((estimaciones - alpha_teorico) ** 2)
        )

        r2_medio = np.nanmean(valores_r2)

        resumen = {
            "H": H,
            "N": N,
            "alpha_teorico": alpha_teorico,
            "media": media,
            "sesgo": sesgo,
            "MAE": mae,
            "RMSE": rmse,
            "desviacion_estandar": desviacion,
            "R2_medio": r2_medio,
            "realizaciones_validas": len(estimaciones)
        }

        print(
            f"H={H:.1f} | "
            f"alpha teórico={alpha_teorico:.3f} | "
            f"media={media:.4f} | "
            f"sesgo={sesgo:.4f} | "
            f"std={desviacion:.4f} | "
            f"R²={r2_medio:.4f} | "
            f"válidas={len(estimaciones)}"
        )

df = pd.DataFrame(resultados)

archivo_individual = (
    CARPETA_SALIDA / "resultados_individuales_DFA_fBm.csv"
)

df.to_csv(archivo_individual, index=False)

resumenes = []

for (N, H), grupo in df.groupby(["N", "H"]):

    teorico = H + 1

    estimaciones = grupo["alpha_estimado"].to_numpy()

    resumenes.append({
        "H": H,
        "N": N,
        "alpha_teorico": teorico,
        "media": np.mean(estimaciones),
        "sesgo": np.mean(estimaciones) - teorico,
        "MAE": np.mean(np.abs(estimaciones - teorico)),
        "RMSE": np.sqrt(np.mean((estimaciones - teorico) ** 2)),
        "desviacion_estandar": np.std(estimaciones, ddof=1),
        "R2_medio": grupo["R2"].mean(),
        "realizaciones_validas": len(grupo)
    })

df_resumen = pd.DataFrame(resumenes)

archivo_resumen = CARPETA_SALIDA / "resumen_DFA_fBm.csv"

df_resumen.to_csv(archivo_resumen, index=False)

print("\nArchivos CSV guardados correctamente.")

plt.figure(figsize=(8, 5))

for N in Ns:

    datos = df_resumen[df_resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["desviacion_estandar"],
        marker="o",
        label=f"N = {N}"
    )

plt.xlabel("Exponente de Hurst H")
plt.ylabel("Desviación estándar de α DFA")

plt.title("Dispersión de las estimaciones DFA para fBm")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "desviacion_estandar_vs_H.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

for N in Ns:

    datos = df_resumen[df_resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["sesgo"],
        marker="o",
        label=f"N = {N}"
    )

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Exponente de Hurst H")
plt.ylabel("Sesgo")

plt.title("Sesgo de las estimaciones DFA para fBm")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "sesgo_vs_H.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(8, 5))

for N in Ns:

    datos = df_resumen[df_resumen["N"] == N]

    plt.plot(
        datos["H"],
        datos["media"],
        marker="o",
        label=f"Estimación N={N}"
    )

# Relación teórica alpha = H + 1
H_linea = np.linspace(0.1, 0.9, 100)

plt.plot(
    H_linea,
    H_linea + 1,
    "--",
    label=r"Teórico: $\alpha=H+1$"
)

plt.xlabel("Exponente de Hurst H")
plt.ylabel("Exponente DFA estimado")

plt.title("Exponente DFA estimado frente al valor teórico")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "estimacion_media_vs_H.png",
    dpi=300
)

plt.show()

for N in Ns:

    datos_N = df[df["N"] == N]

    grupos = [
        datos_N[datos_N["H"] == H]["alpha_estimado"].values
        for H in Hs
    ]

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        grupos,
        tick_labels=[f"{H:.1f}" for H in Hs],
        showfliers=False
    )

    # Valores teóricos alpha = H + 1
    teoricos = Hs + 1

    plt.plot(
        np.arange(1, len(Hs) + 1),
        teoricos,
        "r--o",
        markersize=4,
        label=r"Teórico: $\alpha=H+1$"
    )

    plt.xlabel("Exponente de Hurst H")
    plt.ylabel("Exponente DFA estimado")

    plt.title(f"Distribución de estimaciones DFA para N = {N}")

    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / f"boxplot_DFA_N{N}.png",
        dpi=300
    )

    plt.show()

