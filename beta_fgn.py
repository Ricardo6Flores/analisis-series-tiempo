# -*- coding: utf-8 -*-
"""
Verificación numérica de beta = 2 - 2H para fGn.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import fGn


Ns = [2**10, 2**12, 2**14, 2**16]
Hs = np.array([0.6, 0.7, 0.8, 0.9])

M = 300
N_BOOT = 2000
SEMILLA_BASE = 1337

K_MIN_FIT, K_MAX_FIT = 10, 100

RUTA = Path("resultados_beta")
RUTA.mkdir(exist_ok=True)

assert K_MIN_FIT >= 1 and M < 10000
assert all(K_MAX_FIT < N // 2 for N in Ns)


def acf_fft(X, kmax):
    """ACF muestral (media teórica = 0, no se resta la media muestral)."""
    N = len(X)
    F = np.fft.fft(X, n=2 * N)

    c = (
        np.fft.ifft(np.abs(F) ** 2).real[:kmax + 1]
        / np.arange(N, N - kmax - 1, -1)
    )

    return c / c[0]


def acf_teorica(k, H):
    k = np.asarray(k, dtype=float)

    return 0.5 * (
        (k + 1) ** (2 * H)
        - 2 * k ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
    )


def estimar_beta(rho, kmin=K_MIN_FIT, kmax=K_MAX_FIT):
    """Ajuste log-log de rho(k) ~ C k^-beta."""

    k = np.arange(len(rho))

    m = (
        (k >= kmin)
        & (k <= kmax)
        & (rho > 0)
        & np.isfinite(rho)
    )

    if m.sum() < 5:
        return np.nan

    return -linregress(
        np.log10(k[m]),
        np.log10(rho[m])
    ).slope


def correccion_holm(p_values, alpha=0.05):
    """
    Corrección de Holm para comparaciones múltiples.

    Devuelve:
        p_ajustados
        rechazos
    """

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    orden = np.argsort(p_values)
    p_ordenados = p_values[orden]

    p_ajustados_ordenados = np.empty(n)

    valores = np.array([
        (n - i) * p_ordenados[i]
        for i in range(n)
    ])

    # Los valores ajustados deben ser monótonos.
    p_ajustados_ordenados = np.maximum.accumulate(valores)
    p_ajustados_ordenados = np.clip(
        p_ajustados_ordenados, 0, 1
    )

    p_ajustados = np.empty(n)
    p_ajustados[orden] = p_ajustados_ordenados

    rechazos = p_ajustados < alpha

    return p_ajustados, rechazos



filas = []
curvas = {}

# Conservamos las ACF individuales para el análisis bootstrap.
acf_realizaciones = {}

k_all = np.arange(K_MAX_FIT + 1)


for i_N, N in enumerate(Ns):

    print(f"\nN = {N}")

    for i_H, H in enumerate(Hs):

        beta_teo = 2 - 2 * H

        A = np.empty((M, K_MAX_FIT + 1))

        for m in range(M):

            semilla = (
                SEMILLA_BASE
                + 1_000_000 * i_N
                + 10_000 * i_H
                + m
            )

            A[m] = acf_fft(
                fGn.fGn(N, H, seed=semilla),
                K_MAX_FIT
            )

        # Estimador principal: ACF promedio
        rho_prom = A.mean(axis=0)

        beta_hat = estimar_beta(rho_prom)

        # Bootstrap original sobre realizaciones
        rng = np.random.default_rng(
            SEMILLA_BASE + 500_000 + 100 * i_N + i_H
        )

        boots = np.array([
            estimar_beta(
                A[rng.integers(0, M, M)].mean(axis=0)
            )
            for _ in range(N_BOOT)
        ])

        boots = boots[np.isfinite(boots)]

        ic_inf, ic_sup = np.percentile(
            boots, [2.5, 97.5]
        )

        # Referencia: ajuste sobre la ACF teórica exacta
        beta_ventana = estimar_beta(
            acf_teorica(k_all, H)
        )

        # Diagnóstico: ajuste individual
        b_ind = np.array([
            estimar_beta(a) for a in A
        ])

        validas = np.isfinite(b_ind).sum()

        filas.append(dict(
            N=N,
            H=H,
            beta_teorico=beta_teo,
            beta_ventana_ACF_exacta=beta_ventana,
            beta_ACF_promedio=beta_hat,
            IC95_inf=ic_inf,
            IC95_sup=ic_sup,
            sesgo=beta_hat - beta_teo,
            H_estimado=1 - beta_hat / 2,
            beta_individual_media=np.nanmean(b_ind),
            beta_individual_mediana=np.nanmedian(b_ind),
            individuales_validas=int(validas),
        ))

        curvas[(N, H)] = rho_prom

        # Única incorporación al almacenamiento original:
        # guardar las ACF individuales para el análisis adicional.
        acf_realizaciones[(N, H)] = A.copy()

        print(
            f"  H={H:.1f} | teo {beta_teo:.3f} | "
            f"ventana exacta {beta_ventana:.3f} | "
            f"ACF prom {beta_hat:.3f} "
            f"[{ic_inf:.3f}, {ic_sup:.3f}] | "
            f"indiv {np.nanmean(b_ind):.3f} "
            f"({validas}/{M})"
        )



# ANÁLISIS DEL RESIDUO DEL EXPONENTE
resultados_delta = []
distribuciones_delta = {}

for i_N, N in enumerate(Ns):

    for i_H, H in enumerate(Hs):

        A = acf_realizaciones[(N, H)]

        beta_hat = df_beta = next(
            fila["beta_ACF_promedio"]
            for fila in filas
            if fila["N"] == N and fila["H"] == H
        )

        beta_teo = 2 - 2 * H

        # Residuo observado
        delta_hat = beta_hat - beta_teo

        # Bootstrap de las realizaciones
        rng = np.random.default_rng(
            SEMILLA_BASE + 800_000 + 100 * i_N + i_H
        )

        boots_delta = np.empty(N_BOOT)

        for b in range(N_BOOT):

            indices = rng.integers(0, M, M)

            rho_boot = A[indices].mean(axis=0)

            beta_boot = estimar_beta(rho_boot)

            boots_delta[b] = beta_boot - beta_teo

        boots_delta = boots_delta[
            np.isfinite(boots_delta)
        ]

        # Intervalo bootstrap del 95 %
        delta_ic_inf, delta_ic_sup = np.percentile(
            boots_delta, [2.5, 97.5]
        )

        # Prueba bootstrap bilateral:
        # H0: E[Delta] = 0
        #
        # Centrar la distribución bootstrap permite aproximar
        # la distribución del estadístico bajo H0.

        boots_centrados = (
            boots_delta - np.mean(boots_delta)
        )

        p_boot = (
            1 + np.sum(
                np.abs(boots_centrados) >= abs(delta_hat)
            )
        ) / (len(boots_centrados) + 1)

        resultados_delta.append(dict(
            N=N,
            H=H,
            delta=delta_hat,
            delta_IC95_inf=delta_ic_inf,
            delta_IC95_sup=delta_ic_sup,
            p_boot=p_boot,
        ))

        distribuciones_delta[(N, H)] = boots_delta


# DataFrame de residuos
df_delta = pd.DataFrame(resultados_delta)

# Corrección de Holm
p_holm, rechazos = correccion_holm(
    df_delta["p_boot"].values,
    alpha=0.05
)

df_delta["p_holm"] = p_holm
df_delta["rechazo_Holm"] = rechazos

# Tasa de rechazo
n_pruebas = len(df_delta)
n_rechazos = rechazos.sum()
tasa_rechazo = n_rechazos / n_pruebas

# Integrar resultados al resumen original
df = pd.DataFrame(filas)

df = df.merge(
    df_delta,
    on=["N", "H"],
    how="left"
)



# MÉTRICAS GLOBALES DE ESTIMACIÓN POR LONGITUD DE SERIE
filas_metricas = []

for N in Ns:

    # Resultados correspondientes a la longitud N
    d = df[df["N"] == N].sort_values("H")

    beta_teorico = d["beta_teorico"].to_numpy()
    beta_estimado = d["beta_ACF_promedio"].to_numpy()

    # Residuos de estimación
    errores = beta_estimado - beta_teorico

    # Sesgo medio
    sesgo_medio = np.mean(errores)

    # Error absoluto medio
    MAE = np.mean(np.abs(errores))

    # Raíz del error cuadrático medio
    RMSE = np.sqrt(np.mean(errores**2))

    # Coeficiente de determinación R²
    SS_res = np.sum(errores**2)

    SS_tot = np.sum(
        (beta_teorico - np.mean(beta_teorico))**2
    )

    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else np.nan

    filas_metricas.append({
        "N": N,
        "sesgo_medio": sesgo_medio,
        "MAE": MAE,
        "RMSE": RMSE,
        "R2": R2
    })


# DataFrame de métricas globales
df_metricas = pd.DataFrame(filas_metricas)

# Guardar resultados
df_metricas.to_csv(
    RUTA / "metricas_globales.csv",
    index=False
)


# Guardar resultados originales
df.to_csv(
    RUTA / "resumen_beta.csv",
    index=False
)

df_delta.to_csv(
    RUTA / "resumen_residuos.csv",
    index=False
)



print("\n" + "=" * 80)
print("ANÁLISIS DEL RESIDUO DEL EXPONENTE")
print("=" * 80)

print(
    df_delta.round(5).to_string(index=False)
)

print("\nRESULTADOS DE LAS PRUEBAS")
print("-" * 50)

print(f"Número de pruebas: {n_pruebas}")
print(f"Nivel de significancia: 0.05")
print(f"Corrección: Holm")
print(f"Rechazos después de Holm: {n_rechazos}")
print(f"Tasa de rechazo: {100 * tasa_rechazo:.2f}%")


print("\n" + "=" * 80)
print("MÉTRICAS GLOBALES DE ESTIMACIÓN")
print("=" * 80)

print(
    df_metricas.round(5).to_string(index=False)
)



#BETA Y RESIDUOS

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.5)
)

Hl = np.linspace(0.55, 0.95, 100)

# Subplot 1: beta estimado frente a H
ax = axes[0]

ax.plot(
    Hl,
    2 - 2 * Hl,
    "k--",
    lw=2,
    label=r"$\beta=2-2H$"
)

for N in Ns:

    d = df[df.N == N]

    ax.errorbar(
        d.H,
        d.beta_ACF_promedio,
        yerr=[
            np.clip(
                d.beta_ACF_promedio - d.IC95_inf,
                0, None
            ),
            np.clip(
                d.IC95_sup - d.beta_ACF_promedio,
                0, None
            )
        ],
        marker="o",
        capsize=4,
        label=f"N = {N}"
    )

ax.set(
    xlabel="H",
    ylabel=r"$\widehat{\beta}$",
    title=r"Estimación de $\beta$"
)

ax.grid(alpha=.3)
ax.legend()


# Subplot 2: residuo Delta
ax = axes[1]

for N in Ns:

    d = df[df.N == N]

    ax.errorbar(
        d.H,
        d.delta,
        yerr=[
            np.clip(
                d.delta - d.delta_IC95_inf,
                0, None
            ),
            np.clip(
                d.delta_IC95_sup - d.delta,
                0, None
            )
        ],
        marker="o",
        capsize=4,
        label=f"N = {N}"
    )

ax.axhline(
    0,
    color="k",
    ls="--",
    lw=1.5
)

ax.set(
    xlabel="H",
    ylabel=r"$\Delta=\widehat{\beta}-(2-2H)$",
    title="Residuo de la relación teórica"
)

ax.grid(alpha=.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    RUTA / "01_beta_y_residuos.png",
    dpi=300
)

plt.show()


#ACF Y RESIDUOS BOOTSTRAP

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 6)
)

# Subplot 1: ACF promedio y ACF teórica
ax = axes[0]

kk = np.arange(1, K_MAX_FIT + 1)

for H in Hs:

    ax.loglog(
        kk,
        acf_teorica(kk, H),
        "--",
        lw=2,
        label=f"Teórica H={H:.1f}"
    )

for N in Ns:

    for H in Hs:

        ax.loglog(
            kk,
            np.clip(
                curvas[(N, H)][1:],
                1e-6,
                None
            ),
            alpha=.35,
            label=f"N={N}, H={H:.1f}"
        )

ax.axvspan(
    K_MIN_FIT,
    K_MAX_FIT,
    color="gray",
    alpha=.08
)

ax.set(
    xlabel="Retardo k",
    ylabel=r"$\rho(k)$",
    title="ACF promedio y teórica"
)

ax.grid(
    True,
    which="both",
    alpha=.3
)

ax.legend(
    fontsize=7,
    ncol=2
)


# Subplot 2: distribuciones bootstrap de Delta
ax = axes[1]

datos_boxplot = []
etiquetas = []

for N in Ns:

    for H in Hs:

        datos_boxplot.append(
            distribuciones_delta[(N, H)]
        )

        etiquetas.append(
            f"{N}\nH={H:.1f}"
        )

ax.boxplot(
    datos_boxplot,
    labels=etiquetas,
    showfliers=False
)

ax.axhline(
    0,
    color="k",
    ls="--",
    lw=1.5
)

ax.set(
    xlabel="Tamaño de serie y parámetro H",
    ylabel=r"$\Delta$ bootstrap",
    title="Distribuciones bootstrap del residuo"
)

ax.tick_params(
    axis="x",
    labelsize=8
)

ax.grid(alpha=.3)

fig.tight_layout()

fig.savefig(
    RUTA / "02_acf_y_residuos.png",
    dpi=300
)

plt.show()