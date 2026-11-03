import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from fGn import fBm



Ns = [2**10, 2**12, 2**14, 2**16]
Hs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
M  = 300 # realizaciones par cada (N, H)
N_JOBS = -1                    

RESOLUCIONES_BASE = [4, 8, 16, 32, 64, 128, 256]


def segmento_en_celda(x0, y0, x1, y1,
                      xmin, xmax, ymin, ymax):
    dx = x1 - x0
    dy = y1 - y0

    entrada, salida = 0.0, 1.0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                entrada = max(entrada, r)
            else:
                salida = min(salida, r)
            if entrada > salida:
                return False

    return True


def box_counting(t, x, m):
    epsilon = 1.0 / m
    ocupadas = np.zeros((m, m), dtype=bool)

    for i in range(len(t) - 1):
        x0, y0 = t[i], x[i]
        x1, y1 = t[i + 1], x[i + 1]

        col_min = max(0, int(np.floor(min(x0, x1) * m)))
        col_max = min(m - 1, int(np.floor(max(x0, x1) * m)))
        fila_min = max(0, int(np.floor(min(y0, y1) * m)))
        fila_max = min(m - 1, int(np.floor(max(y0, y1) * m)))

        for fila in range(fila_min, fila_max + 1):
            for col in range(col_min, col_max + 1):
                if ocupadas[fila, col]:
                    continue

                xmin = col * epsilon
                xmax = (col + 1) * epsilon
                ymin = fila * epsilon
                ymax = (fila + 1) * epsilon

                if segmento_en_celda(x0, y0, x1, y1,
                                     xmin, xmax, ymin, ymax):
                    ocupadas[fila, col] = True

    return np.count_nonzero(ocupadas), ocupadas


#RESOLUCIONES VÁLIDAS PARA UN N DADO
def resoluciones_para(N):
    m_max = min(256, max(4, N // 8))   # necesitamos N >> m
    return [m for m in RESOLUCIONES_BASE if m <= m_max]

def estimar_D(N, H, seed, resoluciones, guardar_mascaras=False):
    senal = np.asarray(fBm(N, H, seed=int(seed)), dtype=float).ravel()[:N]

    rango = senal.max() - senal.min()
    if rango == 0:
        return np.nan, None

    x = (senal - senal.min()) / rango
    t = np.linspace(0, 1, N)

    epsilons = []
    conteos  = []
    mascaras = {}

    for m in resoluciones:
        N_cajas, ocupadas = box_counting(t, x, m)
        epsilons.append(1.0 / m)
        conteos.append(N_cajas)
        if guardar_mascaras:
            mascaras[m] = ocupadas

    epsilons = np.array(epsilons, dtype=float)
    conteos  = np.array(conteos, dtype=float)

    # Guardián: sin variación no se puede ajustar
    if len(np.unique(conteos)) < 2:
        return np.nan, (mascaras if guardar_mascaras else None)

    log_inv_eps = np.log(1.0 / epsilons)
    log_conteos = np.log(conteos)

   
    # Descartar los m más pequeños (rango completo, no fractal)
    # y los más grandes (saturación cuando m ~ N).
    # Criterio: 2 <= log(1/eps) y m <= N/16
    mask = (epsilons <= 0.5) & (epsilons >= 16.0 / N)
    if mask.sum() < 3:                     
        mask = np.ones_like(epsilons, dtype=bool)

    D_box, intercepto = np.polyfit(
        log_inv_eps[mask], log_conteos[mask], 1
    )

    return D_box, (mascaras if guardar_mascaras else None)


resultados = {N: {float(H): np.full(M, np.nan) for H in Hs} for N in Ns}


N_MUESTRA, H_MUESTRA, I_MUESTRA = Ns[0], float(Hs[2]), 0
muestra = None

for N in Ns:
    resoluciones = resoluciones_para(N)
    print(f"\n########## N = {N}  |  m usados = {resoluciones} ##########")

    for H in Hs:
        H = float(H)
        print(f"  --- H = {H:.2f}  (D_teo = {2 - H:.2f}) ---")

        # ---- Paralelizamos las M realizaciones ----
        out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
            delayed(estimar_D)(
                N, H, i, resoluciones,
                guardar_mascaras=(N == N_MUESTRA and H == H_MUESTRA
                                  and i == I_MUESTRA)
            )
            for i in range(M)
        )

        D_vals = np.array([o[0] for o in out], dtype=float)
        resultados[N][H] = D_vals

        # Guardar la muestra del box-counting (una sola vez)
        if N == N_MUESTRA and H == H_MUESTRA:
            for i, (Dv, masc) in enumerate(out):
                if i == I_MUESTRA and masc is not None:
                    # reconstruimos x y t para graficar
                    senal = np.asarray(
                        fBm(N, H, seed=int(i)), dtype=float
                    ).ravel()[:N]
                    xm = (senal - senal.min()) / (senal.max() - senal.min())
                    tm = np.linspace(0, 1, N)
                    muestra = (tm, xm, masc, N, H)
                    break

        print(f"      D = {np.nanmean(D_vals):.4f} "
              f"± {np.nanstd(D_vals):.4f}  "
              f"(nan: {np.isnan(D_vals).sum()}/{M})")


if muestra is not None:
    t_m, x_m, mascaras_m, N_m, H_m = muestra

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, m in zip(axes.flat, resoluciones_para(N_m)):
        ocupadas = mascaras_m[m]
        eps = 1.0 / m

        for fila, col in np.argwhere(ocupadas):
            ax.add_patch(plt.Rectangle(
                (col * eps, fila * eps), eps, eps,
                facecolor="orange", edgecolor="none", alpha=0.35
            ))

        for j in range(m + 1):
            ax.axvline(j / m, color="gray", linewidth=0.35)
            ax.axhline(j / m, color="gray", linewidth=0.35)

        ax.plot(t_m, x_m, color="blue", linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"m={m}, eps={eps:.4f}\n"
                     f"N cajas={np.count_nonzero(ocupadas)}")

    fig.suptitle(f"Box-Counting (muestra) — N={N_m}, H={H_m}", fontsize=14)
    plt.tight_layout()
    plt.show()



#D vs H  (una curva por N)  +  teoría D = 2 − H
fig, axes = plt.subplots(1, len(Ns), figsize=(4.5 * len(Ns), 4.5),
                         sharey=True)
if len(Ns) == 1:
    axes = [axes]

for ax, N in zip(axes, Ns):
    D_med = np.array([np.nanmean(resultados[N][float(H)]) for H in Hs])
    D_std = np.array([np.nanstd(resultados[N][float(H)])  for H in Hs])
    D_teo = 2.0 - Hs

    ax.errorbar(Hs, D_med, yerr=D_std, fmt="o-", capsize=4,
                label="D medido (mean ± std)")
    ax.plot(Hs, D_teo, "k--", label="D = 2 − H")
    ax.set_xlabel("H")
    ax.set_title(f"N = {N}  (m={resoluciones_para(N)})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Dimensión Box-Counting")
plt.tight_layout()
plt.show()



# CONVERGENCIA DE D CON N (una curva por H)
fig, ax = plt.subplots(figsize=(8, 6))
colores = plt.cm.viridis(np.linspace(0, 1, len(Hs)))

for H, c in zip(Hs, colores):
    D_med = np.array([np.nanmean(resultados[N][float(H)]) for N in Ns])
    D_std = np.array([np.nanstd(resultados[N][float(H)])  for N in Ns])
    ax.errorbar(Ns, D_med, yerr=D_std, fmt="o-", color=c, capsize=3,
                label=f"H={H:.1f} (teo {2-H:.2f})")

ax.set_xscale("log", base=2)
ax.set_xlabel("N")
ax.set_ylabel("D")
ax.set_title("Convergencia de D con N, para cada H")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# medido vs D teórico = 2 − H
fig, ax = plt.subplots(figsize=(7, 6))

for N, c in zip(Ns, plt.cm.plasma(np.linspace(0, 1, len(Ns)))):
    D_med = np.array([np.nanmean(resultados[N][float(H)]) for H in Hs])
    ax.scatter(2 - Hs, D_med, color=c, s=40, label=f"N={N}")

ax.plot([1.0, 2.0], [1.0, 2.0], "k--", label="D = 2 − H")
ax.set_xlabel("D teórico = 2 − H")
ax.set_ylabel("D medido (promedio)")
ax.set_title("Recuperación de D = 2 − H")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()