# -*- coding: utf-8 -*-
"""
Created on Tue Oct  11 15:01:04 2026

@author: Ricar

Módulo fGn: generación de ruido gaussiano fraccional (fGn) y
movimiento browniano fraccional (fBm) por el método de Davies-Harte
(incrustación circulante + FFT), válido para 0 < H < 1.

Uso:
    import fGn
    inc = fGn.fGn(N, H, seed=1)     # N incrementos, varianza 1
    x   = fGn.fBm(N, H, seed=1)     # trayectoria de N puntos, x[0] = 0
"""
import numpy as np


def autocovarianza(n, H):
    """gamma(k), k = 0..n, del fGn con varianza unitaria."""
    k = np.arange(0, n + 1, dtype=float)
    return 0.5 * (np.abs(k + 1)**(2*H) - 2*k**(2*H) + np.abs(k - 1)**(2*H))


def fGn(N, H, seed=None, tol=1e-10):
    """N muestras de fGn con exponente de Hurst H (0 < H < 1)."""
    if not (0 < H < 1):
        raise ValueError("H debe estar en (0, 1).")
    if N < 2:
        raise ValueError("N debe ser >= 2.")

    rng = np.random.default_rng(seed)

    g = autocovarianza(N, H)
    r = np.concatenate([g, g[-2:0:-1]])      # primera fila del circulante (tamaño 2N)
    M = len(r)

    lam = np.fft.fft(r).real                 # autovalores del circulante
    if lam.min() < -tol * lam.max():
        raise RuntimeError("Incrustación circulante no semidefinida positiva.")
    lam = np.clip(lam, 0.0, None)            # solo ruido numérico

    z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    return np.fft.fft(np.sqrt(lam / M) * z).real[:N]


def fBm(N, H, seed=None):
    """Trayectoria de fBm de N puntos, con B(0) = 0 (acumula N-1 incrementos)."""
    inc = fGn(N - 1, H, seed=seed)
    return np.concatenate([[0.0], np.cumsum(inc)])
