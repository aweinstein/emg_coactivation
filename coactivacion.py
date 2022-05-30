# -*- coding: utf-8 -*-

"""Función para calcular coactivación muscular.

-Laboratorio Integrativo de Biomecánica y Fisiología del Esfuerzo,
Escuela de Kinesiología, Universidad de los Andes, Chile-
-Escuela de Ingeniería Biomédica, Universidad de Valparaíso, Chile-
        --Profesores: Oscar Valencia & Alejandro Weinstein--
"""
# importar librerías
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import numpy as np
from emg_cvm_norm import ajusta_emg_func


def coactivation_index(emg_A, emg_B, fs):
    """Calcula el índice de coactivación.

    La función calcula la coactivación entre músculos antagonistas [1,2]. El
    cálculo se realiza a partir de dos señales de EMG previamente procesadas y
    normalizadas (en base al valor RMS o promedio de la señal rectificada).

    Ambas señales deben ser adquiridas con la misma frecuencia de
    muestreo y tener el mismo largo.

    Parameters
    ----------
    emg_a : array_like
        EMG funcional de uno de los músculos antagonistas (A).
    emg_b : array_like
        EMG funcional del otro músculo antagonista (B).
    fs : float
        Frecuencia de muestreo de la señal EMG, en hertz. Debe ser la misma
        para ambas señales.

    Return
    -------
    Coeficiente de activación (porcentaje)

    Referencia
    ----------
    .. [1] Falconer K, Winter DA. Quantitative assessment of cocontraction at
    the ankle joint in walking.  Electromyogr Clin Neurophysiol 1985; 25:
    135-149.
    .. [2] Guilleron, C., Maktouf, W., Beaune, B., Henni, S., Abraham, P., &
    Durand, S. (2021). Coactivation pattern in leg muscles during treadmill
    walking in patients suffering from intermittent claudication. Gait &
    Posture, 84, 245-253.
    """
    I_antagonist = trapz(np.minimum(emg_A, emg_B)) / fs
    I_total = trapz(emg_A + emg_B) / fs
    return 2 * I_antagonist / I_total * 100


def plot_coactivacion(ax, emg_A, emg_B, label_musc_A, label_musc_B, ci):
    """Gráfica la coactivación entre dos músculos.

    La función grafica dos señales de EMG. Se muestra el área correspondiente a
    la región de coactivación.

    Parameters
    ----------
    ax : Matplotlib axes
        Ejes sobre los cuales se grafica la coactivación.
    emg_A, emg_B : array_like
        Vectores con la señal de EMG de los dos músculos a graficar.
    label_musc_A, label_musc_B : string
        Etiqueta a utilizar para cada músculo.
    ci : float
        Valor del índice de coactivación.
    """
    t = np.arange(0, len(emg_A) / fs, 1 / fs)
    emg_min = np.minimum(emg_A, emg_B).min()
    emg_max = np.maximum(emg_A, emg_B).max()
    ax.plot(t, emg_A, lw=2, label=label_musc_A)
    ax.plot(t, emg_B, lw=2, label=label_musc_B)
    ax.plot([], [], ' ', label=f'IC = {ci:0.2f}%')
    ax.fill_between(t, np.minimum(emg_A, emg_B), alpha=0.3, color='k')
    ax.set_ylim(0.8 * emg_min, 1.2 * emg_max)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel('Tiempo [s]', fontsize=12)
    ax.set_ylabel('Amplitud EMG [%CVM]', fontsize=12)
    ax.legend()


if __name__ == '__main__':
    # Cargar archivos con EMG de agonista y antagonista
    emg_cvm_vm, emg_cvm_vl = np.loadtxt('P4_SD_CVM.csv',
                                        delimiter=',', skiprows=1).T
    emg_slsa_vm, emg_slsa_vl = np.loadtxt('SLS_LIBFE.csv',
                                          delimiter=',', skiprows=1).T

    # Normalizar con respecto a contracción voluntaria máxima,
    # según artículo O.Valencia et al. 2021.
    fs = 1000  # Frecuencia de muestreo
    fc = 5  # Frecuencia de corte usada por el filtro en la normalización
    forden = 4  # Orden del filtro usado por el filtro en la normailzación
    emg_A, _, _ = ajusta_emg_func(emg_slsa_vm, emg_cvm_vm, fs, fc, forden)
    emg_B, _, _ = ajusta_emg_func(emg_slsa_vl, emg_cvm_vl, fs, fc, forden)

    ci = coactivation_index(emg_A, emg_B, fs)
    print(ci)

    plt.close('all')
    label_musc_A = 'Musculo A'
    label_musc_B = 'Musculo B'
    t = np.arange(0, len(emg_A) / fs, 1 / fs)

    fig, axs = plt.subplots()
    plot_coactivacion(axs, emg_A, emg_B, 'Músculo 1', 'Músculo 2', ci)
    plt.savefig('Coactivación.png')
    plt.show()
