# -*- coding: utf-8 -*-

"""Ajustando una señal electromiográfica funcional:

-Laboratorio Integrativo de Biomecánica y Fisiología del Esfuerzo,
Escuela de Kinesiología, Universidad de los Andes, Chile-
-Escuela de Ingeniería Biomédica, Universidad de Valparaíso, Chile-
        --Profesores: Oscar Valencia & Alejandro Weinstein--

"""
# Importar librerias
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def ajusta_emg_func(emg_fun, emg_cvm, fs, fc, filt_ord):
    """Ajusta EMG funcional según contracción voluntaria máxima.

    La función utiliza una señal EMG funcional y otra basada en la
    solicitación de una contracción isométrica voluntaria máxima. Ambas señales
    son procesadas considerando su centralización (eliminación de
    "offset"), rectificación y filtrado (pasa bajo con filtfilt).

    Parameters
    ----------
    emg_fun : array_like
        EMG funcional del músculo a evaluar
    emg_cvm : array_like
        EMG vinculada a la contracción voluntaria máxima del mismo músculo
    fs : float
       Frecuencia de muestreo, en hertz, de la señal EMG. Debe ser la misma
       para ambas señales.
    fc : float
        Frecuencia de corte, en hertz, del filtro pasa-bajos.
    filt_ord : int
        Orden del filtro pasa bajos

    Return
    ------
    emg_fun_norm : array_like
        EMG funcional filtrada y  normalizada
    emg_fun_env_f : array_like
        Envolvente de EMG funcional filtrada
    emg_cvm_envf_ : array_like
        Envolvente de EMG CVM filtrada
    """
    # Centralizando y rectificando las señales EMG
    emg_fun_env = abs(emg_fun - np.mean(emg_fun))
    emg_cvm_env = abs(emg_cvm - np.mean(emg_cvm))

    # Filtrado pasa-bajo de las señales
    b, a = butter(int(filt_ord), (int(fc)/(fs/2)), btype='low')
    emg_fun_env_f = filtfilt(b, a, emg_fun_env)
    emg_cvm_env_f = filtfilt(b, a, emg_cvm_env)

    # Calculando el valor máximo de emg_cvm y ajustando la señal EMG funcional
    emg_cvm_I = np.max(emg_cvm_env_f)
    emg_fun_norm = (emg_fun_env_f / emg_cvm_I) * 100

    return emg_fun_norm, emg_fun_env_f, emg_cvm_env_f
