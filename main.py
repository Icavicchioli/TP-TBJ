# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:01:37 2024

@author: Ignacio
"""

import funciones as f


datos_salida = f.importar_data("./salida.txt")

datos_transferencia = f.importar_data("./transferencia 2.txt")



salida = {
    "VCE": datos_salida[:,0],   
    "Vbb": datos_salida[:,1],
    "Vcc": datos_salida[:,2],
    "Ib": datos_salida[:,5],
    "Ic": datos_salida[:,6],
    }

transferencia = {
    "Vbb": datos_transferencia[:,1],
    "Vcc": datos_transferencia[:,2],
    "Ib": datos_transferencia[:,5],
    "Ic": datos_transferencia[:,6],
    }

#tenemos arrays de datos para laburar

f.plot_transferencia(transferencia["Vbb"],0,transferencia["Ic"],transferencia["Ib"],"Curva de transferencia",-0.6,-0.5)

f.plot_beta(transferencia["Vbb"],0,transferencia["Ic"],transferencia["Ib"],"Curva de beta",-0.6,-0.5)

f.plot_salida(salida.get("Vbb"),salida.get("Vcc"),salida.get("Ib"),salida.get("Ic"),"Curva de salida",-4,-1)
