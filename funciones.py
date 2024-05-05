# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:02:12 2024

@author: Ignacio
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def importar_data(a):
    x = np.loadtxt(a,skiprows=1,) #salteamos titulos
    return x


def plot_transferencia(Vbb,Vee,Ic,Ib,titulo,minimo,maximo):
    """
    Parameters
    ----------
    Vbb : float array
        tensión base.
    Vee :float array
        V emisor. - es cosntante acá
    Ic : float array
        I colector.
    Ib : float array
        I base.
    titulo : string
        titulo del grafico.
    minimo : float
        minimo para intervalo de regresión.
    maximo : float 
        max para intervalo de regresión.

    Returns
    -------
    None.

    """
    Vbe=Vbb-Vee #tensión entre base y emisor es la diferencia entre Vbb y Vee, que es 0
    
    
    #obtengo logaritmos de las corrientes
    log_Ic=np.log(np.abs(Ic))
    log_Ib=np.log(np.abs(Ib))
    
    #trunqueo los semilogs para hacer la regresión entre mínimo y máximo
    indicesAjuste = np.where((Vbe > minimo) & (Vbe < maximo))
    log_Ic_trunq=log_Ic[indicesAjuste]
    log_Ib_trunq=log_Ib[indicesAjuste]
    Vbe_trunq=Vbe[indicesAjuste]
    
    #calculamos las pendientes y ordenadas al origen
    coefAjusteIc = np.polyfit(Vbe_trunq, log_Ic_trunq, deg=1) # Ajusto una recta y obtengo los coeficientes
    coefAjusteIb = np.polyfit(Vbe_trunq, log_Ib_trunq, deg=1) # Ajusto una recta y obtengo los coeficientes
    Ic_m = coefAjusteIc[0]
    Ic_b = coefAjusteIc[1]
    Ib_m = coefAjusteIb[0]
    Ib_b = coefAjusteIb[1]

    #como la regresión se hace con el logaritmo natural y la I sat es la ordenada de la regresión, se exponencia y se obtiene la Isat real
    Isat = pow(math.e,Ic_b)
    
    k = 1.3806e-23 # [J/K] Constante del Boltzmann
    q = 1.60223e-19 # [C] Carga del electron
    T = 300 # [K] Temperatura de trabajo
    vth = k*T/q # [V] Tensión termica
    
    #creo las rectas de regresión para Ic e Ib
    reg_Ic = []
    reg_Ib = []
    for x in Vbe:
        reg_Ic.append(Ic_m*x+Ic_b)
        reg_Ib.append(Ib_m*x+Ib_b)

    #estas regresiones las paso a arrays y als exponencia porque quedan en escala lineal y no logaritmica
    reg_Ic= np.exp(np.asarray(reg_Ic))
    reg_Ib=np.exp(np.asarray(reg_Ib))
    plt.figure()
    plt.figure(figsize=(9.5,3),dpi=1000)
    plt.yscale('log')
    #plt.semilogy(Vbe,np.abs(Ic),label="Ic",marker='o',linestyle='-')
    #cuando usar el yscale en log le pasas el valor abs de los valores y ya crea el logaritmo
    plt.scatter(Vbe, np.abs(Ic),s = 0.2,label="log(Ic)")
    
    #plt.semilogy(Vbe,np.abs(Ib),label="Ib")
    plt.scatter(Vbe, np.abs(Ib),s = 0.2,label="log(Ib)")
    plt.plot(Vbe,reg_Ic,linestyle="--",label="regresión de log(Ic)",color='red')
    plt.plot(Vbe,reg_Ib,linestyle="--",label="regresión de log(Ib)",color='green')
    plt.grid()
    plt.legend(["Logaritmo de muestras de Ic","Logaritmo de muestras de Ib",f'Regresión de Ic m={Ic_m:.2f} & b={Ic_b:.2f}',f'Regresión de Ib m={Ib_m:.2f} & b={Ib_b:.2f}'])
    plt.annotate(f'1/Vth={(1/vth):.2f}/V \nIntervalo de regresión:({minimo:.2f}:{maximo:.2f}) V\nIsat={Isat:.3e} A',[-0.797,pow(10,-7.9)])

    plt.ylabel('Corriente [A]')
    plt.xlabel('Tensión Vbe [V]')
    #plt.ylim([1e-15, 1]) #Esto me permite controlar los limites del eje y
    plt.show()
    
def plot_beta(Vbb,Vee,Ic,Ib,titulo,minimo,maximo):
    beta=Ic/Ib
    Vbe=Vbb-Vee
    indicesAjuste = np.where((Vbe > minimo) & (Vbe < maximo))
    beta_reg = beta[indicesAjuste]
    Vbe_reg = Vbe[indicesAjuste]
    coefAjuste = np.polyfit(Vbe_reg, beta_reg, deg=0) # Ajusto una recta y obtengo los coeficientes
    #print(coefAjuste)
    #veamos la cte que devolvió
    b=coefAjuste[0]

    #creo datos de regresión, osea una constante
    reg = []
    for x in Vbe:
        reg.append(b)
    
    plt.figure()
    plt.figure(figsize=(9.5,3),dpi=1000)
    plt.scatter(Vbe, beta,s=0.3,zorder=1)
    plt.scatter(Vbe[indicesAjuste],beta[indicesAjuste],s=0.3,zorder=2)
    plt.plot(Vbe,reg,linestyle="--",color='red',zorder=0)
    plt.legend(["Beta","Beta entre límites de regresión",f'Regresión: Bf = {b:.2f}'])
    plt.ylabel('Beta')
    plt.xlabel('Tensión Vbe [V]')
    plt.grid()
    plt.show()
    
    
def plot_salida(Vbb,Vcc,Ib,Ic,titulo,minimo,maximo):
    
    #ploteariamos Ic contra VCE, Vee = 0 asio que VCE es Vcc
    Vce = np.copy(Vcc)
    #buscamos Va, que sale de una regresión entre min y max y Vce sat que quiero hacerlo por el punto de max diferencia entre valores de ic

    diferencias_Ic = np.copy(Ic)
    #esta m*erda estaba haciendo algo raro, como asociar los arrays por puntero entonces cuando se modificaba el de diferencias cambiaba Ic
        
    for x in range(0,len(diferencias_Ic)-1):
        diferencias_Ic[x]=diferencias_Ic[x]-diferencias_Ic[x+1]

    #la diferencia máxima es el punto de máxima derivada
    index_min=np.argmin(diferencias_Ic)
    Vce_sat = Vce[index_min]
    diferencias_Ic = diferencias_Ic[:len(Vce)] #por si no son de igual tamaño
    
    
    #Ahora quiero la regresión de la Ic hasta x=0 para obtener la V de early
    
    indicesAjuste = np.where((Vce > minimo) & (Vce < maximo))
    coefAjuste = np.polyfit(Vce[indicesAjuste],Ic[indicesAjuste], deg=1) # Ajusto una recta y obtengo los coeficientes

    reg_ic = []
    for x in Vce:
        reg_ic.append(1000*(x*coefAjuste[0]+coefAjuste[1]))
    
    Va = -coefAjuste[1]/coefAjuste[0]
    
    print(Va)
    
    #IC *1000 y VA A ESTAR EN MILIAMPERES PQ ES CHIQUITA
    plt.figure()
    plt.figure(figsize=(9.5,3),dpi=1000)
    plt.scatter(Vce, 1000*Ic,s = 0.5,label="Ic")
    plt.scatter(Vce,1000*diferencias_Ic,s = 0.5, label="diferencias",color='green')
    plt.plot(Vce,reg_ic,linestyle="--",zorder=0,color='red')
    plt.grid()
    f_inverse = "f\u207B\u00B9"
    plt.legend(["Muestras de Ic",f'Diferencias entre muestras de Ic con mínimo en Vce_sat={Vce_sat:.2f} V',f'Regresión entre ({minimo:.2f}{maximo:.2f}) V & {f_inverse}(0)=Va={Va:.2f} V '])
    #plt.annotate(f'Vce Sat={Vce_sat:.2f}',[-0.797,pow(10,-7.9)])
    plt.ylabel('Corriente [mA]')
    plt.xlabel('Tension Vce [V]')
    plt.show()
    
    
    
    
    