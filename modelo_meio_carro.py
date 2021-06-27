#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este programa implementa um modelo de meio carro.

---------------------------
\           mb            \
---------------------------
 |                        |
----                    ----
\mw\                    \mw\
----                    ----
 \                        \
 o                        o

Admite-se que a massa do chassis tem dois graus de liberdade: translação em z 
(vertical) e rotação ao redor de y (arfagem). As rodas podem apenas deslocar-se
verticalmente.

A entrada do sistema consiste de deslocamentos nos pontos inferiores das rodas.

Vetor de saída da solução:
    x1 - deslocamento vertical de mb
    x2 - arfagem de mb
    x3 - deslocamento vertical da roda 1
    x4 - deslocamento vertical da roda 2
    x5 - velocidade vertical de mb
    x6 - velocidade de arfagem de mb
    x7 - velocidade vertical da roda 1
    x8 - velocidade vertical da roda 2

@author: Leonardo Bartalini Baruffaldi
"""

import numpy as np
from numpy import matrix, sin, cos, pi, clip
from numpy import eye, zeros, block
from numpy.linalg import inv, solve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import celluloid


'''
PARÂMETROS (mudar os valores aqui para ver os resultados)
''' 

# dados do chassis
mb    = 130.0000           # massa suspensa                 kg
Ib    =  54.0000           # momento inércia susp.          kg.m² 
# massa não suspensa
mw    =  15.0000           # massa não suspensa             kg
# pneu
kt    = 200.00e3           # rigidez pneu                   N/m
# suspensão
ks    = 120.00e3           # rigidez da mola                N/m
cs    = 100.0000           # cof. amortecimento             N.s/m 
# entreeixos
L     =   1.3000           # entreeixos                     m
L1    =   0.9000           # dist. roda 1 até cg chassis    m
L2    =   L-L1             # dist. roda 2 até cg chassis    m
# gravidade
g     =   9.8100           # gravidade                      m/s²
# entradas da simulação
# as excitações são dadas por funções senoidais defasadas de lr/v segundos
lr    =   1.5000           # comprimento de onda da estrada m
v     =  11.0000           # velocidade do veículo          m/s
af    =   0.0100           # amplitude das oscilações       m
# tempo total de simulação
tend  =   5.0000           # tempo total de simulação       s
 


'''
 EQUAÇÕES DE MOVIMENTO
'''

# matriz de massa
M = eye(8)
M[4,4] = mb
M[5,5] = Ib
M[6,6] = mw
M[7,7] = mw

Minv = inv(M)

# matriz de rigidez linearizada (pequenas rotações)

K = matrix([[-2*ks       , L1-L2              ,ks             ,ks       ],
            [ks*(L1-L2)  , -(L1**2+L2**2)*ks  ,-L1*ks         ,L2*ks],
            [ks          , -L1                ,-ks-kt         ,0.00    ],
            [ks          , L2                 ,0              ,-ks-kt  ]])

# matriz de amortecimento linearizadas (pequenas rotações)

C = matrix([[-2*cs       , L1-L2             ,cs                  ,cs       ],
            [cs*(L1-L2)  , -(L1**2+L2**2)*cs ,-L1*cs             ,L2*cs],
            [cs          , -L1               ,-cs                ,0.00    ],
            [cs          , L2                ,0                  ,-cs     ]])


'''
FUNÇÕES PARA SIMULAÇÃO
'''
def zext(t):
    '''
    Função de excitação externa

    Parameters
    ----------
    t : tempo

    Returns
    -------
    the displacement at the wheels

    '''
    
    # cálculo da frequência da senoide
    wf = 2 * pi * v / lr
    
    zext = zeros(8)
    zext[6] = af * sin(wf * t)
    zext[7] = af * sin(wf * clip(t - lr / v,0,1e8))
    
    return zext


def eqmovlin(t, x):
    '''
    Esta função define as equações diferenciais linearizadas do sistema

    Parameters
    ----------
    t : tempo
    x : variáveis de estado

    Returns
    -------
    Vetor da derivada dos estados

    '''
    A = block([[zeros((4,4)),eye(4)],
           [K,C]])
    
    
    grav = zeros(8)
    grav[4] = g
    grav[6:7] = g
    
    dxdt = np.dot(Minv,(np.dot(A,x) + kt * zext(t)).T).T + grav
    return dxdt.A1


def eqmovnl(t, x):
    '''
    Esta função define as equações diferenciais NÃO linearizadas do sistema

    Parameters
    ----------
    t : vetor de tempo
    x : variáveis de estado

    Returns
    -------
    Vetor da derivada dos estados

    '''
   
    # MATRIZES DE RIGIDEZ E MASSA NAO LINEARES
    Knl = K
    Knl[1] = Knl[1] * cos(x[1])
   
    Cnl = C
    Cnl[1] = Cnl[1] * cos(x[1])
   
    A = Minv * block([[zeros((4,4)),eye(4)],
           [Knl,Cnl]])
    
    # vetor de campo gravitacional
    grav = zeros(8)
    grav[4] = g
    grav[6:7] = g
    
    # x é o vetor de variáveis de estado
    # y é a modificação necessária para o formado não-linear das equações
    y = x
    y[1] = sin(x[1])
    y[5] = x[5]*cos(x[1])
    
    dxdt = np.dot(A,y) + np.dot(Minv,kt * zext(t)) + grav
    return dxdt.A1



'''
CÁLCULO DAS POSIÇÕES DE EQUILÍBRIO
'''
# o cálculo é feito com base nas equações de movimento quando as acelerações 
# e velocidades são nulas
xeq = solve(K, -matrix([[mb*g,0,mw*g,mw*g]]).T)
xeq = block([[xeq],[0*xeq]])


# solução das equações de movimento linearizadas
# esta solução não é usada no restante do programa. Descomente caso queira os
# resultados
''' sollin = solve_ivp(eqmovlin,[0,tend],xeq.A1,
                   t_eval=np.linspace(0,tend,int(200*tend)))'''

# solução das equações de movimento não linearizadas
solnl = solve_ivp(eqmovnl,[0,tend],xeq.A1)


# gráficos da solução não linear
plt.subplot(3,1,1)
plt.plot(solnl.t,solnl.y[0]*1e3)
plt.ylabel('Desl. corpo [mm]')
plt.grid()

plt.subplot(3,1,2)
plt.plot(solnl.t,solnl.y[1])
plt.ylabel('Desl. corpo [rad]')
plt.grid()

plt.subplot(3,1,3)
plt.plot(solnl.t,solnl.y[2]*1000)
plt.plot(solnl.t,solnl.y[3]*1000)
plt.ylabel('Desl. roda [mm]')
plt.grid()



# animação com base na solução não linear
fig = plt.figure()
ax = plt.gca()
plt.xlim(-1,1)
ax.set_aspect('equal')
plt.grid()
cam = celluloid.Camera(fig)
for i in range(solnl.t.size):
    corpo = Rectangle((-L/2,.5+solnl.y[0,i]),L,.15,
                      angle=-solnl.y[1,i]*180/pi,
                      linewidth=1,
                      edgecolor='r')
    roda1 = Circle((L/2,0.25+solnl.y[2,i]),radius=0.1,
                   linewidth=1,
                   facecolor='g')
    roda2 = Circle((-L/2,0.25+solnl.y[3,i]),radius=0.1,
                   linewidth=1,
                   facecolor='g')
    ax.add_patch(corpo)
    ax.add_patch(roda1)
    ax.add_patch(roda2)
    ax.text(0,0.8,'tempo = {:1.3f} s \ndistância = {:1.1f} m'
            .format(solnl.t[i],solnl.t[i]*v))
    cam.snap()
animation = cam.animate()
    