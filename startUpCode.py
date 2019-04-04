#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:17:22 2019

@author: SoheilaPC
"""
#    This file is part of MOO_Toolbox.
#
#    MOO_Toolbox is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    MOO_Toolbox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with MOO_Toolbox. If not, see <http://www.gnu.org/licenses/>.
from openmdao.api import Problem, ScipyOptimizeDriver, IndepVarComp,ExecComp, ExplicitComponent
import seaborn,pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import  savefig
from generic6 import ga_generic
#%%
#how to use toolbox?

typeOfInputs  = ['float','float'] # enter types of your variables:bool if it is boolean, float if it float and int if it is integer.
Ngen=100 #select number of generation
A = [] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = []
Aeq = []   #matrix for equality linear constraint
beq = []
lb = [0] + [0]   #lower bound and upper bounds of variables.
ub = [5] + [3]
obj=['min','max']    #define your objectives:min if you want minimize the function ans max
                         #if you want to maximize the function.
def nonlcons(x):       # you need t define the inequality nonlinear constraints. consider g1<0 and write  g1=...
    x1=x[0]
    y=x[1]
    g1=
    g2=
    return g1,g2
def eqnonlcons(x):     # you need t define the equality nonlinear constraints.
    ceq =
    return ceq
def myfun(x):          # you need define your main function here.
    x1=x[0]
    y=x[1]
    f1=
    f2=
    return  f1,f2

my_ga = ga_generic()   # for setup the 4 first is manadatory and the last 4 are optional if you have them use as you see.
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub, A=A,b=b,Aeq=Aeq,beq=beq, nonlcons=nonlcons)


# plot the pareto front
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2']
seaborn.pairplot(df, diag_kind="kde")

 
   
    





#%% this is an example that can help you to fill the information according your problem
#exampl 19
"""
typeOfInputs  = ['float','float']
Ngen=100
A = [[-9,-1],[-9,1]] 
b = [-6,-1]
Aeq = []
beq = []
lb = [0.1,0] 
ub = [1,5] 
obj=['min','min']
def nonlcons(x):
   
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    x2=x[1]
    f1=x1
    f2=x2+1/x1
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub, A=A,b=b)
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('f1')
plt.ylabel('f2')
savefig('2lin.png', format='png', dpi=1000)
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2']
seaborn.pairplot(df, diag_kind="kde")
"""


