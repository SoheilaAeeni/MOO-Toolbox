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
#exampl 1
#linear inequality constraint
"""
typeOfInputs  = ['float','float']
A = [1,2] #matrix for inequality linear constraint
b = [1]
#Aeq = [2,1]
#beq = [1]
lb = [0.0] + [0.0] 
ub = [1.0] + [1.0]
obj=['min']
def nonlcons(x):
    c[0] = (x[0]**2)/9 + (x[1]**2)/4 - 1
    c[1] = x[0]**2 - x[1] - 1
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    x2=x[1]
    z=100*(x2-x1**2)**2 + (1-x1)**2
    return  z

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub, A = A,b = b,)
pop, logbook = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
#seed
#prob diff method
#succesfull with getfunky

"""
#%%
#exampl 2
""""
#2linear inequality constraint


typeOfInputs  = ['float','float']
A = [[1,1],[2,1]] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = [1,2]
Aeq = []
beq = []
lb = [0.0] + [0.0] 
ub = [1.0] + [1.0]
obj=['max']
def nonlcons(x):
    c[0] = (x[0]**2)/9 + (x[1]**2)/4 - 1
    c[1] = x[0]**2 - x[1] - 1
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    z=np.exp(x1)*np.sin(y)
    return  z

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub, A = A,b = b,)
my_ga.main()
"""


#%%
#exampl 3
"""
#1linear equality constraint


typeOfInputs  = ['float','float']
A = [] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = []
Aeq = [2,1]
beq = [2]
lb = [0.0] + [0.0] 
ub = [2.0] + [2.0]
obj=['max']
def nonlcons(x):
    c[0] = (x[0]**2)/9 + (x[1]**2)/4 - 1
    c[1] = x[0]**2 - x[1] - 1
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    z=np.exp(x1)*np.sin(y)
    return  z

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub, Aeq = Aeq,beq = beq)
my_ga.main()
"""
#%%
#exampl 4
#2 linear equality constraint
"""
typeOfInputs  = ['int','int']
A = [] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = []
Aeq = [[2,1],[1,1]]
beq = [2,-7]
lb = [-20] +[-20]
ub = [20] +[20]
obj=['min']
def nonlcons(x):
    c[0] = (x[0]**2)/9 + (x[1]**2)/4 - 1
    c[1] = x[0]**2 - x[1] - 1
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    z=np.exp(x1)*np.sin(y)
    return  z

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,lowerB = lb, upperB = ub,Aeq = Aeq,beq = beq)
my_ga.main()
"""
#%%
#exampl 5
"""
# nonlinear ineq constraint


typeOfInputs  = ['float','float','float']
A = [] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = []
Aeq = []
beq = []
lb = [-20] + [-10] +[-10]
ub = [20] + [20]+ [10]
obj=['max']
def nonlcons(x):
    x1=x[0]
    y=x[1]
    z=x[2]
    c=x1**2+2*y**2+3*z**2-1
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    z=x[2]
    p=x1*y*z
    return  p

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,lowerB = lb, upperB = ub,nonlcons=nonlcons)
my_ga.main()
"""
#%%
#exampl 6
"""
# 2 nonlinear ineq constraint


typeOfInputs  = ['float','float','float']
A = [] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = []
Aeq = []
beq = []
lb = [-20] + [-10] +[-10]
ub = [20] + [20]+ [10]
obj=['max']
def nonlcons(x):
    x1=x[0]
    y=x[1]
    z=x[2]
    c=x1**2+2*y**2+3*z**2-1
    d=x1**2+y**2-0.25
    return c,d
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    z=x[2]
    p=x1*y*z
    return  p

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,lowerB = lb, upperB = ub,nonlcons=nonlcons)
my_ga.main()
"""

#%%
#exampl 7
#boolean
"""

typeOfInputs  = ['bool' for i in range(100)]
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0 for i in range(100)] 
ub = [100 for i in range(100)]
obj=['max','min']
def nonlcons(x):
    return c
def eqnonlcons(x):
    return ceq
def myfun(individual):
    return sum(individual), sum(individual)

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub)
pop, logbook = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
df = pandas.DataFrame(data=front)
seaborn.pairplot(df, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None, size=None)
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
"""
#%%
#exampl 8
#vol-stress-hemo inds-nonlinear cons
"""

typeOfInputs  = ['float','float','float']
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0.0001] + [0.0001]+ [1]
ub = [0.01] + [0.01]+ [3]
obj=['min','min','min']
def nonlcons(individual):
    x1=individual[0]
    x2=individual[1]
    y=individual[2]
    stress_ac=20000*np.sqrt(16+y**2)/(y*x1)
    stress_bc=80000*np.sqrt(1+y**2)/(y*x2)
    stress=max(stress_ac, stress_bc)
    c=stress-1e8
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    x1=individual[0]
    x2=individual[1]
    y=individual[2]
    volume= np.sqrt(16+y**2)*x1+np.sqrt(1+y**2)*x2
    stress_ac=20000*np.sqrt(16+y**2)/(y*x1)
    stress_bc=80000*np.sqrt(1+y**2)/(y*x2)
    stress=max(stress_ac, stress_bc)
    
    return  volume,stress

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub, A = A,b = b,nonlcons=nonlcons)
pop, logbook = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['volume','stress']
seaborn.pairplot(df, diag_kind="kde")

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(x, y, z, label='parametric curve')
#ax.legend()

#plt.show()
"""
#%%
#exampl 9
#vol-stress-z and 2 kind individual-float and integer
"""
typeOfInputs  = ['float','float','int','float']
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0.0001] + [0.0001]+ [1]+[0]
ub = [0.01] + [0.01]+ [3]+[np.pi/2]
obj=['min','min','min']
def nonlcons(individual):
    x1=individual[0]
    x2=individual[1]
    y=individual[2]
    stress_ac=20000*np.sqrt(16+y**2)/(y*x1)
    stress_bc=80000*np.sqrt(1+y**2)/(y*x2)
    stress=max(stress_ac, stress_bc)
    c=stress-1e8
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    x1=individual[0]
    x2=individual[1]
    y=individual[2]
    x3=individual[3]
    volume= np.sqrt(16+y**2)*x1+np.sqrt(1+y**2)*x2
    stress_ac=20000*np.sqrt(16+y**2)/(y*x1)
    stress_bc=80000*np.sqrt(1+y**2)/(y*x2)
    stress=max(stress_ac, stress_bc)
    z=400*(np.sin(x3)+np.sin(x3)*np.cos(x3))
    
    return  volume,stress,-z

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub, A = A,b = b,nonlcons=nonlcons)
pop, logbook = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['volume','stress','-z']
seaborn.pairplot(df, diag_kind="kde")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(df['volume'], df['stress'], df['-z'])
ax.legend()
plt.show()
df2 = df[df['-z'] < -480]
seaborn.pairplot(df2, diag_kind="kde")
"""
#%%
#exampl 10
"""
#clone 2float
typeOfInputs  = ['float','float']
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0]+[0]
ub = [10] + [20]
obj=['min','min']
def nonlcons(individual):
    r=individual[0]
    h=individual[1]
    c=-(np.pi/3.0)*r*r*h+200
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    r=individual[0]
    h=individual[1]
    
    lateral= np.pi * r * np.sqrt(r*r+h*h)
    total=  np.pi*r*(r+ np.sqrt(r*r+h*h))
    return  lateral, total


my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub, A = A,b = b,nonlcons=nonlcons)
pop, logbook = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['lateral','total']
seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#vol-stress-z and 2 kind inds
#exampl 11
"""
typeOfInputs  = ['float','float','int']
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0.0001] + [0.0001]+ [1]
ub = [0.01] + [0.01]+ [3]
obj=['min','min','min']
def nonlcons(individual):
    x1=individual[0]
    x2=individual[1]
    y=individual[2]
    stress_ac=20000*np.sqrt(16+y**2)/(y*x1)
    stress_bc=80000*np.sqrt(1+y**2)/(y*x2)
    stress=max(stress_ac, stress_bc)
    c=stress-1e8
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    x1=individual[0]
    x2=individual[1]
    y=individual[2]
    volume= np.sqrt(16+y**2)*x1+np.sqrt(1+y**2)*x2
    stress_ac=20000*np.sqrt(16+y**2)/(y*x1)
    stress_bc=80000*np.sqrt(1+y**2)/(y*x2)
    stress=max(stress_ac, stress_bc)
    
    return  volume,stress

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj, lowerB = lb, upperB = ub,nonlcons=nonlcons)
pop, logbook = my_ga.main()
#indReal = my_ga.deBinerize(ind)
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['volume','stress']
seaborn.pairplot(df, diag_kind="kde")

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(df['volume'], df['stress'], df['z'])
#ax.legend()
#plt.show()
"""
#%%
#exampl 12
""""
# 2 nonlinear ineq constraint
typeOfInputs  = ['float','int','float']
A = [] #matrix for inequality linear constraint
                    #example:if you have one linear constraint like 2x+y<1
                    #A=[2,1] and b=[1] 
                    #if you have 2 linear constraint like 2x+y<1 and 3x+6y<3
                    #A=[[2,1],[3,6]] and b=[1,3]
b = []
Aeq = []
beq = []
lb = [-20] + [-10] +[-10]
ub = [20] + [20]+ [10]
obj=['max']
def nonlcons(x):
    x1=x[0]
    y=x[1]
    z=x[2]
    c=x1**2+2*y**2+3*z**2-1
    d=x1**2+y**2-0.25
    return c,d
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    z=x[2]
    p=x1*y*z
    return  p

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,lowerB = lb, upperB = ub)
pop, logbook,ind = my_ga.main()
indReal = my_ga.deBinerize(ind)
#print(indReal)
#pop, logbook = my_ga.main()
"""
#%%
#exampl 13
#Binh and Korn function:
"""
typeOfInputs  = ['float','float']
Ngen=20
A = [] 
b = []
Aeq = []
beq = []
lb = [0] + [0] 
ub = [5] + [3]
obj=['min','min']
def nonlcons(x):
    x1=x[0]
    y=x[1]
    g1=(x1-5)**2+(y)**2-25
    g2=-(x1-8)**2-(y+3)**2+7.7
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    f1=4*x1**2+4*y**2
    f2=(x1-5)**2+(y-5)**2
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub, nonlcons=nonlcons)
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2']
seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#exampl 14
"""
#Chakong and Haimes function:
Ngen=100
typeOfInputs  = ['int','float']
A = [[1,-3]]
b = [-10]
Aeq = []
beq = []
lb = [-20] + [-20] 
ub = [20] + [20]
obj=['min','min']
def nonlcons(x):
    x1=x[0]
    y=x[1]
    g1=(x1)**2+(y)**2-225
    return g1
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    y=x[1]
    f1=2+(x1-2)**2+(y-1)**2
    f2=9*x1-((y-1)**2)
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub,A=A,B=b, nonlcons=nonlcons)
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('f1')
plt.ylabel('f2')
savefig('heter.png', format='png', dpi=1000)
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2']
seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#exampl 15
"""
typeOfInputs  = ['float','float','float','float','float','float']
Ngen=500
A = [[-1,-1,0,0,0,0],[1,1,0,0,0,0],[-1,1,0,0,0,0],[1,-3,0,0,0,0]]
b = [-2,6,2,2]
Aeq = []
beq = []
lb = [0] + [0] +[1]+[0]+[1]+[0]
ub = [10] + [10]+[5]+[6]+[5]+[10]
obj=['min','min']
def nonlcons(x):
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    g5=(x3-3)**2+x4-4
    g6=-(x5-3)**2-x6+4
    return g5,g6
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    f1=-25*(x1-2)**2-(x2-2)**2-(x3-1)**2-(x4-4)**2-(x5-1)**2
    f2=x1**2+x2**2+x3**2+x4**2+x5**2+x6**2
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub, A=A,b=b,nonlcons=nonlcons)
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('f1')
plt.ylabel('f2')
savefig('both.png', format='png', dpi=1000)
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2']
seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#exampl 17
#bridging with OPenMDAO
"""
typeOfInputs  = ['float','float']

A = [] 
b = []
Aeq = []
beq = []
lb = [-50] + [-50] 
ub = [50] + [50]
obj=['min']
Ngen=10
def nonlcons(x):
   
    return g1,g2,
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp, IndepVarComp
    prob = Problem()
    indeps = prob.model.add_subsystem('indeps', IndepVarComp())
    indeps.add_output('x', x[0])
    indeps.add_output('y', x[1])

    prob.model.add_subsystem('paraboloid', ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

    prob.model.connect('indeps.x', 'paraboloid.x')
    prob.model.connect('indeps.y', 'paraboloid.y')
    prob.setup()
    prob.run_model()
    return(prob['paraboloid.f'][0])

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub)
pop, logbook,ind = my_ga.main()
"""
#%%
#exampl 18 with OpenMDAO
"""
typeOfInputs  = ['float','float']
Ngen=10
A = [] 
Aeq = []
beq = []
lb = [-20] + [-20]
ub = [20] + [20]
obj=['min','min']
def nonlcons(x):
    x1=x[0]
    x2=x[1]
    g1=(x1)**2+(x2)**2-225
    g2=x1-3*x2+10
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    class RectangleComp(ExplicitComponent):
        def setup(self):
            self.add_input('x1')
            self.add_input('x2')
            self.add_output('f1')
            self.add_output('f2')
        def compute(self, inputs, outputs):
            outputs['f1'] = 2+(inputs['x1']-2)**2+(inputs['x2']-1)**2
            outputs['f2'] = 9*inputs['x1']- ((inputs['x2']-1)**2)
            # build the model
    prob = Problem()
    indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
    indeps.add_output('x1',x[0])
    indeps.add_output('x2',x[1])   
    prob.model.add_subsystem('a', RectangleComp(),
                            promotes_inputs=['x1', 'x2'])

    prob.setup()
    prob.run_model()
    return(prob['a.f1'][0],prob['a.f2'][0])

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen,lowerB = lb, upperB = ub, nonlcons=nonlcons)
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2']
seaborn.pairplot(df, diag_kind="kde")
"""

#%%
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
#%%
#exampl 20
"""
typeOfInputs  = ['float','float']
Ngen=100
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [-3] + [-3]
ub = [3] + [3]
obj=['min','min','min']
def nonlcons(individual):
   
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    x1=individual[0]
    x2=individual[1]
    f1=0.5*(x1**2+x2**2)+np.sin(x1**2+x2**2)
    f2=(3*x1-2*x2+4)**2/8 + (x1-x2+1)**2/27 + 15
    f3=1/(x1**2+x2**2+1) - 1.1*np.exp(-(x1**2+x2**2))
    
    return  f1,f2,f3

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen, lowerB = lb, upperB = ub,)
pop, logbook,ind = my_ga.main()
#indReal = my_ga.deBinerize(ind)
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('weight')
plt.ylabel('L/D')
savefig('LD.png', format='png', dpi=1000)
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f1','f2','f3']# should be order like the return from my fun
#seaborn.pairplot(df, diag_kind="kde")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(df['f2'], df['f1'], df['f3'])
ax.legend()
plt.show()
"""
#%%
#example 21
"""
typeOfInputs  = ['float' for i in range(30)]
Ngen=100
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0 for i in range(30)]
ub = [1 for i in range(30)]
obj=['min','min','min','min']
def nonlcons(individual):
   
    return c
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):    
    f1=individual[0]
    g=1+(9/29)*(sum(individual[1:]))
    h=1-np.sqrt(f1/g)-(f1/g)*np.sin(10*np.pi*f1)
    f2=g*h
    
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen, lowerB = lb, upperB = ub)
pop, logbook,ind = my_ga.main()
#indReal = my_ga.deBinerize(ind)
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.show()
df = pandas.DataFrame(data=front)
df.columns=['f2','f1']
#seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#exampl 22
"""
typeOfInputs  = ['float','float']
Ngen=1000
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [0] + [0]
ub = [1] + [1]
obj=['min','min']
def nonlcons(individual):
    x1=individual[0]
    x2=individual[1]
    f1=x1
    f2=(1+x2)*np.exp(-x1/(1+x2))
    g1=1-f2/(0.858*np.exp(-0.541*f1))
    g2=1-f2/(0.728*np.exp(-0.295*f1))
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    x1=individual[0]
    x2=individual[1]
    f1=x1
    f2=(1+x2)*np.exp(-x1/(1+x2))
    
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen, lowerB = lb, upperB = ub, nonlcons=nonlcons)
pop, logbook,ind = my_ga.main()
#indReal = my_ga.deBinerize(ind)
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('f1')
plt.ylabel('f2')
savefig('22.png', format='png', dpi=1000)
plt.show()
#df = pandas.DataFrame(data=front)
#df.columns=['f1','f2','f3']# should be order like the return from my fun
#seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#exampl 23
"""
typeOfInputs  = ['float']
Ngen=1000
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [-5] 
ub = [10]
obj=['min','min']
def nonlcons(individual):
    
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(individual):
    x1=individual[0]
    if x1<1:
        f1=-x1
    elif x1>1 and x1<=3:
        f1=x1-2
    elif x1>3 and x1<=4:
        f1=4-x1
    else:
        f1=x1-4
    f2=(x1-5)**2
    
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen, lowerB = lb, upperB = ub)
pop, logbook,ind = my_ga.main()
#indReal = my_ga.deBinerize(ind)
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('f1')
plt.ylabel('f2')
savefig('22.png', format='png', dpi=1000)
plt.show()
#df = pandas.DataFrame(data=front)
#df.columns=['f1','f2','f3']# should be order like the return from my fun
#seaborn.pairplot(df, diag_kind="kde")
"""
#%%
#exampl 24

typeOfInputs  = ['float','float','float']
Ngen=100
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [-5,-5,-5] 
ub = [10,10,10]
obj=['min','min']
def nonlcons(individual):
    
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    f1 = [-10*np.exp(-0.2*np.sqrt((x[i])**2+(x[i+1])**2)) for i in range(2)]
    f1 = sum(f1)
    f2=[(np.abs(x[i]))**0.8+5*np.sin((x[i]**3)) for i in range(3)]
    f2=sum(f2)
    return  f1,f2

my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen, lowerB = lb, upperB = ub)
pop, logbook,ind = my_ga.main()
front = np.array([ind.fitness.values for ind in pop])
plt.scatter(front[:,0], front[:,1], c="b")
plt.axis("tight")
plt.xlabel('f1')
plt.ylabel('f2')
savefig('22.png', format='png', dpi=1000)
plt.show()
#%%
#exampl 25
"""
typeOfInputs  = ['float','float']
Ngen=100
A = [] #matrix for inequality linear constraint
b = []
Aeq = []
beq = []
lb = [-50,-50] 
ub = [50,50]
obj=['min']
def nonlcons(individual):
    
    return g1,g2
def eqnonlcons(x):
    ceq = []
    return ceq
def myfun(x):
    prob = Problem()
    indeps = prob.model.add_subsystem('indeps', IndepVarComp())
    indeps.add_output('x', x[0])
    indeps.add_output('y', x[1])
    prob.model.add_subsystem('paraboloid', ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))
    prob.model.connect('indeps.x', 'paraboloid.x')
    prob.model.connect('indeps.y', 'paraboloid.y')
    prob.setup()
    prob.run_model()
    return  (prob['paraboloid.f'][0])
my_ga = ga_generic()
my_ga.setup(myfun, typeOfInputs,obj,Ngen=Ngen, lowerB = lb, upperB = ub)
pop, logbook,ind = my_ga.main()

#indReal = my_ga.deBinerize(ind)
#front = np.array([ind.fitness.values for ind in pop])
#plt.scatter(front[:,0], front[:,1], c="b")
#plt.axis("tight")
#plt.xlabel('f1')
#plt.ylabel('f2')
#savefig('22.png', format='png', dpi=1000)
#plt.show()
"""