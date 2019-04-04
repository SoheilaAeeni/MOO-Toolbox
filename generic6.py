#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import array
import random
import numpy as np
import math
import sys
from deap import base, creator, tools

class ga_generic():
     def setup(self,f, t,obj,Ngen, **kwargs):
         self.fun = f
         self.t = t
         self.Ngen=Ngen
         self.obj=obj
         self.fd=25
         self.fb=16
         self.ib=16
         self.id=0
         if ('lowerB' in kwargs):
             self.lb = kwargs['lowerB']
         if ('upperB' in kwargs):
             self.ub = kwargs['upperB']
         if ('A' in kwargs):
             self.A = kwargs['A']
         if ('b' in kwargs):
             self.b = kwargs['b']
         if ('Aeq' in kwargs):
             self.Aeq = kwargs['Aeq']
         if ('beq' in kwargs):
             self.beq = kwargs['beq']
         if ('nonlcons' in kwargs):
             self.nonlcons = kwargs['nonlcons']
         if ('eqnonlcons' in kwargs):
             self.eqnonlcons = kwargs['eqnonlcons']
     def myObjective(self):
         obj=self.obj
         NrOfObj=len(obj)
         for i in obj:
             if i !='max' and i != 'min':
                 return 0
         if NrOfObj==1:
             if obj[0]=='max':
                 return (1.0,)
             elif obj[0]=='min':
                 return (-1.0,)
         elif NrOfObj>1:
            return tuple(1.0 if i == 'max' else -1.0 for i in obj)
     def intN(self,lb1,ub1):
         diff=ub1-lb1+1
         n=math.log(diff,2)
         n=np.ceil(n)
         return n
     def deBinerize(self,x):
         ind=[]
         t=self.t
         typeOfInput=len(t)
         Lb=self.lb
         Ub=self.ub
         m1=0
         for j in range(typeOfInput):
             if t[j]=="float":
                 b=x[m1:m1+self.fb+self.fd]
                 m1+=self.fb+self.fd
                 lb=Lb[j]
                 ub=Ub[j]
                 n= self.fb+self.fd
                 m= 0
                 y=b
                 powers = list(range(n-1,-m-1,-1))
                 b2d = [y[i]*2**(powers[i]) for i in range(len(y))]
                 b2d= (sum(b2d)/((2**(m+n)-1)))*(ub-lb)+lb
                 ind.append(b2d)
             elif t[j]=="int":
                 lb=Lb[j]
                 ub=Ub[j]
                 n=int(self.intN(lb,ub))
                 b=x[m1:m1+n]
                 m1+=n
                 m= 0
                 y=b
                 powers = list(range(n-1,-m-1,-1))
                 b2d = [y[i]*2**(powers[i]) for i in range(len(y))]
                 b2d=int((sum(b2d)/(2**(m+n)-1))*(ub-lb)+lb)
                 ind.append(b2d)
             elif t[j]=="bool":
                 lb=0
                 ub=1
                 n=int(self.intN(lb,ub))
                 b=x[m1:m1+n]
                 m1+=n
                 m= 0
                 y=b
                 powers = list(range(n-1,-m-1,-1))
                 b2d = [y[i]*2**(powers[i]) for i in range(len(y))]
                 b2d=int((sum(b2d)/(2**(m+n)-1))*(ub-lb)+lb)
                 ind.append(b2d)
         return ind
     def myfun(self,x):
         if len(list(set(self.t)))>1:
             x=self.deBinerize(x)
         penalty1 = 0
         penalty2 = 0
         penalty3 = 0
         penalty4 = 0
         obj=self.obj
         NrOfObj=len(obj)
         objSigns = self.myObjective()
         try:
             if self.A!=[]:
                 linCon = np.matmul(self.A,x)-self.b
                 #linCon = np.matmul(self.A,np.matrix(x).T)-self.b
                 linVal = [max(linCon[i],0) for i in range(len(linCon))]
                 penalty1 += 1.e9*(max(linVal))**2
         except:
             penalty1 = 0
         try:
             if self.Aeq!=[]:
                 linEqCon = (np.matmul(self.Aeq,x)-self.beq)**2
                 linEqVal = [linEqCon[i] for i in range(len(linEqCon))]
                 penalty2+= 1.e9*max(linEqVal)
         except:
             penalty2 = 0
         try:
             if self.nonlcons:
                 nonlinCon=(self.nonlcons(x))
                 if isinstance(nonlinCon,tuple):
                     nonlinVal = [max(nonlinCon[i],0) for i in range(len(nonlinCon))]
                 else:
                     nonlinVal = [max(nonlinCon,0)]
                 penalty3 += 1.e5*(max(nonlinVal))**2
                 #nlval = max(self.nonlcons(x),0)
                 #penalty3+= 1000000*(nlval)**2
         except:
             penalty3 = 0
         try:
             if self.eqnonlcons:
                 nonlinEqCon=(self.eqnonlcons(x))
                 if isinstance(nonlinEqCon,tuple):
                     nonlinEqVal = [nonlinEqCon[i]**2 for i in range(len(nonlinEqCon))]
                 else:
                     nonlinEqVal = [nonlinEqCon**2]
                     
                 
                 #nonlinEqVal = [nonlinEqCon[i]**2 for i in range(len(nonlinEqCon))]
                 penalty4+= 1.e9*max(nonlinEqVal)
                # penalty4+= max(1000000*((self.eqnonlcons(x))**2))
         except:
             penalty4 = 0
         penalties= penalty1+penalty2+penalty3+penalty4
         if NrOfObj==1:
             return self.fun(x)+ penalties*objSigns[0]*-1,
         else:
             # for loop in multiobj problems
             v= self.fun(x)
             ret = tuple(v[i]+penalties*objSigns[i]*-1 for i in range(len(v)))
             return ret 
         #vol+penalties*objSigns[0]*-1,stress+penalties*objSigns[0]*-1,z+penalties*objSigns[0]*-1
     
     def binaryBuilder(self,icls):
         genome = list()
         t=self.t
         typeOfInput=len(t)
         for i in range(typeOfInput):
             if t[i]=='float':
                 n = self.fb;              
                 m = self.fd;
                 totalBits = m+n
                 param_1 = [random.randint(0,1) for i in range(totalBits)]
                 genome=genome+param_1
             elif t[i]=='int':
                 n = self.ib;              
                 m = self.id;
                 totalBits = m+n
                 param_2 = [random.randint(0,1) for i in range(totalBits)]
                 genome=genome+param_2
             elif t[i]== 'bool':
                 param_3 = [random.randint(0,1)]
                 genome=genome+param_3
         return icls(genome)
     def genFunkyInd(self,icls, L,U):
         genome = list()
         t=self.t
         typeOfInput=len(t)
         for i in range(typeOfInput):
             if t[i]=='float':
                 param_1 = random.uniform(L[i],U[i])
                 genome.append(param_1)
             elif t[i]=='int':
                 param_2 = random.randint(L[i],U[i])
                 genome.append(param_2)
             elif t[i]== 'bool':
                 param_3 = random.randint(0,1)
                 genome.append(param_3)
         return icls(genome)
     def my_crossover(self,LL,UU,ind1, ind2):
         t=self.t
         typeOfInput=len(t)
         ind11 = []
         ind22 = []
         for i in range(typeOfInput):
             if t[i] =='float':
                 ind1var1,ind2var1=tools.cxSimulatedBinaryBounded([ind1[i]], [ind2[i]], low=LL[i], up=UU[i], eta=20.0)
                 ind11.append(ind1var1[0])
                 ind22.append(ind2var1[0])
             elif t[i]=='int':
                 ind1var2,ind2var2=tools.cxUniform([ind1[i]], [ind2[i]],indpb=0.9)
                 ind11.append(ind1var2[0])
                 ind22.append(ind2var2[0])
             elif t[i]=='bool':
                 toss = random.random()
                 if toss > 0.5:
                     ind1var3,ind2var3=ind2[i], ind1[i]
                     ind11.append(ind1var3)
                     ind22.append(ind2var3)
         return ind11,ind22
     def my_mutation(self,LL,UU,individual):
         t=self.t
         typeOfInput=len(t)
         individual_ = []
         for i in range(typeOfInput):
             if t[i] =='float':
                 ind1=tools.mutPolynomialBounded([individual[i]], low=LL[i], up=UU[i], eta=20.0,indpb=1.0/len(self.t))
                 individual_.append(ind1[0][0])
             elif t[i] =='int' or t[i]== 'bool':
                 ind2=tools.mutUniformInt([individual[i]], low=LL[i], up=UU[i], indpb= 0.9)
                 individual_.append(ind2[0][0])    
         return individual_
             

     def compute(self):
         try:
             if self.ub!=[] and self.lb!=[]:
                 BOUND_LOW, BOUND_UP = self.lb, self.ub
                 self.flagUL=0
             else:
                 BOUND_LOW, BOUND_UP =  -10000,  10000
                 self.flagUL=1
         except:
             BOUND_LOW, BOUND_UP = -10000,  10000
             self.flagUL=1
         obj = self.myObjective()
         t=self.t
         typeOfInput=len(t)
         if obj ==0:
             print('The objective function can take \'min\' or \'max\' values. Please correct them and run agin!')
         creator.create("FitnessMin", base.Fitness, weights=obj)
         creator.create("Individual", list, fitness=creator.FitnessMin)
         self.toolbox = base.Toolbox()
         ####################################
         ############# individuals ##########
         ####################################
         if list(set(t))[0] =='bool' and len(list(set(t)))==1:
             self.toolbox.register("attr_bool", random.randint, 0, 1)
             self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool,len(t))
         elif list(set(t))[0] =='float' and len(list(set(t)))==1:
             def uniform(low, up, size=None):
                 try:
                     return [random.uniform(a, b) for a, b in zip(low, up)]
                 except TypeError:
                     return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
             self.toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, len(t))
             self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_float)
         elif list(set(t))[0] =='int' and len(list(set(t)))==1:
             def uniform_int(low, up, size=None):
                 try:
                     return [random.randint(a, b) for a, b in zip(low, up)]
                 except TypeError:
                     return [random.randint(a, b) for a, b in zip([low] * size, [up] * size)]
             self.toolbox.register("attr_int", uniform_int, BOUND_LOW,BOUND_UP)
             self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_int)
         else:
             self.toolbox.register('individual', self.binaryBuilder, creator.Individual)
             #self.toolbox.register('individual', self.genFunkyInd, creator.Individual, BOUND_LOW,BOUND_UP)
         self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
         self.toolbox.register("evaluate", self.myfun)
         #####################################
         ############# crossover #############
         #####################################
         if list(set(t))[0] =='bool' and len(list(set(t)))==1:
             self.toolbox.register("mate", tools.cxTwoPoint)
         elif list(set(t))[0] =='float' and len(list(set(t)))==1:
             self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
         elif list(set(t))[0] =='int' and len(list(set(t)))==1:
             self.toolbox.register("mate", tools.cxUniform, indpb=0.1)
         else:
             self.toolbox.register("mate", tools.cxTwoPoint)
             #self.toolbox.register("mate",self.my_crossover,BOUND_LOW,BOUND_UP)
        #####################################
        ############# mutation ##############
        #####################################
         if list(set(t))[0] =='bool' and len(list(set(t)))==1:
             self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)
         elif list(set(t))[0] =='float' and len(list(set(t)))==1:
             if self.flagUL==0:
                 self.toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/len(self.t))
             elif self.flagUL==1:
                 self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.3)
         elif list(set(t))[0] =='int' and len(list(set(t)))==1:
             self.toolbox.register("mutate", tools.mutUniformInt, low=BOUND_LOW, up=BOUND_UP, indpb= 0.3)
         else:
             self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)
             #self.toolbox.register("mutate",self.my_mutation,BOUND_LOW,BOUND_UP)
         ####################################
         #################selection #########
         ####################################
         self.toolbox.register("select", tools.selNSGA2)
     def main(self,seed=64):
         random.seed(seed)
         self.compute()
         if self.flagUL==0:
             NGEN = self.Ngen
             MU = 100
         elif self.flagUL==1:
             NGEN = 2000
             MU = 200
         
         CXPB = 0.9
         stats = tools.Statistics(lambda ind: ind.fitness.values)
         # stats.register("avg", numpy.mean, axis=0)
         # stats.register("std", numpy.std, axis=0)
         stats.register("min", np.min, axis=0)
         #stats.register("max", np.max, axis=0)
         logbook = tools.Logbook()
         logbook.header = "gen", "evals", "std", "min", "avg", "max"
         
         pop = self.toolbox.population(n=MU)
         # Evaluate the individuals with an invalid fitness
         invalid_ind = [ind for ind in pop if not ind.fitness.valid]
         fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
         for ind, fit in zip(invalid_ind, fitnesses):
             ind.fitness.values = fit

         # This is just to assign the crowding distance to the individuals
         # no actual selection is done
         pop = self.toolbox.select(pop, len(pop))
         record = stats.compile(pop)
         logbook.record(gen=0, evals=len(invalid_ind), **record)
         print(logbook.stream)
     
         # Begin the generational process
         for gen in range(1, NGEN):
             # Vary the population
             offspring = tools.selTournamentDCD(pop, len(pop))
             offspring = [self.toolbox.clone(ind) for ind in offspring]
             
             for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                 if random.random() <= CXPB:
                     self.toolbox.mate(ind1, ind2)
                 
                 self.toolbox.mutate(ind1)
                 self.toolbox.mutate(ind2)
                 del ind1.fitness.values, ind2.fitness.values
            
             # Evaluate the individuals with an invalid fitness
             invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
             fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
             for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit
     
             # Select the next generation population
             pop = self.toolbox.select(pop + offspring, MU)
             record = stats.compile(pop)
             logbook.record(gen=gen, evals=len(invalid_ind), **record)
             print(logbook.stream)
             #print(pop)
             #print(ind)
         return pop, logbook,ind
         


            