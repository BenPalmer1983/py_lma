#!/usr/bin/python3
################################################################################
#
#  LMA Optimisation
#
################################################################################

# Calculates the variables of a function so that it minimises the sum
# of squares between the function and the data points
#
# LMA algorithm
# Levenberg, K.: 1944, Quarterly of Applied Mathematics 2, 164:
#    Dampened Newton Gauss Algorithm
# Marquardt, D.: 1963, Journal of the Society for Industrial and Applied Mathematics 11(2), 431:
#    Introduction of Lambda (Marquardt Parameter)
# R. Fletcher 1971, A modified Marquardt Subroutine for Non-linear Least Squares
#    Starting Lambda and lower lambda cutoff (where it is replaced by 0)
# M. Transtrum, J. Sethna 2012, Improvements to the Levenberg-Marquardt algorithm for nonlinear
# least-squares minimization:
#    Delayed gratification scheme for varying lambda
# Jens Jessen-Hansen 2011, Levenberg-Marquardts algorithm Project in Numerical Methods:
#    Independent/Diagonal covariance matrix for weighting
# Parameter limits implemented
# Finite difference method used to calculate J

#
# Includes simulated annealing preoptimiser
# References:
# https://en.wikipedia.org/wiki/Simulated_annealing
# http://katrinaeg.com/simulated-annealing.html
# http://mathworld.wolfram.com/SimulatedAnnealing.html
#

# 
# Example useage
#
# See End
#

import sys
import os
import random
import numpy as np

class lma:

  def __init__(self, file_input = None):
    self.reset()
    if(file_input != None):
      if(isinstance(file_input, (list,))):
        self.set_data(file_input)
      elif(isinstance(file_input,np.ndarray)):
        self.set_data(file_input)
      elif(os.path.isfile(file_input)):
        self.load_file(file_input)


  ###################################
  #  Defaults
  ###################################

  def reset(self):
    self.verbose = False
    self.conv_thr = 1.0E-9
    self.max_cycles = 10
    self.h = 0.0001
    self.run_sa = False
    self.sa_count = 100
    self.sa_temp_start = 10.0
    self.sa_temp_end = 0.1
    self.sa_temp_reduction = 0.9

    self.parameters = np.zeros((10))
    self.parameters_upper = None
    self.parameters_lower = None
    self.function = None
    self.p_count = None
    self.lam_cutoff = 0.1
    self.lam = 0.1



  ###################################
  #  Load Data
  ###################################

  def load_file(self, file_name=None):
    if(file_name == None):
      return False 

    # Init variable
    file_data = ""

    # Read it in line by line
    fh = open(file_name, "r")
    for file_row in fh:
      file_data = file_data + file_row.strip() + '\n'

    # Clean
    file_data = lma.clean(file_data)

    self.load_data(file_data)

    
  def load_data(self, file_data):  # 
    lines = file_data.split('\n')
  
    data_list = []
    for line in lines:
      fields = line.split(',')
      if(len(fields) == 2):
        try: 
          data_list.append([float(fields[0]), float(fields[1])])
        except:
          pass
    self.set_data(data_list)   


  def set_data(self, data):
    self.data = np.zeros((len(data), 2))
    for row in range(len(data)):
      self.data[row,0] = data[row][0]
      self.data[row,1] = data[row][1]
    self.data_len = len(self.data)



  ###################################
  #  Setters
  ###################################

  def set_threshold(self, convThreshold):
    self.conv_thr = conv_thr


  def set_fit(self, func, p):
    # Set function and parameter count
    self.function = func
    self.p_count = len(p)
                  
    # Set parameters  
    self.parameters = np.zeros((len(p)))    
    for i in range(len(p)):
      self.parameters[i] = p[i]            
     
      
  def set_sa(self, settings, pl=None, pu=None):
    #{"temp_start": 10.0, "temp_end": 1.0, "factor": 0.5, "count": 10}
    self.sa_count = settings['count']
    self.sa_temp_start = settings['temp_start']
    self.sa_temp_end = settings['temp_end']
    self.sa_temp_reduction = settings['factor']
    
    # Set parameter bounds
    self.parameters_lower = np.zeros((len(pl)))
    self.parameters_upper = np.zeros((len(pu)))
    for i in range(len(pl)):
      if(pl != None):
        self.parameters_lower[i] = pl[i]
    for i in range(len(pl)):
      if(pu != None): 
        self.parameters_upper[i] = pu[i]
    if(pl != None and pu != None and self.p_count == len(pl) and self.p_count == len(pu)):
      self.run_sa = True
      

  ###################################
  #  Calc
  ###################################


  def calc(self):
    self.rss_start = self.calc_rss()
    p_input = np.copy(self.parameters)
    while(True):
      try: 
        self.sa()
        self.outer_cycle()
        break
      except:
        self.parameters = np.copy(p_input)
        pass
    return self.parameters, self.calc_rss()


  def sa(self):
    if(self.run_sa == False):
      return 0
 
    p_opt = np.copy(self.parameters)
    rss_opt = self.calc_rss()
    
    temperature = self.sa_temp_start   
    while(temperature > self.sa_temp_end):
      n = 0 
      while(n<self.sa_count):
        p_best = np.copy(self.parameters)
        rss_best = self.calc_rss()
        
        # Vary
        self.parameters = self.parameters_lower[:] + (self.parameters_upper[:] - self.parameters_lower[:]) * np.random.rand(len(self.parameters_lower))
        rss = self.calc_rss()
        
        if(rss < rss_best):
          rss_best = rss
          if(rss_best < rss_opt):
            rss_opt = rss_best
            p_opt = np.copy(self.parameters)
        else:  
          a = self.sa_acceptance(temperature, rss_best, rss)
          if(a > random.random()):
            rss_best = rss
          else:
            self.parameters = np.copy(p_best)
        # Increment
        n = n + 1    
      # Reload optimum and cool  
      self.parameters = np.copy(p_opt)  
      temperature = temperature * self.sa_temp_reduction
      
    if(self.verbose): 
      print(self.parameters, rss_opt)

  def sa_acceptance(self, temperature, best, new):
    return np.exp((best - new) / temperature)

  def outer_cycle(self):
    # (JTJ+Lambda*diag(JTJ)) P = (-1*JTR)
    # (H+Lambda*diag(H)) P = (-1*JTR)
    i = 0
    self.converged = False
    while(i < 100 and self.converged == False):
      i = i + 1
      self.make_residual()        # R
      self.make_jacobian()        # J
      self.make_jacobian_t()      # JT
      self.make_hessian()         # H ~ JTJ
      self.inner_cycle()
      if(self.calc_rss() < self.conv_thr):  
        self.converged = True


  def inner_cycle(self):
    last_rss = self.calc_rss() 
    for i in range(0,20):
      # Store last values
      p_last = np.copy(self.parameters)
      # Calculate matrices
      self.make_hessian()         # H ~ JTJ   
      self.make_dampening()
      self.make_nJTR()            # -JTR
      self.dampen_hessian()
      self.update_parameters()
      rss = self.calc_rss()
      # Set parameters/rss
      if(rss>last_rss):
        self.parameters = np.copy(p_last)
        self.lam = self.lam * 1.5e0
      elif(rss == last_rss):
        self.lam = self.lam * 0.2e0
      else:
        last_rss = rss
        self.lam = self.lam * 0.2e0


  def make_residual(self):
    # Calculate residual
    self.r = self.function(self.parameters, self.data[:, 0]) - self.data[:, 1]



  def make_jacobian(self):
    self.J = np.zeros((self.data_len, self.p_count))
    
    for i in range(0, self.data_len):
      for j in range(0, self.p_count):
        # Reset parameters
        for k in range(self.p_count):
          p = np.copy(self.parameters)

        # Vary jth parameter
        p[j] = p[j] + self.h
        
        r = (self.function(p, self.data[i, 0]) - self.data[i, 1])
        self.J[i,j] = (r - self.r[i]) / self.h




  def make_jacobian_t(self):
    self.JT = np.transpose(self.J)

  def make_hessian(self):
    self.H = np.matmul(self.JT, self.J)


  def make_dampening(self):
    self.damp = np.identity(self.p_count)
    for i in range(0,self.p_count):
      self.damp[i,i] = self.lam * self.H[i,i]


  def make_nJTR(self):
    self.nJTR = -1 * np.matmul(self.JT, self.r)


  def dampen_hessian(self):
    for i in range(0,self.p_count):
      self.H[i,i] = self.H[i,i] + self.damp[i,i]

#l_cutoff




  def update_parameters(self):
    # A x = y
    x = np.linalg.solve(self.H, self.nJTR)
    p = np.copy(self.parameters)
    self.parameters[0:self.p_count] = p[0:self.p_count] + x[:]

  def calc_rss(self):
    return sum((self.function(self.parameters, self.data[:, 0]) - self.data[:, 1])**2)
    



#####################
# Static Functions
#####################

  @staticmethod
  def clean(str_in):  
    str_out = ""
    l = len(str_in)
    for i in range(l):
      # Last, Next, This
      if(i == 0):
        last = None
      else:
        last = str_in[i-1]
      if(i < (l-1)):
        next = str_in[i+1]
      else:  
        next = None
      char = str_in[i]
    
      # Check
      ok = True
      if(last == " " and char == " "):
        ok = False
      elif(last == "\n" and char == "\n"):
        ok = False
      elif(last == "\n" and char == " "):
        ok = False
      elif(char == " " and next == "\n"):
        ok = False

      # Add to string  
      if(ok):
        str_out += char
    return str_out    





################################################################################
 
 
 
# Example 1

"""
from lma import lma
import numpy as np

# Define function the data is being fit to
def double_exp(p, x):
  return p[0] * np.exp(x * p[1]) + p[2] * np.exp(x * p[3])
  

fit = lma("2exp.csv")
p = [0.156,3.7,0.87,-5.1]
fit.set_fit(double_exp, p)
fit.set_sa({"temp_start": 10.0, "temp_end": 1.0, "factor": 0.5, "count": 10}, [0,-5,0,-5],[1,5,1,5])
p, rss = fit.calc()
print(p)
print(rss)
print()

"""



# Example 2

"""
from lma import lma
import numpy as np

# Define function the data is being fit to
def isolated(p, x):
  return p[0] + p[1] * np.exp(p[2] * (x + p[3]))
  

data = [[10,-157.63712186],[11,-157.47857768],[12,-157.35301373],[13,-157.25535275],[14,-157.17942578]]
fit = lma(data)
p = [0,0,0,0]
fit.set_fit(isolated, p)
fit.set_sa({"temp_start": 10.0, "temp_end": 0.1, "factor": 0.9, "count": 1000}, [-200,-5,-2,-12], [200,5,2,10])
p, rss = fit.calc()
print(p)
print(rss)
print()

"""