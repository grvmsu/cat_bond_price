# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate as si
from scipy.stats import pareto
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
np.random.seed(123)



def transform_threshold(threshold, T):
    
    return T * threshold


class CAT():
    
    def __init__(self, df, r0, I, mu, threshold, b, T, rp):

        np.random.seed(123)

        self.df = df
        self.r0 = r0
        self.I = I        
        self.b = b
        self.T = T
        self.rp = rp
        self.mu = 12
        self.M = int(self.T * self.mu)
        self.t = 0.08333333333333
        self.I = I
        self.threshold = threshold
        self.main_losses = None
        
        self.kappa = None
        self.sigma = None
        self.theta = None
        
        self.rates_cir = None
        self.price = None
        self.frontier = None
        self.mean = None
        
        self.a_b_params()
        self.NSS()
        self.ols_params()
        self.calibrate_theta(1)
        self.minimize_error()
    
    def run_main_funcs(self):
        
        self.cir_rates()
        self.bond_price()
        self.sim_losses()
        self.cat_prices()
        
    def a_b_params(self):
        
        self.a1 = 0
        self.a2 = 0.6
        k = 1.6
        list_of_self_a = [0, 0, 0.6] * 4
        for i in range (2, 10):
          list_of_self_a[i + 1] = list_of_self_a[i] + list_of_self_a[2] * k**(i - 1)
        list_of_self_b = [0, list_of_self_a[2]] * 6
        for i in range (1, 10):
          list_of_self_b[i + 1] = list_of_self_b[i] * k
        self.a3 = list_of_self_a[3]
        self.a4 = list_of_self_a[4]
        self.a5 = list_of_self_a[5]
        self.a6 = list_of_self_a[6]
        self.a7 = list_of_self_a[7]
        self.a8 = list_of_self_a[8]
        self.a9 = list_of_self_a[9]
        self.b1 = list_of_self_b[1]
        self.b2 = list_of_self_b[2]
        self.b3 = list_of_self_b[3]
        self.b4 = list_of_self_b[4]
        self.b5 = list_of_self_b[5]
        self.b6 = list_of_self_b[6]
        self.b7 = list_of_self_b[7]
        self.b8 = list_of_self_b[8]
        self.b9 = list_of_self_b[9]
    
    def NSS(self):
 
        self.df['rate_1m'] = self.df.BETA0 / 10000 + \
               (self.df.BETA1 / 10000 + self.df.BETA2 / 10000) * (self.df.TAU / self.t) * \
                   (1 - np.exp( - self.t / self.df.TAU)) - \
               self.df.BETA2 / 10000 * np.exp( - self.t / self.df.TAU) + \
               self.df.G1 / 10000 * np.exp( - ((self.t - self.a1) * (self.t - self.a1)) / (self.b1 * self.b1)) + \
               self.df.G2 / 10000 * np.exp( - ((self.t - self.a2) * (self.t - self.a2)) / (self.b2 * self.b2)) + \
               self.df.G3 / 10000 * np.exp( - ((self.t - self.a3) * (self.t - self.a3)) / (self.b3 * self.b3)) + \
               self.df.G4 / 10000 * np.exp( - ((self.t - self.a4) * (self.t - self.a4)) / (self.b4 * self.b4)) + \
               self.df.G5 / 10000 * np.exp( - ((self.t - self.a5) * (self.t - self.a5)) / (self.b5 * self.b5)) + \
               self.df.G6 / 10000 * np.exp( - ((self.t - self.a6) * (self.t - self.a6)) / (self.b6 * self.b6)) + \
               self.df.G7 / 10000 * np.exp( - ((self.t - self.a7) * (self.t - self.a7)) / (self.b7 * self.b7)) + \
               self.df.G8 / 10000 * np.exp( - ((self.t - self.a8) * (self.t - self.a8)) / (self.b8 * self.b8)) + \
               self.df.G9 / 10000 * np.exp( - ((self.t - self.a9) * (self.t - self.a9)) / (self.b9 * self.b9))
    
    def ols_params(self):
        
        self.df['diff'] = np.concatenate([self.df['rate_1m'][1:].to_numpy() - self.df['rate_1m'][:-1].to_numpy(), np.empty(1)])
        
        X = np.vstack((np.ones(len(self.df['rate_1m'])), self.df['rate_1m'])).T
        beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.df['diff'])
        yhat = X.dot(beta_hat)
        ybar = np.sum(self.df['diff'])/len(self.df['diff'])
        
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((self.df['diff'] - ybar) ** 2)
        rsq = ssreg / sstot
        
        rsq_adj = 1 - (1 - rsq) * (len(self.df['diff']) - 1) / (len(self.df['diff']) - 2)
        
        self.kappa = - beta_hat[1] * 250
        self.theta = - beta_hat[0] / beta_hat[1]    
        
        self.se = np.sqrt(1 - rsq_adj) * np.std(self.df['diff'])
        self.sigma = self.se * np.sqrt(250)
                
        self.kappa = 1.519142290996501
        self.theta = 0.07259130303977346
        self.sigma = 0.03411707187632048

    def calibrate_theta(self, k_fit):
        
        # https://www-2.rotman.utoronto.ca/~hull/ofod/index.html
        # https://www.cbr.ru/hd_base/zcyc_params/zcyc/
        t = np.array([0.25, 0.50, 0.75, 1.00, 2.00, 3.00, 5.00, 7.00, 10.00, 15.00, 20.00, 30.00])
        self.today_kbd = np.array([0.063, 0.0685, 0.0687, 0.0688, 0.0692, 0.0693, 0.0695, 0.0699, 0.0712, 0.0734, 0.0751, 0.0772])
        
        self.kappa_new = self.kappa + k_fit * self.sigma
        self.theta_new = self.theta * self.kappa / self.kappa_new
        self.gamma = np.sqrt(self.kappa_new * self.kappa_new + 2 * self.sigma * self.sigma)
        
        # переход к риск-нейтральным параметрам
        
        self.B_cir = (2 * (np.exp(self.gamma * t) - 1))/((self.gamma + self.kappa_new)*(np.exp(self.gamma * t) - 1) + 2 * self.gamma)
        self.A_cir = (2 * self.gamma * np.exp((self.kappa_new + self.gamma) * t / 2)/
                      ((self.kappa_new + self.gamma) * (np.exp(self.gamma * t) - 1) + 2 * self.gamma)) ** (2 * self.kappa_new * self.theta_new / (self.sigma * self.sigma))
        
        self.R_cir = (-np.log(self.A_cir) + self.B_cir * self.r0) / t
        self.k_fit = k_fit
        error = np.sum((self.R_cir - self.today_kbd)**2)
        self.error = error
        
        return error
        
    
    def minimize_error(self):
        
        try:
            self.res = minimize(self.calibrate_theta, 1, method = 'SLSQP')
            self.kappa = self.kappa_new[0]
            self.theta = self.theta_new[0]
        except:
            print('оптимизация прошла неудачно, theta та же')
        
        
    def cir_rates(self):
        
        self.rates_cir = np.zeros((self.M + 1, self.I))
        
        self.rates_cir[0] = self.r0
        for t in range(1, self.M + 1):
            # r(t+1) = r(t) + kappa*(theta - r(t))*(df) + sigma*sqrt(r(t))*sqrt(dt)*dz
            self.rates_cir[t] = self.rates_cir[t - 1] + \
                             self.kappa * \
                             (self.theta - self.rates_cir[t - 1]) * \
                             self.T / self.M + \
                             self.sigma * \
                             np.sqrt(np.maximum(self.rates_cir[t - 1], 0)) * \
                             np.sqrt(self.T / self.M) * \
                             np.random.standard_normal(self.I)
    
    def bond_price(self):
        
        self.price = np.zeros(self.I)
        for i in range(self.I):
            self.price[i] = np.exp( - \
                            (si.simps((self.rates_cir[1: ,:])[:, i].T, \
                            np.linspace(0, self.T, num = self.M + 1)[1: ])))
        self.frontier = np.sqrt(np.var(self.price)) * 1.96 / np.sqrt(self.I)
        self.mean = np.mean(self.price)
    
    def distribution_by_period(self, total_time_intervals, events):
        
        losses = []
        
        for period in range(1, self.M + 1):
            losses.append(np.sum(np.where(((period - 1)/self.M < total_time_intervals) & \
                                (total_time_intervals < period / self.M), events, 0)))  
        return np.array(losses)
        
    def catastrophe_pareto_generator(self, rn):
        
        if self.b:
            return pareto.rvs(self.b, 0, 0.127, rn)
        else:
            return pareto.rvs(1.904, 0, 0.127, rn)
        
    
    def compound_poisson_process(self):
        
        low_num = 4
        high_num = 17
         
        low_num_T = int(np.floor(low_num * self.T))
        high_num_T = int(np.floor(high_num * self.T))      
        
        rn = np.random.randint(low_num_T, high_num_T)
        time_intervals = -np.log(np.random.random(rn)) / self.M
        total_time_intervals = time_intervals.cumsum()
        events = self.catastrophe_pareto_generator(rn)
        simulated_losses = self.distribution_by_period(total_time_intervals, events).cumsum()
        
        return simulated_losses
    
    def sim_losses(self):
        
        self.main_losses = []
        for i in range(self.I):
            self.main_losses.append(self.compound_poisson_process())
            
        self.loss_less_threshold = []
        for i in self.main_losses:
            if i[-1] <= self.threshold:
                self.loss_less_threshold.append(1)
            else:
                self.loss_less_threshold.append(self.rp)
                
    def cat_prices(self):
        
        self.cat_price = self.price * np.array(self.loss_less_threshold)
        self.frontier_cat = np.sqrt(np.var(self.cat_price)) * 1.96 / np.sqrt(self.I)
        self.mean_cat = np.mean(self.cat_price)



