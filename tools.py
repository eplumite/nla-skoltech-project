import datetime
import os

import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt

def preprocess_dataset(X):
    return preprocessing.scale(X)

def create_sparsity_pattern(X, title=None, precision=0.1, markersize=5, filename=None): 
    plt.figure(figsize=(10, 10))
    plt.spy(x, precision=precision, markersize=markersize)
    plt.title(title)
    if not os.path.exists('images'):
        os.mkdir('images')
    if not filename:
        filename = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(f'images/sparsity_pattern_{filename}.png')

def create_plot(data, title=None, xlabel=None, ylabel=None, legend=[], yscale=None, filename=None): 
    plt.figure(figsize=(12, 7))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    if yscale == 'log':
        plt.yscale('log')
    plt.grid()
    if not os.path.exists('images'):
        os.mkdir('images')
    if not filename:
        filename = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(f'images/plot_{filename}.png')


