# Author: Almero Gouws <14366037@sun.ac.za>
"""
This module contains functions that are used in testing the PyBNT toolbox.

Functions contained in this module are:
    'are_equal': Tests whether two Numpy ndarray's are equivalent.
    'read_example': Reads test data from a text file.
"""
import numpy as np
from os import path

def read_samples(fname, nsamples=0):
    """
    This function reads a set of values from a text file into a numpy array.

    Parameters
    ----------
    fname: String
        The path to the text file containing the values
    """
    f = open('./Data/' + fname, 'r')
    lines = f.readlines()
    ans = []
    for line in lines:
        line = line.split()
        temp = []
        for val in line:
            temp.append(float(val) - 1)
        ans.append(temp)
    ans = np.array(ans)
    f.close()

    return  ans

def create_all_evidence(nodes, sizes):
    """
    """
    num_samples = (sizes+1)**nodes
    counts = []
    cur_counts = []
    cur_value = []
    for i in range(0, nodes):
        counts.append((sizes+1)**i)
        cur_counts.append(0)
        cur_value.append([])
    counts.reverse()

    samples = []
    for i in range(0, num_samples):
        sample = []
        for i in range(0, nodes):
            sample.append(cur_value[i])
            cur_counts[i] = cur_counts[i] + 1
            if cur_counts[i] == counts[i]:
                if type(cur_value[i]) == list:
                    cur_value[i] = 0
                elif cur_value[i] != sizes - 1:
                    cur_value[i] = cur_value[i] + 1
                else:
                    cur_value[i] = []
                cur_counts[i] = 0
            
        samples.append(sample)

    return samples
        
