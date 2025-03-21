from itertools import combinations
import pandas as pd
import numpy as np
import re
import os
from MDAnalysis.analysis.distances import distance_array


def sort_dictionary_values(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: -item[1]))

def parse_id(contact):
    '''
    take the contact name (column id) and return a dictionary of
    the residue A descriptors and residue B descriptors
    '''
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    ### for combined contact data, the prepended name needs to be removed from 
    ### chain a
    ##### This might break something if multiple contacts
    #### are going into the keys of another dictionary because
    #### duplicate names will be overwritten.
    ## shouldn't be a problem for averaging functions because combined data
    ## will be produced from pre-averaged data
    ## to_heatmap() will not give correct results as is - need to prepare
    ## the data with original names for that....
    
    if "_" in chaina:
        chaina = chaina.split("_")[1]

    return {'chaina':chaina, 'resna':resna, 'resida':resida,
             'chainb':chainb, 'resnb':resnb, 'residb':residb}

def split_id(contact):
    '''
    take the contact name and split it into its two residue parts
    returns a dictionary where 'resa' will contain 'CH:RES:NUM'
    '''
    resa, resb = re.split("-", contact)
    return {'resa':resa, 'resb':resb}

def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
        
def normit(data, center=True):
    '''
    normalize a list of values.
    '''
    if center == True:
        return (data - data.mean())/np.abs((data - data.mean())).max()
    else:
        return data/np.abs(data).max()
    
def multi_intersection(lists, cutoff=None, verbose=False):
    '''
    Return the intersection of the values in lists.  
    Parameters
    ----------
    lists : list of lists
        The lists of values to identify the shared elements from.

    cutoff : float or int
        If not None, return the intersection of a subset of lists that meet the criteria.
        float < 1 will only include lists that have a length of cutoff percent of
        the longest list.
        int > 1 will only include lists that are longer than cutoff.

    verbose : bool
        If True, print the number of lists that were provided as input and the number
        of lists that were used in constructing the intersection.

    Returns
    -------
    list
    intersection of values in lists.
        
    '''

    initial = len(lists)
    if cutoff is not None and cutoff < 1:
        longest_len = max([len(data) for data in lists])
        lists = [data for data in lists if len(data) > longest_len*cutoff]
    elif cutoff is not None and cutoff > 1:
        lists = [data for data in lists if len(data) > cutoff]

    final = len(lists)
    set1 = set(lists[0])
    setlist = [set(data) for data in lists[1:]]
    if verbose == True:
        print(f'n lists initial: {initial} \nn lists final: {final}')
    return sorted(list(set1.intersection(*setlist)))

def get_subplot_rows(n_items, n_columns):
    return (n_items + n_columns - 1) // n_columns

def sort_nested_dict(d):
    '''
    Sort the split sum dictionary. This is expecting the keys of the nested dictionary to be
    in the form of "A:ALA:5". 
    '''
    sorted_dict = {}
    for outer_key, nested_dict in d.items():
        sorted_keys = sorted(nested_dict.keys(), key=lambda x: (x.split(":")[0], int(x.split(":")[-1])))
        sorted_nested_dict = {key: nested_dict[key] for key in sorted_keys}
        sorted_dict[outer_key] = sorted_nested_dict
    return sorted_dict


def distribute_files(num_processes, files):
    """
    Distributes files from the given directory among the specified number of processes.
    """
    
    total_files = len(files)
    
    # Calculate the number of files each process should handle
    files_per_process = math.ceil(total_files / num_processes)
    
    # Create tasks as lists of files
    tasks = []
    for process_id in range(num_processes):
        start_idx = process_id * files_per_process
        end_idx = min(start_idx + files_per_process, total_files)
        tasks.append(files[start_idx:end_idx])
    
    return tasks

