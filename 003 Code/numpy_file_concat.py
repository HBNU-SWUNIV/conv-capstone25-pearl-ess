import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os

numpy_file = glob('/root/ssd/Yeonseo/ESS/data/numpy_file_Blind/*')
numpy_file.sort()
dic_name = dict()

for npy in numpy_file:
    all_file = glob(npy+'/*.npy')
    all_file.sort()
    file_name = list()
    for i in range(len(all_file)):
        compare = all_file[i].split("/")[-1].split("_")
        part = '_'.join(compare[:2])
        file_name.append(part)
    file_name = set(file_name)
    dic_name[npy.split("/")[-1]] = file_name


for npy in tqdm(numpy_file, 'file success'):
    all_file = glob(npy+"/*.npy")
    all_file.sort()
    ess_name = npy.split("/")[-1]
    if ess_name in dic_name.keys():
        file_list = dic_name[ess_name]
    
    for r_name in tqdm(file_list, 'rack module compare'):
        X = []
        for A_name in all_file:
            if r_name in A_name:
                N = np.load(A_name, allow_pickle=True)
                if len(X) == 0:
                    X = N
                else:
                    X = np.concatenate((X, N), axis=0)
        path = f'/root/ssd/Yeonseo/ESS/data/cel_concat/blind/{npy.split("/")[-1]}'
        os.makedirs(path, exist_ok=True)
        np.save(path+f'/{r_name}.npy', X)