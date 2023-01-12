import torch
import time
import csv
import numpy as np
from torch.utils.cpp_extension import load
import pandas as pd
from torch.nn import Parameter, init
import torch.nn.functional as F

#==========define your testing functions/methods below============
def torch_matmul(m1,m2):
    return torch.matmul(m1,m2)

def torch_spmm(m1,m2):
    return torch.sparse.mm(m1, m2)

def ge_spmm(m1,m2):
    return spmm.csr_spmm (m1.crow_indices().type(torch.int32),
                     m1.col_indices().type(torch.int32),
                     m1.values().type(torch.float32),
                     m2.type(torch.float32))

name=["torch_matmul","torch_spmm","ge-spmm"]
f=[torch_matmul,torch_spmm,ge_spmm]
csr=[2,0,1,0]

#==========End of function/methods definitions============
class methods():
    def __init__(self,name,fun,label:int):
        self.name=name
        self.func=fun
        self.label=label

    def __call__(self,m,m_d):
        #the first matrix can be sparse or dense
        #the second matrix must be dense 
        self.func(m,m_d)

def generate_size(max_size:int,stride=50)->list:
    return [i for i in range(10,max_size+1,stride)]

def generate_density(max_dense:float,stide=0.05)->float:
    assert max_dense<=1, "max density must be smaller or eq to 1"
    
    return [i for i in np.arange(0.0,max_dense,stide)]

def build_methods(name:list,func:list,csr:list)->list:
    res=[]
    for i in range(len(func)):
        res.append(methods(name[i],func[i],csr[i]))
    return res

def timing(k,m1,m2):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t = time.perf_counter()
        k(m1,m2)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return time.perf_counter() - t
    
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise RuntimeError(e)
        torch.cuda.empty_cache()
        return float(-1)
        
def time_testing(size:list,density:list,methods:list,iteration=1)->None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('result.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["iteration","size","density","method","time"])
        for it in range(iteration):
            for i in size:
                for j in density:
                    #create the target dense matrix
                    mask_t=(torch.empty(i,i).bernoulli_(j)).to(device)
                    dense_t=((torch.randn(i,i).to(device))*mask_t)
                    
                    #creating the testing sparse matrix
                    mask=(torch.empty(i,i).bernoulli_(j)).to(device)
                    dense=((torch.randn(i,i).to(device))*mask)
                    sparse_csr=dense.to_sparse_csr()
                    sparse_coo=dense.to_sparse()

                    for k in methods:        
                        if k.label==1:                  #use csr tensor
                            t=timing(k,sparse_csr,dense_t)
                            csvwriter.writerow([str(it),str(i),str(j),k.name,str(t)])
                        elif k.label==2:                #use dense tensor
                            t=timing(k,dense,dense_t)
                            csvwriter.writerow([str(it),str(i),str(j),k.name,str(t)])
                        else:                           #use COO tensor
                            t=timing(k,sparse_coo,dense_t)
                            csvwriter.writerow([str(it),str(i),str(j),k.name,str(t)])
    return



if __name__ == '__main__':
    size            =generate_size(3500,50)
    density         =generate_density(1)
    methods         =build_methods(name,f,csr)

    #==========Put Needed Kernels below for initial compilation=============
    spmm = load(name='spmm', sources=['spmm.cpp', 'spmm_kernel.cu'], verbose=True)

    #==========End of Kernel Compilation=============

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=torch.device("cuda").sum()

    time_testing(size,density,methods)

    #==========Do Panda Data Analysis and draw 2 graphs one =============
    #df=pd.read_csv("result.csv")
    #df = df.astype({'method':'string'})
    
    
    
                    

