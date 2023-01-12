import torch
import torch.utils.benchmark as benchmark
import csv
import numpy as np
from torch.utils.cpp_extension import load
from torch.nn import Parameter, init
import torch.nn.functional as F

class methods():
    def __init__(self,name,fun,label:int):
        self.name=name
        self.func=fun
        self.label=label

def generate_size(max_size:int,stride=2)->list:
    result=[16]
    initial=16
    for i in range(16,max_size+1):
        result.append(initial+initial)
        initial*=2
    return result

def generate_density(base=2,exp=8)->float:
    res=[]
    for i in range(exp+1):
        res.append(1/(base**i))
    return res

def build_methods(name:list,func:list,csr:list)->list:
    res=[]
    for i in range(len(func)):
        res.append(methods(name[i],func[i],csr[i]))
    return res

def timing(k,m1,m2,sput=0):
    if(sput==0):
        t = benchmark.Timer(
            stmt=k.func,
            setup='from __main__ import {}'.format(k.name),
            globals={'m1': m1,'m2':m2})
    else:
        rowind=(torch.argsort(torch.diff(m1.crow_indices()),descending=False)).type(torch.int32)
        
        t = benchmark.Timer(
            stmt=k.func,
            setup='from __main__ import {}'.format(k.name),
            globals={'m1': m1,'m2':m2,'rowind':rowind})

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        time=t.timeit(30)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return time.mean
    
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise RuntimeError(e)
        torch.cuda.empty_cache()
        return float(-1)
        
def time_testing(size:list,density:list,methods:list)->None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('result.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["size","density","method","time"])
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
                        csvwriter.writerow([str(i),str(j),k.name,str(t)])
                    if k.label==3:                  #use csr tensor
                        t=timing(k,sparse_csr,dense_t,sput=1)
                        csvwriter.writerow([str(i),str(j),k.name,str(t)])
                    elif k.label==2:                #use dense tensor
                        t=timing(k,dense,dense_t)
                        csvwriter.writerow([str(i),str(j),k.name,str(t)])
                    else:                           #use COO tensor
                        t=timing(k,sparse_coo,dense_t)
                        csvwriter.writerow([str(i),str(j),k.name,str(t)])
    return

if __name__ == '__main__':
    #==========define your testing functions/methods below============
    def torch_matmul(m1,m2):
        return torch.matmul(m1,m2)

    def torch_spmm(m1,m2):
        return torch.sparse.mm(m1, m2)

    #def ge_spmm(m1,m2):
    #    return sp.csr_spmm (m1.crow_indices().type(torch.int32),
    #                     m1.col_indices().type(torch.int32),
    #                     m1.values().type(torch.float32),
    #                     m2.type(torch.float32))

    def sputnik(m1,m2,rowind):
        return spmm.cuda_spmm(rowind,
                              m1.values().type(torch.float32),
                              m1.crow_indices().type(torch.int32),
                              m1.col_indices().type(torch.int32),
                              m2)

    

    name=["torch_matmul","torch_spmm","sputnik"]
    f=['torch_matmul(m1,m2)','torch_spmm(m1,m2)','sputnik(m1,m2,rowind)']
    csr=[2,0,3]
    #==========End of function/methods definitions============

    
    size            =generate_size(4000)
    density         =generate_density()
    methods         =build_methods(name,f,csr)

    #==========Put Needed Kernels below for initial compilation=============
    #sp = load(name='spmm', sources=['spmm.cpp', 'spmm_kernel.cu'], verbose=True)

    spmm = load(name='spmm', 
            sources=['cu_spmm.cpp', 'cuda_spmm.cu'],
            extra_include_paths=['/home/jinyaox/sputnik'],
            verbose=True)
    
    #==========End of Kernel Compilation=============

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=torch.device("cuda")).sum()

    time_testing(size,density,methods)
