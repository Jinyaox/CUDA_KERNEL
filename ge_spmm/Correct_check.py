import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter, init
import torch.nn.functional as F 


"""torch::Tensor csr_spmm(
    torch::Tensor A_rowptr,
    torch::Tensor A_colind,
    torch::Tensor A_csrVal,
    torch::Tensor B
) """


"""this is the simple script to check if the ge-spmm is running and working"""

spmm = load(name='spmm', sources=['spmm.cpp', 'spmm_kernel.cu'], verbose=True)

a = torch.tensor([[0, 0, 1, 0],
                  [1, 2, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 3]], dtype = torch.float32)

sp = a.to_sparse_csr()

out = spmm.csr_spmm (sp.crow_indices().type(torch.int32),
                     sp.col_indices().type(torch.int32),
                     sp.values().type(torch.float32),
                     a)

print(out)
