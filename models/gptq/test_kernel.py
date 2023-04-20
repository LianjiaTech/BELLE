import torch
import torch.nn as nn

import quant_cuda
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking LLaMa-7B FC2 matvec ...')

DEV = torch.device('cuda:0')

B = 5
L = 128
M = 4096
N = 11008

DTYPE = torch.half
mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
vec = torch.randn((B, M), device=DEV, dtype=DTYPE)
mul = torch.zeros((B, N), device=DEV, dtype=DTYPE)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    torch.matmul(vec, mat, out=mul) 
    torch.cuda.synchronize()
print('FP16:', (time.time() - tick) / COUNT)

DTYPE = torch.float
mat = mat.to(DTYPE)
vec = vec.to(DTYPE)
mul = mul.to(DTYPE)

mat = torch.randint(-1000000000, 1000000000, (M // 256 * 32, N), device=DEV, dtype=torch.int)
scales = torch.randn(N, device=DEV, dtype=DTYPE)
zeros = torch.randint(-1000000000, 1000000000, (1, N // 256 * 32), device=DEV, dtype=torch.int)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant2matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
print('2bit:', (time.time() - tick) / COUNT)

tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant3matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
print('3bit:', (time.time() - tick) / COUNT)

tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant4matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
print('4bit:', (time.time() - tick) / COUNT)

tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant8matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
print('8bit:', (time.time() - tick) / COUNT)
print('Verifiying kernel correctness ...')

M = 4096
N = 11008

from quant import *

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(2, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(2, -1, layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    print('2bit Simu:', qlayer(vec))
    print('2bit Kern:', layer.to(DEV)(vec))
    print('\n')

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(3, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(3, -1, layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    print('3bit Simu:', qlayer(vec))
    print('3bit Kern:', layer.to(DEV)(vec))
    print('\n')

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(4, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(4, -1, layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV) 

with torch.no_grad():
    print('4bit Simu:', qlayer(vec))
    print('4bit Kern:', layer.to(DEV)(vec))
    print('\n')

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(8, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(8, -1, layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    print('8bit Simu:', qlayer(vec))
    print('8bit Kern:', layer.to(DEV)(vec))
