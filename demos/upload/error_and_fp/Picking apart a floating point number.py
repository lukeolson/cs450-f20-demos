#!/usr/bin/env python
# coding: utf-8

# # Picking apart a floating point number

# https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Double-precision_examples

# In[ ]:


import struct

def double2bin(x, precision='double'):
    if precision == 'double':
        s = struct.unpack(">Q", struct.pack(">d", x))[0]  # pack as a double, unpack as unsigned long long
        b = f'{s:064b}' # print to string as binary and force 64 bits
    
    if precision == 'single':
        s = struct.unpack(">L", struct.pack(">f", x))[0]  # pack as a single, unpack as unsigned long
        b = f'{s:032b}' # print to string as binary and force 32 bits

    return b

def printbits(b):
    if len(b) == 32:
        esize = 8
        offset = -127
    elif len(b) == 64:
        esize = 11
        offset = -1023
    else:
        raise ValueError('only 32 or 64 bit')
    sign = b[0]
    exponent = b[1:1+esize]
    significand = b[1+esize:]

    print(f'          Sign bit: {sign}')
    print(f'(shifted) Exponent: {exponent} ({int(exponent, 2)} -> {offset+int(exponent, 2)})')
    print(f'       Significand: 1.{significand}')
    print("                      |         |         |         |         |         | ")
    print("                      0         1         2         3         4         5 ")
    print("                      0123456789012345678901234567890123456789012345678901")


# In[ ]:


printbits(double2bin(0.25, precision='single'))


# Things to try:
# 
# * Twiddle the sign bit
# * 1,2,4,8
# * 0.5,0.25
# * $2^{\pm 1023}$, $2^{\pm 1024}$
# * `float("nan")`
