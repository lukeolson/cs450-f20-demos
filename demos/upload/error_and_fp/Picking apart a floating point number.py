#!/usr/bin/env python
# coding: utf-8

# # Picking apart a floating point number

# In[25]:


# Never mind the details of this function...

def pretty_print_fp(x):
    print("---------------------------------------------")
    print("Floating point structure for %r" % x)
    print("---------------------------------------------")
    import struct
    s = struct.pack("d", x)

    def get_bit(i):
        byte_nr, bit_nr = divmod(i, 8)
        return int(bool(
            s[byte_nr] & (1 << bit_nr)
            ))

    def get_bits(lsb, count):
        return sum(get_bit(i+lsb)*2**i for i in range(count))

    # https://en.wikipedia.org/wiki/Double_precision_floating-point_format

    print("Sign bit (1:negative):", get_bit(63))
    exponent = get_bits(52, 11)
    print("Stored exponent: %d" % exponent)
    print("Exponent (with offset): %d" % (exponent - 1023))
    fraction = get_bits(0, 52)
    if exponent != 0:
        significand = fraction + 2**52
    else:
        significand = fraction
    print("Significand (binary):", bin(significand)[2:])
    print("Shifted significand:", repr(significand / (2**52)))


# In[27]:


pretty_print_fp(3)


# Things to try:
# 
# * Twiddle the sign bit
# * 1,2,4,8
# * 0.5,0.25
# * $2^{\pm 1023}$, $2^{\pm 1024}$
# * `float("nan")`

# In[ ]:




