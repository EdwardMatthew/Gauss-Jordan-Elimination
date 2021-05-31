#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


# gauss-jordan elimination
def gauss_jordan(a,b):
    a = np.array(a,float)
    b = np.array(b,float)
    n = len(b)
    
    # main loop
    for k in range(n):
        # using partial pivoting
        # find pivot with the largest value in the left most column
        # swap rows until pivot is in the first row 
        if np.fabs(a[k,k]) < 1.0e-12:
            for i in range(k+1, n):
                # find the new pivot with the largest value in the second column
                # swap rows until pivot is in the second row
                if np.fabs(a[i,k]) > np.fabs(a[k,k]):
                    for j in range(k,n):
                        a[k,j],a[i,j] = a[i,j],a[k,j]
                    b[k],b[i] = b[i],b[k]
                    break
        # division of the pivot row
        pivot = a[k,k]
        for j in range(k,n):
            a[k,j] /= pivot
        b[k] /= pivot
        # elimination loop until RREF is achieved
        for i in range(n):
            if i == k or a[i,k] == 0: 
                continue
            factor = a[i,k]
            for j in range(k,n):
                a[i,j] -= factor * a[k,j]
            b[i] -= factor * b[k]
    return b,a


# In[5]:


a = [[0,2,0,1],[2,2,3,2],[4,-3,0,1],[6,1,-6,5]]
b = [0,-2,7,6]
x, A = gauss_jordan(a,b)
print("The solution is: ")
print(x)
print("The transformed matrix is")
print(A)


# In[ ]:




