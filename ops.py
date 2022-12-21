import numpy as np

# <------ Operators ------>

# Discrete gradient
# Parameters:
#   img: (m,n)-shaped ndarray
# Return value: (2,m,n)-shaped ndarray
def grad(img):
    (m,n) = img.shape
    
    # Pad
    padded_x = np.zeros([m+1,n])
    padded_x[:m] = img
    padded_x[m] = img[-1]
    padded_y = np.zeros([m,n+1])
    padded_y[:,:n] = img
    padded_y[:,n] = img[:,-1]
    
    # Diff
    grad_x = np.diff(padded_x, axis=0)
    grad_y = np.diff(padded_y, axis=1)
    
    o = np.array([grad_x, grad_y])           
    return o

# Discrete divergence
# Parameter:
#   g: (2,m,n)-shaped ndarray
# Return value: (m,n)-shaped ndarray
def div(g):
    (m,n) = g.shape[-2:]
    
    # Pad
    g_x = np.zeros([m+1,n])
    g_x[1:m] = g[0,:m-1,:]
    g_y = np.zeros([m,n+1])
    g_y[:,1:n] = g[1,:,:n-1]
    
    # Diff
    div_from_x = np.diff(g_x, axis=0)
    div_from_y = np.diff(g_y, axis=1)
    
    o = div_from_x + div_from_y     
    return o

# Discrete total variation
# Parameters:
#   x: (m,n)-shaped ndarray
# Return value: float
def tv(x):
    (m,n) = x.shape
    grad_x = grad(x)
    
    sum = 0.0
    for i in range(m):
        for j in range(n):
            sum += np.linalg.norm(grad_x[:,i,j])
            
    return sum
