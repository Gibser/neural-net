import numpy as np

def flatten_input(input, kernel_shape, s, p):
    n_filters, k_h, k_w, ch = kernel_shape
    H_out = (input.shape[1] + 2*p - k_h)//s + 1 
    W_out = (input.shape[2] + 2*p - k_w)//s + 1
    M = np.zeros((H_out*W_out*input.shape[0], k_h*k_w*ch), dtype=np.float32)
    for j in range(M.shape[0]):
        l = int(j / (H_out*W_out))
        p = j % (H_out*W_out)
        m = p % W_out
        t = int(p / W_out)
        isw = t*s
        ish = m*s
        M[j, :] = input[l, ish:ish+k_w, isw:isw+k_h, :].flatten()
    return M

'''
A = np.random.rand(4, 4)
print(A)
print(flatten_input(A, (1, 3, 3, 1), 1, 0))
'''