cpdef list get_bad_idx(int p, int n, int g):
    cdef int a = g
    cdef list bad_idx = list()
    for idx in range(p-3):
        if a >= n+2: bad_idx += [idx]
        a = (a*g) % p
    return bad_idx