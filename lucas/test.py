n_nd=3

boundary_nodes = [0]          # example: cantilever root
B = []
for i in boundary_nodes:
    B.extend(range(6*i, 6*i+6))
print(B)


all_dofs = list(range(6*n_nd))
I = [k for k in all_dofs if k not in B]

print(I)