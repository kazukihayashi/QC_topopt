from math import sqrt
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import numpy as np
from LinearStructuralAnalysis import LinearStructuralAnalysis
from Draw import draw

'''

'''
nx=1
ny=1

# node [mm]
nn = (nx+1)*(ny+1)
node = np.zeros((nn,3),dtype=np.float64)
for i in range(nn):
    node[i,1], node[i,0] = np.divmod(i,nx+1)
node *= 250 

# member
nm = (1+4*ny)*nx+ny
connectivity = np.zeros((nm,2),dtype=np.int32)

count = 0
# horizontal member
for i in range(ny+1):
    for j in range(nx):
        connectivity[count,0] = i*(nx+1)+j
        connectivity[count,1] = i*(nx+1)+j+1
        count += 1
# vertical member
for i in range(ny):
    for j in range(nx+1):
        connectivity[count,0] = i*(nx+1)+j
        connectivity[count,1] = (i+1)*(nx+1)+j
        count += 1
# bracing member
for i in range(ny):
    for j in range(nx):
        connectivity[count,0] = i*(nx+1)+j
        connectivity[count,1] = (i+1)*(nx+1)+j+1
        count += 1
        connectivity[count,0] = i*(nx+1)+j+1
        connectivity[count,1] = (i+1)*(nx+1)+j
        count += 1

support = np.zeros_like(node,dtype=bool)
support[:,2] = True
for i in range(ny+1):
    support[(nx+1)*i] = True

# load [N]
load = np.zeros_like(node)
load[nx,1] = -10

def Member_Length(node,member):
    l = np.linalg.norm(node[member[:,1]]-node[member[:,0]],axis=1)
    l_x = node[member[:,1],0]-node[member[:,0],0]
    l_y = node[member[:,1],1]-node[member[:,0],1]
    return l, l_x, l_y

def Strain_Energy_H_A(L,E,elongation,scale=1.0):
    '''
    Compute the Hamiltonian of strain energy with respect to cross-sectional areas.
    '''
    h_diag_linear = np.zeros(len(L))
    H_quadratic = np.zeros((len(L),len(L)))
    for i in range(len(L)):
        h_diag_linear[i] = 0.5*E[i]*elongation[i]**2/L[i]
    return h_diag_linear*scale, H_quadratic*scale

def Total_Volume_Constraint_H(Vmax,L,scale=1.0):
    '''
    H = (L[0]*A[0]+L[1]*A[1]+･･･+L[m-1]*A[m-1] - Vmax)^2
    '''
    h_diag_linear = np.zeros(len(L))
    H_quadratic = np.zeros((len(L),len(L)))
    for i in range(len(L)):
        h_diag_linear[i] = -2*Vmax*L[i]
        H_quadratic[i,i] = L[i]**2
        for j in range(i+1,len(L)):
            H_quadratic[i,j] = 2*L[i]*L[j]
    return h_diag_linear*scale, H_quadratic*scale

def Area_Change_Penalty_H(A_before,scale=1.0):
    '''
    H = (A - A_before)^2
    '''
    h_diag_linear = np.zeros(len(A))
    H_quadratic = np.zeros((len(A),len(A)))
    for i in range(len(L)):
        h_diag_linear[i] = - 2*A_before[i]
        H_quadratic[i,i] = 1
    return h_diag_linear*scale, H_quadratic*scale

def Realize_Positive(H_quadratic,h_diag_linear,v_max,n_binary):
    '''
    Express real positive value 0 <= x <= v_max
    using (n_binary) binary variables.
    '''
    facs = np.power(2,np.arange(n_binary))
    facs_sum = np.sum(facs)

    Hp_diag_quadratic = np.zeros((n_binary,n_binary))
    for j in range(n_binary):
        Hp_diag_quadratic[j,j] = facs[j]**2
        for k in range(j+1,n_binary):
            Hp_diag_quadratic[j,k] = 2*facs[j]*facs[k]
    Hp_diag_quadratic *= v_max**2/facs_sum**2

    Hp_diag_linear = np.zeros((n_binary,n_binary))
    for j in range(n_binary):
        Hp_diag_linear[j,j] = facs[j]
    Hp_diag_linear *= v_max/facs_sum

    Hp_nondiag = np.zeros((n_binary,n_binary))
    for j in range(n_binary):
        for k in range(n_binary):
            Hp_nondiag[j,k] = facs[j]*facs[k]
    Hp_nondiag *= v_max**2/facs_sum**2

    H2 = np.zeros((H_quadratic.shape[0]*n_binary,H_quadratic.shape[0]*n_binary))
    for i in range(H_quadratic.shape[0]):
        H2[5*i:5*(i+1),5*i:5*(i+1)] = H_quadratic[i,i]*Hp_diag_quadratic + h_diag_linear[i]*Hp_diag_linear
        for j in range(i+1,H_quadratic.shape[0]):
            H2[5*i:5*(i+1),5*j:5*(j+1)] = H_quadratic[i,j]*Hp_nondiag
    
    return H2

def ToReal(q,v_max,n_binary):
    facs = np.power(2,np.arange(n_binary))
    r = np.zeros(len(q)//5)
    for i in range(len(r)):
        r[i] = v_max * np.sum(facs*q[5*i:5*(i+1)]) / np.sum(facs)
    return r

A = np.ones(nm)*50 # [N/mm2]
E = np.ones(nm)*5 # [N/mm2]
L,_,_ = Member_Length(node,connectivity)
Amax = 155 # [mm2]
Vmax = np.sum(A*L) # [mm3]
free = np.where(~support[:,0])[0]
n_binary = 5
n_iter = 30
n_top = 10 # How many solutions do we investigate in each iteration
V_penalty_scale = 100
A_penalty_scale = 1

A_history = np.zeros((n_iter,nm))
H_history = np.zeros((n_iter,3))
EV_history = np.zeros((n_iter+1,2))

d,s,r = LinearStructuralAnalysis(node,connectivity,support,load,A,E)
elongation = s*L/E
strain_energy_init = np.sum(0.5*E*A/L*elongation**2) # total strain energy obtained by structural analysis
EV_history[0] = strain_energy_init, np.sum(A*L)
# draw(node,connectivity,free,A,0)

for i in range(n_iter):
    A2 = np.copy(A)
    A2[A2<1e-3] = 1e-3 # avoid numerical instability
    d,s,r = LinearStructuralAnalysis(node,connectivity,support,load,A2,E)
    elongation = s*L/E

    h1_linear,H1_quadratic = Strain_Energy_H_A(L,E,elongation,scale=1/strain_energy_init)
    h2_linear,H2_quadratic = Total_Volume_Constraint_H(Vmax,L,scale=V_penalty_scale/Vmax**2)
    h3_linear,H3_quadratic = Area_Change_Penalty_H(A,scale=A_penalty_scale/np.sum(A**2))

    h_linear = - h1_linear + h2_linear + h3_linear
    H_quadratic = - H1_quadratic + H2_quadratic + H3_quadratic
    H = Realize_Positive(H_quadratic,h_linear,Amax,n_binary)
    Q = dict()
    for j1 in range(H.shape[0]):
        Q.update({(f"x{j1:0>3}", f"x{j1:0>3}"):H[j1,j1]})
        for j2 in range(j1):
            Q.update({(f"x{j1:0>3}", f"x{j2:0>3}"):H[j2,j1]})
    response = EmbeddingComposite(DWaveSampler(token="DEV-d233afa6c8183f64d10b844331d613d8b0dcf4cb")).sample_qubo(Q, num_reads=1000)
    # for sample, energy, num_occurrences, chain_break_fraction in list(response.data())[:10]:
    #     print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
    response_list = list(response.data())
    Es = np.array([response_list[i][1] for i in range(len(response_list))])
    Is = Es.argsort()[:n_top] # Indices of n_top solutions in view of the least energy
    print(f"Candidate indices:{Is}")
    Scores = np.empty(n_top)
    for j in range(n_top):
        q = np.array([response_list[Is[j]][0][f"x{k:0>3}"] for k in range(nm*n_binary)],dtype=np.float64)
        A = ToReal(q,Amax,n_binary)
        A2 = np.copy(A)
        A2[A2<1e-3] = 1e-3 # avoid numerical instability
        d,s,r = LinearStructuralAnalysis(node,connectivity,support,load,A2,E)
        elongation = s*L/E
        Scores[j] = np.sum(0.5*E*A/L*elongation**2) + V_penalty_scale*(np.sum(A*L)-Vmax)**2

    best_i = Is[Scores.argmin()]
    print(f"Best index:{best_i}")
    best_vars = response_list[best_i][0]
    q = np.array([best_vars[f"x{j:0>3}"] for j in range(nm*n_binary)],dtype=np.float64)
    A = ToReal(q,Amax,n_binary)
    print(f"A_values:{A}")
    A_history[i] = A
    H_values = np.zeros(3)
    H_values[0] = np.dot(h1_linear,A) + np.linalg.multi_dot([A,H1_quadratic,A])
    H_values[1] = np.dot(h2_linear,A) + np.linalg.multi_dot([A,H2_quadratic,A])
    H_values[2] = np.dot(h3_linear,A) + np.linalg.multi_dot([A,H3_quadratic,A])
    print(f"H_values:{H_values}")
    H_history[i] = H_values
    # draw(node,connectivity,free,A,i+1)
    EV_history[i+1] = H_values[0]*strain_energy_init, np.sum(A*L)

# np.savetxt("results/H_history.txt",H_history)
# np.savetxt("results/A_history.txt",A_history)
# np.savetxt("results/EV_history.txt",EV_history)

'''
Expected output is as follows:
{'x00': 0, 'x01': 1, 'x02': 0, 'x03': 1, 'x04': 1, 'y00': 0, 'y01': 1, 'y02': 0} Energy:  -508.0 Occurrences:  5
'''