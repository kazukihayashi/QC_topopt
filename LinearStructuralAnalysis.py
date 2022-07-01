import numpy as np
from numba import njit, f8, i4, b1
from numba.types import Tuple

CACHE = True
PARALLEL = False

bb1 = np.zeros((6,6),dtype=np.float64) # for linear stiffness matrix
bb1[0,0] = bb1[3,3] = 1
bb1[0,3] = bb1[3,0] = -1

bb2 = np.zeros((6,6),dtype=float) # for geometry stiffness matrix (Saka 1991 "Optimum design of geometrically nonlinear space trusses")
bb2[1,1] = bb2[2,2] = bb2[4,4] = bb2[5,5] = 1
bb2[1,4] = bb2[4,1] = bb2[2,5] = bb2[5,2] = -1

@njit(Tuple((f8[:,:,:],f8[:],f8[:,:]))(f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL)
def TransformationMatrices(node,member):
	'''
	(input)
	node[nn,3]<float>  : nodal locations (x,y coordinates) [mm]
	member[nm,3]<int>  : member connectivity

	(output)
	tt[nm,6,6]<float>  : transformation matrices (local to global)
	length[nm]<float>  : member lengths [mm]
	tp[nm,nn*3]<float> : transformation matrices (global to local)
	'''
	nn = np.shape(node)[0]
	nm = np.shape(member)[0]
	dxyz = np.zeros((nm,3),dtype=np.float64)
	length = np.zeros(nm,dtype=np.float64)
	for i in range(nm):
		dxyz[i] = node[member[i,1],:] - node[member[i,0],:]
		length[i] = np.linalg.norm(dxyz[i])
	tt = np.zeros((nm,6,6),dtype=np.float64)
	for i in range(nm):
		tt[i,0:3,0] = dxyz[i]/length[i]
	flag = np.abs(tt[:,0,0]) >= 0.9
	tt[flag,1,1] = 1.0
	tt[~flag,0,1] = 1.0
	for i in range(nm):
		for j in range(3):
			tt[i,j,2] = tt[i,(j+1)%3,0] * tt[i,(j+2)%3,1] - tt[i,(j+2)%3,0] * tt[i,(j+1)%3,1]
		tt[i,:,2] /= np.linalg.norm(tt[i,:,2])
		for j in range(3):
			tt[i,j,1] = tt[i,(j+1)%3,2] * tt[i,(j+2)%3,0] - tt[i,(j+2)%3,2] * tt[i,(j+1)%3,0]
	tt[:,3:,3:] = tt[:,:3,:3]

	tp = np.zeros((nm,nn*3),dtype=np.float64)
	for i in range(nm):
		indices = np.array((3*member[i,0],3*member[i,0]+1,3*member[i,0]+2,3*member[i,1],3*member[i,1]+1,3*member[i,1]+2),dtype=np.int32)
		for j in range(6):
			tp[i, indices[j]] -= tt[i, j, 0]
			tp[i, indices[j]] += tt[i, j, 3]

	return tt, length, tp


@njit(Tuple((f8[:,:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:],f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def LinearStiffnessMatrix(node,member,support,A,E,L,tt):
	'''
	(input)
	node[nn,3]: Nodal coordinates
	member[nm,2]: Member connectivity
	support[nn,3]: True if supported, else False
	A[nm]: Cross-sectional area.
	E[nm]: Young's modulus.
	L[nm]: Member lengths.
	tt[nm,6,6]: transformation matrices.

	(output)
	Kl_free[nn,nn]: Global linear stiffness matrix with respect to DOFs only
	Kl[nn,nn]: Global linear stiffness matrix
	'''

	## Organize input model
	nn = node.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	tt = np.ascontiguousarray(tt)

	## Linear element stiffness matrices
	kl_el = np.zeros((nm,6,6),dtype=np.float64)
	for i in range(nm):
		kl_el[i] = np.dot(tt[i],E[i]*A[i]/L[i]*bb1)
		kl_el[i] = np.dot(kl_el[i],tt[i].transpose())

	## Assembling element matrices to the global matrix
	Kl = np.zeros((3*nn,3*nn),np.float64)
	for i in range(nm): # assemble element matrices into one matrix
		Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kl_el[i,0:3,0:3]
		Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kl_el[i,0:3,3:6]
		Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kl_el[i,3:6,0:3]
		Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kl_el[i,3:6,3:6]

	Kl_free = Kl[free][:,free] # Extract DOFs	

	return Kl_free, Kl

@njit(Tuple((f8[:,:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def GeometryStiffnessMatrix(node,member,support,N,L,tt):
	'''
	(input)
	node[nn,3]: Nodal coordinates
	member[nm,2]: Member connectivity
	support[nn,3]: True if supported, else False
	N[nm]: Axial forces (positive for tension, negative for compression).
	L[nm]: Member lengths.
	tt[nm,6,6]: transformation matrices.

	(output)
	Kg_free[nn,nn]: Global geometry stiffness matrix with respect to DOFs only
	Kg[nn,nn]: Global geometry stiffness matrix
	'''
	## Organize input model
	nn = node.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	tt = np.ascontiguousarray(tt)

	## Geometry element stiffness matrices
	kg_el = np.zeros((nm,6,6),dtype=np.float64)
	for i in range(nm):
		kg_el[i] = np.dot(tt[i],bb2*N[i]/L[i])
		kg_el[i] = np.dot(kg_el[i],tt[i].transpose())

	## Assembling element matrices to the global matrix
	Kg = np.zeros((3*nn,3*nn),np.float64) # geometry stiffness matrix
	for i in range(nm): # assemble element matrices into one matrix
		Kg[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kg_el[i,0:3,0:3]
		Kg[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kg_el[i,0:3,3:6]
		Kg[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kg_el[i,3:6,0:3]
		Kg[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kg_el[i,3:6,3:6]

	Kg_free = Kg[free][:,free]

	return Kg_free, Kg

@njit(f8[:,:](f8[:],b1[:],i4),cache=CACHE,parallel=PARALLEL)
def Reshape_freevec_to_n3(vec,free,nn):
	extended_vec = np.zeros(nn*3,dtype=np.float64)
	extended_vec[free] = vec
	mat = extended_vec.reshape((nn,np.int32(3)))
	return mat

@njit(Tuple((f8[:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def StiffnessMatrixEig(node0,member,support,A,E):
	'''
	(input)
	node[nn,3]: nodal coordinates
	member[nm,2]: member connectivity
	support[nn,3]: True if supported, else False
	  (note) Isolated nodes can be ignored by setting the "support" values associated with them to True.
	A[nm]: Cross-sectional area.
	  (note) Assign exactly 0 to vanishing members so as to correctly compute the rank of the stifness matrix

	(output)
	eig_vals[nDOF]: eigen-values
	eig_modes[nDOF,nn,3]: eigen-modes
	'''

	## Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False

	## Transformation matrices (tt) and initial lengths (ll0)
	tt,ll0,tp = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)

	## Linear stiffness matrix 
	Kl_free, _ = LinearStiffnessMatrix(node0,member,support,A,E,ll0,tt)
	eig_vals,eig_vecs = np.linalg.eigh(Kl_free)

	## Reshape eig_vecs to obtain eigen-modes
	u = np.zeros((len(eig_vals),nn*3),dtype=np.float64)
	for i in range(len(eig_vals)):
		uu = u[i] # This is a shallow copy, and u also changes in the next line
		uu[free] = eig_vecs[:,i]
	eig_modes = u.reshape((len(eig_vals),nn,3))

	return eig_vals,eig_modes

@njit(Tuple((f8[:,:],f8[:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def LinearStructuralAnalysis(node0,member,support,load,A,E):

	'''
	(input)
	node[nn,3]: Nodal coordinates
	member[nm,2]: Member connectivity
	support[nn,3]: True if supported, else False
	load[nn,3]: Load magnitude. 0 if no load is applied.
	A[nm]: Cross-sectional area.
	E[nm]: Young's modulus.

	(output)
	deformation[nn,3]: nodal deformations
	stress[nm]: member stresses
	reaction[nn,3]: reaction forces
	  (note): Only supported coordinates can take a non-zero value. The other coordinates (i.e., DOFs) takes 0.
	'''

	## Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	pp = load.flatten()[free].astype(np.float64)

	## Transformation matrices (tt) and initial lengths (ll0)
	tt,ll0,tp = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)
	tp = np.ascontiguousarray(tp)

	## Linear stiffness matrix 
	Kl_free, Kl = LinearStiffnessMatrix(node0,member,support,A,E,ll0,tt)

	## Solve the stiffness equation (Kl_free)(Up) = (pp) to obtain the deformation
	Up = np.linalg.solve(Kl_free,pp) # Compute displacement Up (size:nDOF), error occurs at this point when numba is not in use
	# Up = sp_solve(Kl_free,pp,assume_a ='sym',check_finite=False) # Use this for better precision when numba is not in use

	## Deformation and stresses
	deformation = Reshape_freevec_to_n3(Up,free,nn)
	U = np.zeros(nn*3,dtype=np.float64) # Displacement vector U (size:nn)
	U[free] = Up # Synchronize U to Up
	stress = np.dot(tp,U)*E/ll0 # axial forces
	
	## Reaction forces
	Rp = np.dot(Kl[~free][:,free],Up)
	R = np.zeros(nn*3,dtype=np.float64)
	R[~free] = Rp
	R[~free] -= load.flatten()[~free]
	reaction = R.reshape((nn,3))

	return deformation, stress, reaction

@njit(Tuple((f8[:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def LinearBucklingAnalysis(node0,member,support,load,A,E):
	'''
	(input)
	node[nn,3]: Nodal coordinates
	member[nm,2]: Member connectivity
	support[nn,3]: True if supported, else False
	load[nn,3]: Load magnitude. 0 if no load is applied.
	A[nm]: Cross-sectional area.
	E[nm]: Young's modulus.

	(output)
	eig_vals[nDOF]: eigen-values (load factors that cause the buckling)
	eig_modes[nDOF,nn,3]: eigen-modes (buckling modes)
	'''

	## Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	pp = load.flatten()[free].astype(np.float64)

	## Transformation matrices (tt) and initial lengths (ll0)
	tt,ll0,tp = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)
	tp = np.ascontiguousarray(tp)

	## Linear stiffness matrix 
	Kl_free, _ = LinearStiffnessMatrix(node0,member,support,A,E,ll0,tt)

	## Solve the stiffness equation (Kl_free)(Up) = (pp) to obtain the deformation
	Up = np.linalg.solve(Kl_free,pp) # Compute displacement Up (size:nDOF), error occurs at this point when numba is not in use
	# Up = sp_solve(Kl_free,pp,assume_a ='sym',check_finite=False) # Use this for better precision when numba is not in use

	## Deformed shape and forces
	U = np.zeros(nn*3,dtype=np.float64) # Displacement vector U (size:nn)
	U[free] = Up # Synchronize U to Up

	'''
	## The relationship between external loads and internal forces is not linear if computing "force" using the following equations:
	# node = node0 + deformation # Deformed shape
	# _,ll,_ = TransformationMatrices(node,member) # Recompute member lengths
	# force = A*E*(ll-ll0)/ll0 # tensile forces are positive, compressive forces are negative
	'''

	N = np.dot(tp,U)*E*A/ll0 # axial forces

	## Linear stiffness matrix 
	Kg_free, _ = GeometryStiffnessMatrix(node0,member,support,N,ll0,tt)

	## Solve the eigenvalue problem
	eig_vals_comp, eig_vecs_comp = np.linalg.eig(np.dot(np.ascontiguousarray(-np.linalg.inv(Kg_free)),np.ascontiguousarray(Kl_free)))
	# eig_vals_comp, eig_vecs_comp = sp_eig(-Kl_free,Kg_free) # Use this for better precision, but numba cannot be used
	eig_vals = eig_vals_comp.real.astype(np.float64) # Extract real numbers
	eig_vecs = eig_vecs_comp.real.astype(np.float64) # Extract real numbers
	eig_modes = np.empty((len(eig_vals),nn,3),dtype=np.float64)
	for i in range(len(eig_vals)):
		eig_modes[i] = Reshape_freevec_to_n3(eig_vecs[:,i],free,nn)

	return eig_vals, eig_modes


# node = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]],dtype=float) # 節点座標 [m]
# connectivity = np.array([[0,1],[1,2],[3,4],[4,5],[1,4],[2,5],[0,4],[1,5]],dtype=int) # どの節点同士を部材でつなげるか
# support = np.zeros((node.shape[0],3),dtype=bool)
# support[[0,3]] = True
# support[:,2] = True
# load = np.zeros((node.shape[0],3),dtype=np.float64)
# load[2,1] = -1000
# A = np.ones(connectivity.shape[0])*1e-4
# E = np.ones(connectivity.shape[0])*2.05e11
# d,s,r = LinearStructuralAnalysis(node,connectivity,support,load,A,E)
# print(d)

# node = np.array([[0,0,0],[1,0,0]],dtype=np.float64)
# connectivity = np.array([[0,1]],dtype=np.int32)
# support = np.array([[1,1,1],[1,1,1]],dtype=bool)
# load = np.array([[1,0,0],[0,0,0]],dtype=np.float64)
# A = np.array([1.0],dtype=np.float64)
# E = np.array([1.0],dtype=np.float64)

# node = np.array([[0,0,0],[8,0,0],[4,4,0]],dtype=np.float64)
# connectivity = np.array([[0,1],[0,2],[1,2]],dtype=np.int32)
# support = np.array([[1,1,1],[0,1,1],[0,0,1]],dtype=bool)
# load = np.array([[0,0,0],[0,0,0],[0,-1,0]],dtype=np.float64)
# A = np.array([1.0,1.0,1.0],dtype=np.float64)
# E = np.array([1.0,1.0,1.0],dtype=np.float64)

# d,s,r = LinearStructuralAnalysis(node,connectivity,support,load,A,E)

# import time
# t1 = time.perf_counter()
# for i in range(100):
# 	d,s,c = StructuralAnalysis(node,connectivity,support,load,A,E)
# t2 = time.perf_counter()
# print("d={0}".format(d))
# print("s={0}".format(s))
# print("time={0}".format(t2-t1))

# node = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]],dtype=np.float64)
# connectivity = np.array([[0,1],[1,2],[3,4],[4,5],[1,4],[2,5],[0,4],[1,3],[1,5],[2,4]],dtype=np.int32)
# support = np.array([[1,1,1],[0,0,1],[0,0,1],[1,1,1],[0,0,1],[0,0,1]],dtype=bool)
# A = np.array([1,0,1,1,1,1,1,1,0,1],dtype=np.float64)
# E = np.ones(connectivity.shape[0],dtype=np.float64)
# eig_val, eig_mode = StiffnessMatrixEig(node,connectivity,support,A,E)