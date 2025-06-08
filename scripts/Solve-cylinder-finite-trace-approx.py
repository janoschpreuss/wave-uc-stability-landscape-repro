import sys 
from ngsolve import *
from xfem import *
from xfem.lset_spacetime import *
from math import pi, log
from ngsolve.solvers import GMRes
from space_time import SpaceTimeMat
from space_time_trace import space_time_trace 
from meshes import GetMeshDataAllAround, GetMeshHalfCylinder
import numpy as np
np.random.seed(0) #fix random seed

tol = 1e-7 #GMRes
maxh_init = 0.4

# parameters for specifying geoemetry
R_Omega = 1.0 
R_data = 0.75
epsilon = 0.05
beta = 0.5
rr = 0.75
rho0 = rr**2 + beta**2
rho1 = (rr + beta)**2
rho  = rho0 + 1*(rho1-rho0)/10



delta = rho/2
tend = sqrt((rho1-rho)/(1-epsilon))
R = 1.0
alpha_idx = 0 
alphas = [1.0,0.75,0.5,0.25] 
alphas_strs = ["one","3quarter","half","quarter"]
alpha_scale = alphas[alpha_idx] 
alpha_str = alphas_strs[alpha_idx] 
rho *= alpha_scale 
print("rho = ", rho)
print("tend = ", tend)


max_ref_lvl = 4

# Space finite element order
order_global = 2
k = order_global
kstar = 1
q = order_global
qstar = 0
time_order = 2*max(q,qstar)

# stabiliation parameter
stabs = {"data": 1e4,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1, 
         "Tikh": 1e-15,
         "trace-proj": 1e0,
        }

Ns = [4,8,16,32,64,128]

M_modes = 2
m_sol = 2

meshes_list = [ ]
meshes_uncurved = [ ] 
for i in range(max_ref_lvl):
    mesh = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh = maxh_init,rename_bnd=True,order=order_global)
    Draw(mesh)
    for n in range(i):
        mesh.Refine()
    mesh.Curve(order_global)
    meshes_list.append(mesh)


def SolveProblem(ref_lvl,eta,eta_str):
    
    N = Ns[ref_lvl]
    #maxh = maxhs[ref_lvl]
    delta_t = 2*tend / N
    print("delta_t = ", delta_t)
    mesh = meshes_list[ref_lvl]

    # Level-set functions specifying the geoemtry
    told = Parameter(-tend)
    t = told + delta_t * tref

    def get_levelset(s):
        return s - (x-beta)**2 - y**2 + (1-epsilon)*t**2 

    B_lset = get_levelset(rho)
    Q_lset = get_levelset(delta)
    data_lset = x**2 +  y**2  - R_data**2 
    levelset = B_lset
    lset_Omega =  x**2 +  y**2  - R_Omega**2  
    
    # define exact solution
    t_slice = [ -tend + n*delta_t + delta_t*tref for n in range(N)]
    t_bottom = [-tend + n*delta_t  for n in range(N)]
    
    qpi = pi/4
    u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) 
                      + eta  *cos( sqrt(2) * (m_sol+1) * qpi * t_slice[n] ) * cos( (m_sol+1)  * qpi * x) * cos( (m_sol+1)  * qpi * y) 
                     for n in range(N)]

    ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) 
                      - eta * sqrt(2) * (m_sol+1) * qpi * sin( sqrt(2) * (m_sol+1) * qpi * t_slice[n] ) * cos( (m_sol+1) * qpi * x) * cos( (m_sol+1)  * qpi * y) 
                      for n in range(N)]

    trace_slice = [  [cos( sqrt(2) * m * qpi * t_slice[n] ) * cos(m * qpi * x) * cos(m * qpi * y) for m in range(1, M_modes+1) ] for n in range(N) ]
    trace_bottom = [  [cos( sqrt(2) * m * qpi * t_bottom[n] ) * cos(m * qpi * x) * cos(m * qpi * y) for m in range(1, M_modes+1) ] for n in range(N) ]
 
    bnd_names = {"Omega_BND":"bc_Omega|bc_axis",
                 "trace_BND":"bc_axis"}

    st = space_time_trace(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,bnd_names=bnd_names,
                    t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, 
                    trace_slice = trace_slice, trace_bottom = trace_bottom,   
                    told=told, perturbations=None,
                    )

    st.SetupSpaceTimeFEs()
    st.SetupRightHandSide()
    st.PreparePrecondGMRes()
    A_linop = SpaceTimeMat(ndof = st.X.ndof,mult = st.ApplySpaceTimeMat)
    PreTM = SpaceTimeMat(ndof = st.X.ndof,mult = st.TimeMarching)

    st.gfuX.vec.data = GMRes(A_linop, st.f.vec, pre=PreTM, maxsteps = 10000, tol = tol,
                  callback=None, restart=None, startiteration=0, printrates=True)
   
    for n in range(N):
        print("st.gfuX.components[2].components[{0}].vec = {1} ".format(n,st.gfuX.components[2].components[n].vec ))

    l2_error_B = st.MeasureErrors(st.gfuX,levelset,domain_type=NEG)
    print("st.MeasureErrors in B = ", l2_error_B ) 
    l2_errors_B_complement = st.MeasureErrors(st.gfuX,levelset,domain_type=POS)
    print("Errors in B complement = ", l2_errors_B_complement )
    l2_errors_omega = st.MeasureErrors(st.gfuX, data_lset, data_domain=True)
    print("Errors in omega = ", l2_errors_omega )
    l2_errors_Q = st.MeasureErrors(st.gfuX, data_lset, Q_all=True )
    print("Errors in Q (all) = ", l2_errors_Q) 
    
    return delta_t, l2_error_B, l2_errors_B_complement, l2_errors_omega, l2_errors_Q  
    #return l2_errors_omega 


eta_idx = 3 
etas = [1e-1,1e-2,1e-3,1e-4] 
etas_strs = ["ten","hundred","thousand","tenthousand"]

for eta_idx in range(4):
    eta = etas[eta_idx] 
    eta_str = etas_strs[eta_idx]

    errors = { "delta-t": [], 
               "B" : [ ],
              "B-complement": [],
              "omega" : [], 
              "Q_all" : [ ] 
              } 

    for ref_lvl in range( max_ref_lvl):
        plotting = False
        result = SolveProblem(ref_lvl,eta,eta_str)
        errors["delta-t"].append(result[0])
        errors["B"].append(result[1])
        errors["B-complement"].append(result[2])
        errors["omega"].append(result[3])
        errors["Q_all"].append(result[4])
        
        name_str = "Cylinder" + "-"  + "-q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}".format(kstar)+"msol{0}-Mmodes{1}-eta-{2}".format(m_sol+1, M_modes,eta_str)+".dat"
        results = [np.array(errors["delta-t"],dtype=float), np.array( errors["B"],dtype=float), np.array( errors["B-complement"] ,dtype=float), 
                   np.array( errors["omega"],dtype=float),  np.array( errors["Q_all"],dtype=float) ]
        header_str = "deltat L2-err-B L2-err-Bcompl L2-err-omega Qall"
        np.savetxt(fname ="../data/{0}".format(name_str),
                   X = np.transpose(results),
                   header = header_str,
                   comments = '')

