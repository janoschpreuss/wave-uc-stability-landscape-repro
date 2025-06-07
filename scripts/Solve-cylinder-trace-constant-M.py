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
print("rho = ", rho)
rho *= alpha_scale 
print("rho = ", rho)
print("tend = ", tend)


#tend = 1/50

max_ref_lvl = 3
#max_ref_lvl = 2

# Space finite element order
order_global = 2
k = order_global
kstar = 1
q = order_global
qstar = 0
time_order = 2*max(q,qstar)

# case of perturbed data 
perturbations = { }
perturb = False 
if perturb:
    poly_t_order = 4
    perturbations["exponent"] = 1
    perturbations["scal"] = 4e-1
    perturbations["maxh"] = maxh_init 

# stabiliation parameter
stabs = {"data": 1e4,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1, 
         "Tikh": 1e-5,
         "trace-proj": 1e0,
        }

Ns = [4,8,16,32,64,128]

meshes_list = [ ]
meshes_uncurved = [ ] 
for i in range(max_ref_lvl):
    mesh = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh = maxh_init,rename_bnd=True,order=order_global)
    Draw(mesh)
    for n in range(i):
        mesh.Refine()

    mesh.Curve(order_global)
    meshes_list.append(mesh)



def SolveProblem(ref_lvl,M_modes=2, m_sol=2, min_space = False ):
    
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
    u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]

    ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]

    f_ccc = [  cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y)  for n in range(N)]
    f_scc = [  sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y)  for n in range(N)]
    f_csc = [  cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * sin(m_sol  * qpi * x) * cos(m_sol  * qpi * y)  for n in range(N)]
    f_ccs = [  cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * sin(m_sol  * qpi * y)  for n in range(N)]
    f_ssc = [  sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * sin(m_sol  * qpi * x) * cos(m_sol  * qpi * y)  for n in range(N)]
    f_scs = [  sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * sin(m_sol  * qpi * y)  for n in range(N)]
    f_css = [  cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * sin(m_sol  * qpi * x) * sin(m_sol  * qpi * y)  for n in range(N)]
    f_sss = [  sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * sin(m_sol  * qpi * x) * sin(m_sol  * qpi * y)  for n in range(N)]

    if min_space: 
        trace_slice = [  [cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol * qpi * y)   ] for n in range(N) ]
        trace_bottom = [  [cos( sqrt(2) * m_sol * qpi * t_bottom[n] ) * cos(m_sol * qpi * x) * cos(m_sol * qpi * y) ] for n in range(N) ]
    else:
        trace_slice = [  [cos( sqrt(2) * m * qpi * t_slice[n] ) * cos(m * qpi * x) * cos(m * qpi * y) for m in range(1, M_modes+1) ] for n in range(N) ]
        trace_bottom = [  [cos( sqrt(2) * m * qpi * t_bottom[n] ) * cos(m * qpi * x) * cos(m * qpi * y) for m in range(1, M_modes+1) ] for n in range(N) ]


    
    bnd_names = {"Omega_BND":"bc_Omega|bc_axis",
                 "trace_BND":"bc_axis"}

    st = space_time_trace(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,bnd_names=bnd_names,
                    t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, 
                    trace_slice = trace_slice, trace_bottom = trace_bottom,   
                    told=told, perturbations=perturbations,
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

    norm_slab = []  
    n = 0 
    st.told.Set(st.tstart)
    norms_f = { "ccc": [],
                "scc": [],
                "csc": [],
                "ccs": [],
                "ssc": [],
                "scs": [],
                "css": [],
                "sss": []
              }

    while st.tend - st.told.Get() > st.delta_t / 2:
        
        norms_f["ccc"].append(  Integrate(  (  f_ccc[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["scc"].append(  Integrate(  (  f_scc[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["csc"].append(  Integrate(  (  f_csc[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["ccs"].append(  Integrate(  (  f_ccs[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["ssc"].append(  Integrate(  (  f_ssc[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["scs"].append(  Integrate(  (  f_scs[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["css"].append(  Integrate(  (  f_css[n]**2 )  * st.dxt_ho, st.mesh) )    
        norms_f["sss"].append(  Integrate(  (  f_sss[n]**2 )  * st.dxt_ho, st.mesh) )    

        st.told.Set(st.told.Get() + st.delta_t)
        n += 1 
    

    norm_H2 =  (1+4*m_sol**2*qpi**2 ) * sqrt( sum( norms_f["ccc"] )  )
    norm_H2 += ( m_sol*qpi*sqrt(2) + 6 * sqrt(2)*(m_sol*qpi)**3 ) * sqrt( sum( norms_f["scc"] ) )
    norm_H2 += ( m_sol*qpi +  (3*sqrt(2) + 4) *(m_sol*qpi)**3  ) * sqrt( sum( norms_f["csc"] )  )
    norm_H2 += ( m_sol*qpi +  (3*sqrt(2) + 4) *(m_sol*qpi)**3  ) * sqrt( sum( norms_f["ccs"] )  )
    norm_H2 +=  ( 2*sqrt(2)* (m_sol*qpi)**2 ) * sqrt( sum( norms_f["ssc"] )  )
    norm_H2 +=  ( 2*sqrt(2)* (m_sol*qpi)**2 ) * sqrt( sum( norms_f["scs"] )  )
    norm_H2 +=  ( 2 * (m_sol*qpi)**2 ) * sqrt( sum( norms_f["css"] )  )
    norm_H2 +=  ( sqrt(2) * (m_sol*qpi)**3 ) * sqrt( sum( norms_f["sss"] )  )

    print("m_sol = {0}, norm_H2 = {1}".format(m_sol,norm_H2))  
    st.told.Set(st.tstart)

    return l2_errors_Q / ( norm_H2 * st.delta_t**3 )  

  
consts = [] 
Mmax = 12
Ms = np.array([m for m in range(1,Mmax+1)]) 

for m in Ms:
    consts.append(SolveProblem(ref_lvl=max_ref_lvl-1, M_modes=m, m_sol=m,min_space = False  ))
    
if True:
    name_str = "Cylinder-Const" + "-"  + "-q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}".format(kstar)+"msol-eq-Mmodes"+".dat"
    results = [np.array(Ms,dtype=float), np.array(consts, dtype=float)  ]
    header_str = "M Const"
    np.savetxt(fname ="../data/{0}".format(name_str),
               X = np.transpose(results),
               header = header_str,
               comments = '')

consts = [] 
Ms = np.array([m for m in range(1,Mmax+1)]) 

for m in Ms:
    consts.append(SolveProblem(ref_lvl=max_ref_lvl-1, M_modes=m, m_sol=m,min_space = True  ))
    
if True:
    name_str = "Cylinder-Const" + "-"  + "-q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}".format(kstar)+"msol-var-Mmodes-1"+".dat"
    results = [np.array(Ms,dtype=float), np.array(consts, dtype=float)  ]
    header_str = "M Const"
    np.savetxt(fname ="../data/{0}".format(name_str),
               X = np.transpose(results),
               header = header_str,
               comments = '')

#consts = [] 
#Ms = np.array([m for m in range(1,Mmax+1)]) 

#for m in Ms:
#    consts.append(SolveProblem(ref_lvl=max_ref_lvl-1, M_modes=m, m_sol=1,min_space = False  ))
    
#if True:
#    name_str = "Cylinder-Const" + "-"  + "-q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}".format(kstar)+"msol-1-Mmodes-var"+".dat"
#    results = [np.array(Ms,dtype=float), np.array(consts, dtype=float)  ]
#    header_str = "M Const"
#    np.savetxt(fname ="../data/{0}".format(name_str),
#               X = np.transpose(results),
#               header = header_str,
#               comments = '')
