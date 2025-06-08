import sys 
from ngsolve import *
from xfem import *
from xfem.lset_spacetime import *
from math import pi, log
from ngsolve.solvers import GMRes
from space_time import space_time, SpaceTimeMat
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

rr = R_data
rho0 = rr**2 + beta**2
rho1 = (rr + beta)**2
#print("rho0 = {0}, rho1 = {1}".format( rho0,rho1 ) )
R = 1.0
alphas = [1.0,0.75,0.5,0.25] 
alphas_strs = ["one","3quarter","half","quarter"]
max_ref_lvl = 5


# case of perturbed data 
perturbations = { }
perturb = False 
if perturb:
    poly_t_order = 4
    perturbations["exponent"] = 1
    #perturbations["scal"] = 7e-2
    perturbations["scal"] = 4e-1
    perturbations["maxh"] = maxh_init 

# stabiliation parameter
stabs = {"data": 1e4,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1,
         "Tikh": 1e-5
        }

Ns = [4,8,16,32,64,128]
m_sol = 2

if perturb:
    nx = 50
    ny = 50
    values = np.array([-1 + 2*np.random.rand() for y in np.linspace(0,2*np.pi,ny) for x in np.linspace(0,2*np.pi, nx)])
    values = values.reshape(nx,ny)
    #help(VoxelCoefficient)
    func = VoxelCoefficient((-1,-1), (0,1), values, linear=True)
    poly_t = [-1 + 2*np.random.rand() for porder in range(poly_t_order)]
    #poly_t = [np.random.rand() for porder in range(poly_t_order)]
    #tp = np.linspace(0,tend,100)
    #p_vals = poly_t[0] + tp*(poly_t[1] + tp*(poly_t[2] + tp*poly_t[3]  )) 
    #import matplotlib.pyplot as plt
    #plt.plot(tp,p_vals)
    #plt.show()

def get_order(order_global):
    
    k = order_global
    kstar = 1
    q = order_global
    if order_global == 1:
        qstar = 1
    else:
        qstar = 0

    return q,k,qstar,kstar


def SolveProblem(ref_lvl,order_global, alpha_idx = 0, plotting=True,plot_geom=False):

    alpha_scale = alphas[alpha_idx] 
    alpha_str = alphas_strs[alpha_idx] 

    rho  = rho0 + 1*(rho1-rho0)/10
    tend = sqrt((rho1-rho)/(1-epsilon))
    rho *= alpha_scale 
    delta = rho/2

    # Space finite element order
    meshes_list = [ ]
    for i in range(max_ref_lvl):
        mesh = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh = maxh_init,order=order_global)
        for n in range(i):
            mesh.Refine()
        mesh.Curve(order_global)
        meshes_list.append(mesh)

    q,k,qstar,kstar = get_order(order_global)
    time_order = 2*max(q,qstar)

    N = Ns[ref_lvl]
    delta_t = 2*tend / N
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
    #levelset = data_lset
    lset_Omega =  x**2 +  y**2  - R_Omega**2  

    # define exact solution
    qpi = pi/4 
    t_slice = [ -tend + n*delta_t + delta_t*tref for n in range(N)]
    u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    
    if perturb:
        noise_slice = [  (poly_t[0] + t_slice[n]*(poly_t[1] + t_slice[n]*(poly_t[2] + t_slice[n]*poly_t[3])))  for n in range(N)]
        perturbations["noise-slice"] = noise_slice

    st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
                    t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, told=told,
                    perturbations=perturbations)

    st.SetupSpaceTimeFEs()
    st.SetupRightHandSide()
    st.PreparePrecondGMRes()
    A_linop = SpaceTimeMat(ndof = st.X.ndof,mult = st.ApplySpaceTimeMat)
    PreTM = SpaceTimeMat(ndof = st.X.ndof,mult = st.TimeMarching)

    st.gfuX.vec.data = GMRes(A_linop, st.f.vec, pre=PreTM, maxsteps = 10000, tol = tol,
                  callback=None, restart=None, startiteration=0, printrates=True)


    l2_error_B = st.MeasureErrors(st.gfuX,levelset,domain_type=NEG)
    print("st.MeasureErrors in B = ", l2_error_B ) 
    l2_errors_B_complement = st.MeasureErrors(st.gfuX,levelset,domain_type=POS)
    print("Errors in B complement = ", l2_errors_B_complement )
    l2_errors_omega = st.MeasureErrors(st.gfuX, data_lset, data_domain=True)
    print("Errors in omega = ", l2_errors_omega )
    l2_errors_Q = st.MeasureErrors(st.gfuX, data_lset, Q_all=True )
    print("Errors in Q (all) = ", l2_errors_Q) 

    if plot_geom: 
         
        fesp1 = H1(mesh, order=1, dgjumps=True)
        # Time finite element (nodal!)
        tfe = ScalarTimeFE(st.q)
        # Space-time finite element space
        st_fes = tfe * fesp1
        
        lset_p1 = GridFunction(st_fes)

        lset_adap_st = LevelSetMeshAdaptation_Spacetime(mesh, order_space=st.k,
                                                        order_time=st.q,
                                                        threshold=0.1,
                                                        discontinuous_qn=True)
        ci = CutInfo(mesh, time_order=0)

        vtk_out = [B_lset , lset_Omega, data_lset, Q_lset ]
        vtk_out_names = ["B", "Omega", "data", "Q"]
        vtk = SpaceTimeVTKOutput(ma=mesh, coefs=vtk_out, names=vtk_out_names,
                         filename="spacetime_vtk_alpha_{0}".format(alpha_str), subdivision_x=1,
                         subdivision_t=3)
        n = 0 
        told.Set(st.tstart)
        while tend - told.Get() > st.delta_t / 2:
            SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
            dfm = lset_adap_st.CalcDeformation(levelset, tref)
            ci.Update(lset_p1, time_order=0)

            vtk.Do(t_start=told.Get(), t_end=told.Get() + delta_t)
            told.Set(told.Get() + delta_t)
            n += 1 


    if plotting:
        # Plotting
        fesp1 = H1(mesh, order=1, dgjumps=True)
        # Time finite element (nodal!)
        tfe = ScalarTimeFE(st.q)
        # Space-time finite element space
        st_fes = tfe * fesp1
        lset_p1 = GridFunction(st_fes)
        lset_adap_st = LevelSetMeshAdaptation_Spacetime(mesh, order_space=st.k,
                                                        order_time=st.q,
                                                        threshold=0.1,
                                                        discontinuous_qn=True)
        ci = CutInfo(mesh, time_order=0)

        uh_slab = GridFunction(st.W_slice_primal)
        u_slab = GridFunction(st.W_slice_primal)
        diff_slab = GridFunction(st.W_slice_primal)
        u_slab_node = GridFunction(st.V_space)

        vtk_out = [B_lset , lset_Omega, data_lset, Q_lset,u_slab, uh_slab, lset_p1, diff_slab ]
        vtk_out_names = ["B", "Omega", "data", "Q","u", "uh","lsetp1","diff"]

        vtk = SpaceTimeVTKOutput(ma=mesh, coefs=vtk_out, names=vtk_out_names,
                                 filename="2D-cylinder-reflvl{0}-q{1}".format(ref_lvl,q), subdivision_x=3,
                                 subdivision_t=3)
        print("ploting ...")
        n = 0 
        told.Set(st.tstart)
        while tend - told.Get() > st.delta_t / 2:
            #print("n = ", n) 
            SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
            dfm = lset_adap_st.CalcDeformation(levelset, tref)
            ci.Update(lset_p1, time_order=0)
            uh_slab.vec.FV().NumPy()[:] = st.gfuX.components[0].components[n].vec.FV().NumPy()[:]
            
            times = [xi for xi in st.W_slice_primal.TimeFE_nodes()]
            for i,ti in enumerate(times):
                u_slab_node.Set(fix_tref(st.u_exact_slice[n],ti ))
                u_slab.vec[i*st.V_space.ndof : (i+1)*st.V_space.ndof].data = u_slab_node.vec[:]
            diff_slab.vec.FV().NumPy()[:] = u_slab.vec.FV().NumPy()[:] - uh_slab.vec.FV().NumPy()[:]

            vtk.Do(t_start=told.Get(), t_end=told.Get() + st.delta_t)
            told.Set(told.Get() + st.delta_t)
            n += 1 

    return delta_t, l2_error_B, l2_errors_B_complement, l2_errors_omega, l2_errors_Q  
 
if False:

    for order_global in [1,2,3]:
        
        errors = { "delta-t": [], 
                   "B" : [ ],
                  "B-complement": [],
                  "omega" : [ ],  
                  "Q_all" : [ ] 
                  } 

        q,k,qstar,kstar = get_order(order_global)
        for ref_lvl in range(max_ref_lvl + 1 - order_global):
            plotting = False
            plot_geom = False
            if order_global == 2 and ref_lvl == 1:
                plotting = True
                plot_geom = False
                
            result = SolveProblem(ref_lvl,order_global = order_global, alpha_idx = 0, plotting=plotting,plot_geom=plot_geom)
            errors["delta-t"].append(result[0])
            errors["B"].append(result[1])
            errors["B-complement"].append(result[2])
            errors["omega"].append(result[3])
            errors["Q_all"].append(result[4])
            #errors["omega"].append(result)
        
        name_str = "Cylinder" + "-"  + "-q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}-msol2".format(kstar)+".dat"
        results = [np.array(errors["delta-t"],dtype=float), np.array( errors["B"],dtype=float), np.array( errors["B-complement"] ,dtype=float), 
                   np.array( errors["omega"],dtype=float), np.array( errors["Q_all"],dtype=float)  ]
        header_str = "deltat L2-err-B L2-err-Bcompl L2-err-omega Qall"
        np.savetxt(fname ="../data/{0}".format(name_str),
                   X = np.transpose(results),
                   header = header_str,
                   comments = '')


order_global = 1
q,k,qstar,kstar = get_order(order_global)
for alpha_idx in range(4):
    
    errors = { "delta-t": [], 
               "B" : [ ],
               "B-complement": [],
               "omega" : [ ],  
               "Q_all" : [ ] 
             } 

    alpha_str = alphas_strs[alpha_idx] 

    for ref_lvl in range(max_ref_lvl + 1 - order_global):
        plotting = False
        plot_geom = False
        if ref_lvl == 1:
            plotting = False
            plot_geom = True
            
        result = SolveProblem(ref_lvl,order_global = order_global, alpha_idx = alpha_idx, plotting=plotting,plot_geom=plot_geom)
        errors["delta-t"].append(result[0])
        errors["B"].append(result[1])
        errors["B-complement"].append(result[2])
        errors["omega"].append(result[3])
        errors["Q_all"].append(result[4])
    
    name_str = "Cylinder" + "-"  + "-q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}-msol2".format(kstar)+"alpha-"+alpha_str+".dat"
    results = [np.array(errors["delta-t"],dtype=float), np.array( errors["B"],dtype=float), np.array( errors["B-complement"] ,dtype=float), 
               np.array( errors["omega"],dtype=float), np.array( errors["Q_all"],dtype=float)  ]
    header_str = "deltat L2-err-B L2-err-Bcompl L2-err-omega Qall"
    np.savetxt(fname ="../data/{0}".format(name_str),
               X = np.transpose(results),
               header = header_str,
               comments = '')


