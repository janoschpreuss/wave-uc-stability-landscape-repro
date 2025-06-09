import sys 
from ngsolve import *
from xfem import *
from xfem.lset_spacetime import *
from math import pi, log
from ngsolve.solvers import GMRes
from space_time import space_time, SpaceTimeMat, GetSpaceTimeFESpace
from meshes import GetMeshDataAllAround, GetMeshHalfCylinder
import numpy as np
import scipy.sparse.linalg
np.random.seed(0) #fix random seed

load_modes = True


#tol = 1e-7 #GMRes
tol = 2e-6 #GMRes
maxh_init = 2

# parameters for specifying geoemetry
R_Omega = 1.0 
R_data = 0.75
epsilon = 0.05
beta = 0.5

rr = R_data
rho0 = rr**2 + beta**2
rho1 = (rr + beta)**2
print("rho0 = {0}, rho1 = {1}".format( rho0,rho1 ) )
#input("")
#rho  = rho0 + (rho1-rho0)/2
rho  = rho0 + 1*(rho1-rho0)/10
delta = rho/2
tend = sqrt((rho1-rho)/(1-epsilon))
R = 1.0
alpha_idx = 0 
alphas = [1.0,0.75,0.5,0.25] 
alphas_strs = ["one","3quarter","half","quarter"]
alpha_scale = alphas[alpha_idx] 
alpha_str = alphas_strs[alpha_idx] 
#print("rho = ", rho)
rho *= alpha_scale 
#print("rho = ", rho)
#print("tend = ", tend)

sigma = 1e-4 
mode_max_ref_lvl = 4 
max_ref_lvl = 4

# Space finite element order
order_mode = 1

order_global = 1
k = order_global
#kstar = 1
kstar = order_global
q = order_global
qstar = order_global
#qstar = 1
time_order = 2*max(q,qstar)


stabs = {"data": 1e4,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1,
         "Tikh": 1e-5
        }


Ns = [n for n in range(11,65)]
#Ns = [n for n in range(11,25)]
noise_lvl = [ 1    for nn in Ns ] 

m_sol = 2


# Reference discretization for the mode
mesh_mode = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh = maxh_init,order=order_global)
for i in range(mode_max_ref_lvl):
    mesh_mode.Refine()
mesh_mode.Curve(order_global)
N_f = 64

meshes_list = [ ]
for n in Ns:
    mesh = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh = 2/n,order=order_global)
    mesh.Curve(order_global)
    meshes_list.append(mesh)


def SolveProblem(ref_lvl, load_modes, theta, plotting=True,plot_geom=False):

    # case of perturbed data 
    perturbations = { }
    perturb = True
    if perturb:
        poly_t_order = 4
        perturbations["exponent"] = theta
        perturbations["scal"] = 1e0
        perturbations["maxh"] = maxh_init 

    if not load_modes: 
        N = N_f
    else:
        N = Ns[ref_lvl]
    #maxh = maxhs[ref_lvl]
    delta_t = 2*tend / N
    if not load_modes:
        mesh = mesh_mode 
    else:
        mesh = meshes_list[ref_lvl]
    #mesh = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh=maxh)
    #Draw(mesh)
    #input("")

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
 
    t_slice = [ -tend + n*delta_t + delta_t*tref for n in range(N)]
    qpi = pi/4
    
    if load_modes: 
        u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
        print(" cos( - sqrt(2) * m_sol * qpi * t_end  ) = ", cos( - sqrt(2) * m_sol * qpi * tend  ) ) 
        ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    else:
        u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * x)  * cos(m_sol  * qpi * y) for n in range(N)]
        ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * x)  * cos(m_sol  * qpi * y) for n in range(N)]


    # define exact solution
    if not load_modes: 
             
        q = order_mode 
        qstar = order_mode 
        k = order_mode 
        kstar = order_mode

        st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
                        t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, told=told,
                        perturbations=None)

        st.SetupSpaceTimeFEs()
        st.SetupRightHandSide()
        st.PreparePrecondGMRes()   

        A_linop = SpaceTimeMat(ndof = st.X.ndof,mult = st.ApplySpaceTimeMat)

        st_shifted = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
                        t_slice=t_slice, u_exact_slice=u_exact_slice, ut_exact_slice=ut_exact_slice, told=told,
                        perturbations=None,shift = sigma)

        st_shifted.SetupSpaceTimeFEs()
        st_shifted.SetupRightHandSide()
        st_shifted.PreparePrecondGMRes()
        
        st.SetupSlabMassMatrix(scale = 1.0 )

        A_shifted_linop = SpaceTimeMat(ndof = st.X.ndof,mult = st_shifted.ApplySpaceTimeMat)
        PreTM = SpaceTimeMat(ndof = st.X.ndof,mult = st_shifted.TimeMarching)
        MassM = SpaceTimeMat(ndof = st.X.ndof,mult = st.ApplySpaceTimeMassMat )

        #st.gfuX.vec.data = GMRes(A_linop, st.f.vec, pre=PreTM, maxsteps = 10000, tol = tol,
        #              callback=None, restart=None, startiteration=0, printrates=True)

        tmp1 = st.f.vec.CreateVector()
        tmp2 = st.f.vec.CreateVector()
        #tmp3 = st.f.vec.CreateVector()
       
        ttmp1 = st_shifted.f.vec.CreateVector()
        ttmp2 = st_shifted.f.vec.CreateVector()


        def apply_atilde(v):
            if len(v.shape) > 1:
                tmp1.FV().NumPy()[:] = v[:,0]
            else:
                tmp1.FV().NumPy()[:] = v
            tmp2.data = A_linop * tmp1
            #tmp3.data = invmX * tmp2
            #tmp3.data -= proj_freedofs * tmp3
            #return tmp3.FV().NumPy()
            return tmp2.FV().NumPy()

        def apply_opinv(v):
            if len(v.shape) > 1:
                ttmp1.FV().NumPy()[:] = v[:,0]
            else:
                ttmp1.FV().NumPy()[:] = v
            
            ttmp2.data = GMRes(A_shifted_linop, ttmp1, pre=PreTM, maxsteps = 10000, tol = tol,
                      callback=None, restart=None, startiteration=0, printrates=True)

            return ttmp2.FV().NumPy()

        
        def apply_mtilde(v):
            if len(v.shape) > 1:
                tmp1.FV().NumPy()[:] = v[:,0]
            else:
                tmp1.FV().NumPy()[:] = v
            tmp2.data = MassM * tmp1
            #tmp3.data = invmX * tmp2
            #tmp3.data -= proj_freedofs * tmp3
            #return tmp3.FV().NumPy()
            return tmp2.FV().NumPy()

        atilde_linop = scipy.sparse.linalg.LinearOperator( (A_linop.Height(),A_linop.Width()),dtype = float, matvec=apply_atilde)
        mtilde_linop = scipy.sparse.linalg.LinearOperator( (MassM.height,MassM.width),dtype = float, matvec=apply_mtilde)
        opinv_linop = scipy.sparse.linalg.LinearOperator( (PreTM.Height(),PreTM.Width()),dtype = float, matvec=apply_opinv)
        
        kmodes = 1 

        if False:
            eigval,eigvec =  scipy.sparse.linalg.eigsh(atilde_linop, M = mtilde_linop, sigma=sigma,   k=kmodes  , which='LM', OPinv= opinv_linop )
            print("eigval =" , eigval ) 
            print("len( eigvec[:,0] ) = ", len( eigvec[:,0] ) )
            st.gfuX.vec.FV().NumPy()[:] = eigvec[:,0]

            n = 0
            l2_norm = 0.0
            for n in range(st.N):
                l2_norm +=  Integrate( (  st.gfuX.components[0].components[n])**2 * st.dxt , st.mesh )
                n += 1 
            l2_norm = sqrt( l2_norm )
            print("l2_norm = ", l2_norm)  
            for n in range(st.N): 
                st.gfuX.components[0].components[n].vec.data *= 1 / l2_norm 

        if True:
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
            u_slab_node = GridFunction(st.V_space)
            n = 0 
            told.Set(st.tstart)
            print("st.N = ", st.N)
            print(" len(st.gfuX.components[0].components)  = ",  len(st.gfuX.components[0].components) )
            while tend - told.Get() > st.delta_t / 2:
                #print("n = ", n) 
                #SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
                #dfm = lset_adap_st.CalcDeformation(levelset, tref)
                #ci.Update(lset_p1, time_order=0)
                
                times = [xi for xi in st.W_slice_primal.TimeFE_nodes()]
                print("n = ", n)
                for i,ti in enumerate(times):
                    u_slab_node.Set(fix_tref(u_exact_slice[n],ti ))
                    print(" (i+1)*st.V_space.ndof = ", (i+1)*st.V_space.ndof)
                    print("len(st.gfuX.components[0].components[n].vec) = ", len(st.gfuX.components[0].components[n].vec)) 
                    st.gfuX.components[0].components[n].vec[i*st.V_space.ndof : (i+1)*st.V_space.ndof].data = u_slab_node.vec[:]
                told.Set(told.Get() + st.delta_t)
                n += 1 

        #print("len(  st.gfuX.components[0].vec.FV().NumPy()[:] ) =  ", len(  st.gfuX.components[0].vec.FV().NumPy()[:] ) )
        
        #np.savetxt('singular/lvl{0}/mode{1}-ref_lvl{0}.out'.format(ref_lvl,0), eigvec[:,0], delimiter=',')
        #np.savetxt('singular/lvl{0}/mode{1}-ref_lvl{0}.out'.format(ref_lvl,0),  st.gfuX.components[0].vec.FV().NumPy()[:] , delimiter=',')
        np.savetxt('smooth{1}-ref_lvl{0}.out'.format(ref_lvl,0),  st.gfuX.components[0].vec.FV().NumPy()[:] , delimiter=',')
        #np.savetxt('p2/lvl{0}/mode{1}-ref_lvl{0}.out'.format(ref_lvl,0),  st.gfuX.components[0].vec.FV().NumPy()[:] , delimiter=',')


    if load_modes: 
        q = order_global 
        qstar = order_global 
        k = order_global
        kstar = order_global
        #s_mode =  noise_lvl[ref_lvl] * np.loadtxt('singular/lvl{0}/mode{1}-ref_lvl{0}.out'.format(ref_lvl,0) ,delimiter=',')[:]
        
        #s_mode = np.loadtxt('singular/lvl{0}/mode{1}-ref_lvl{0}.out'.format(mode_max_ref_lvl ,0) ,delimiter=',')[:]
        s_mode = np.loadtxt('smooth{1}-ref_lvl{0}.out'.format(mode_max_ref_lvl ,0) ,delimiter=',')[:]
        #s_mode = np.loadtxt('p2/lvl{0}/mode{1}-ref_lvl{0}.out'.format(mode_max_ref_lvl  ,0) ,delimiter=',')[:]
        
        #s_mode =  np.loadtxt('singular/lvl{0}/mode{1}-ref_lvl{0}.out'.format(ref_lvl,0) ,delimiter=',')[:]

        #print("len(s_mode) = " , len(s_mode) )
            
        #N_f = Ns[mode_max_ref_lvl  ]
        #N_f = Ns[ref_lvl]

        #print("N_f = ", N_f)
        #V_f = GetSpaceTimeFESpace(meshes_list[ref_lvl],k,q,bc=[])
        V_f = GetSpaceTimeFESpace(mesh_mode , order_mode , order_mode ,bc=[])
        X_f = FESpace([V_f  for n in range(N_f)])
        #print(" X_f.ndof = ",  X_f.ndof)
        data_f = GridFunction(X_f)
        #print("len( data_f.vec.FV().NumPy()) = ", len( data_f.vec.FV().NumPy()) ) 
        data_f.vec.FV().NumPy()[:] = s_mode[:]
        delta_t_f = 2*tend / N_f

        st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=tend,delta_t=delta_t,mesh=mesh,stabs=stabs,
                        t_slice=t_slice, u_exact_slice = u_exact_slice , ut_exact_slice =  None, told=told,
                        perturbations=None)
        st.SetupSpaceTimeFEs() 
        delta_t_c = st.delta_t  
        gfu_space_c = GridFunction(st.V_space)
        ndof_node = len(gfu_space_c.vec)
        data = GridFunction(st.X)
        
        #if st.N != N_f:
        if True:
            uu = st.V_space.TrialFunction()
            vv = st.V_space.TestFunction()
            mm = BilinearForm(st.V_space, symmetric=True)
            mm += uu * vv *  dx
            #mm += 0.01 * specialcf.mesh_size * InnerProduct( ( grad(uu) - grad(uu).Other()) * specialcf.normal(mesh.dim), ( grad(vv) - grad(vv).Other())  * specialcf.normal(mesh.dim) ) * dx(skeleton=True) 
            mm.Assemble() 
            mm_inv = mm.mat.Inverse(st.V_space.FreeDofs(), inverse="sparsecholesky") 

            
            #def ComputeL2Proj(coeff):
            #    pass

            for n_c in range(st.N):
                print("n_c =" , n_c)
                tn_c = st.tstart + n_c *  delta_t_c  
                #for i,ti in enumerate(st.space.TimeFE_nodes()):
                j = 0
                for i,ti in enumerate(st.X.components[0].components[n_c].TimeFE_nodes()):
                    #print("i = ", i)
                    t_phys = tn_c + delta_t_c * ti
                    # Find corresponding slab of the fine discretization that contains t_phys 
                    n_slab_f = 0 
                    t_n_f = st.tstart
                    t_nn_f = st.tstart + delta_t_f 
                    slab_found = False
                    while not slab_found and n_slab_f <= N_f: 
                        if t_phys >= t_n_f and t_phys <= t_nn_f + 1e-12:
                            slab_found = True 
                        if not slab_found:
                            n_slab_f += 1 
                            t_n_f  +=  delta_t_f 
                            t_nn_f +=  delta_t_f
                    t_n_f_local = (t_phys - t_n_f) / delta_t_f
                    if t_n_f_local < 0 or t_n_f_local > 1.0: 
                        #print("Warning t_n_f_local = ",t_n_f_local)
                        #print("Applying correction:")
                        if t_n_f_local < 0:
                            t_n_f_local = 0
                        if t_n_f_local > 1.0:
                            t_n_f_local = 1.0

                    if st.X.components[0].components[n_c].IsTimeNodeActive(i):
                        #ngsolveSet(gfu_space_c,fix_tref(cf,ti), *args, **kwargs)
                        #print("n_slab_f = ", n_slab_f)
                        gfu_space_c.Set(fix_tref(data_f.components[n_slab_f], t_n_f_local))
                        #ff = LinearForm(st.V_space)
                        #ff +=  fix_tref(data_f.components[n_slab_f], t_n_f_local) * vv * dx
                        #ff.Assemble() 
                        #gfu_space_c.vec.data = mm_inv * ff.vec
                        data.components[0].components[n_c].vec[j*ndof_node : (j+1)*ndof_node].data = gfu_space_c.vec[:]
                        #print("data.components[0].components[n_c].vec[j*ndof_node : (j+1)*ndof_node] = ", data.components[0].components[n_c].vec[j*ndof_node : (j+1)*ndof_node])
                        #self.vec[j*ndof_node : (j+1)*ndof_node].data = gfu_space_c.vec[:]
                        j += 1

            #if False:
            #st.u_exact_slice = data.components[0].components
        else:
            st.u_exact_slice = data_f.components
            #st.u_exact_slice = s_mode[:]  
         
        noise_slice = data.components[0].components
        #help(noise_slice)
        # normalize noise 
        n = 0
        l2_norm_noise = [ ] 
        st.told.Set(st.tstart)
        while st.tend - st.told.Get() > st.delta_t / 2:
            l2_n = Integrate( noise_slice[n]**2 * st.dxt_omega, st.mesh)
            l2_norm_noise.append(l2_n)
            st.told.Set(st.told.Get() + st.delta_t)
            n += 1 
        n = 0
        for n in range(len(noise_slice)):
            noise_slice[n].vec[:] *= 1/sqrt(sum(l2_norm_noise))
        

        perturbations["noise-slice"] = noise_slice
        st.perturbations = perturbations

        n = 0
        l2_error_data = [ ] 
        st.told.Set(st.tstart)
        while st.tend - st.told.Get() > st.delta_t / 2:
            err_n = Integrate( (u_exact_slice[n]  - st.u_exact_slice[n]  )**2 * st.dxt, st.mesh)
            #err_n = Integrate( (u_exact_slice[n]  - st.u_exact_slice[n]  )**2 * st.dxt_omega, st.mesh)
            #print("Interpolation error on slab n = {0} is: {1}".format(n, err_n ))
            l2_error_data.append(err_n)
            st.told.Set(st.told.Get() + st.delta_t)
            n += 1 
        n = 0 
        #print("Data error = ",  sqrt(sum(l2_error_data)))
        #input("")


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
        #l2_errors_omega = st.MeasureErrors(st.gfuX, data_lset, data_domain=True)
        print("Errors in omega = ", l2_errors_omega )
        l2_errors_Q = st.MeasureErrors(st.gfuX, data_lset, Q_all=True )
        print("Errors in Q (all) = ", l2_errors_Q) 


        return delta_t, l2_error_B, l2_errors_B_complement, l2_errors_omega, l2_errors_Q  
 



load_modes = False
it_lvl = mode_max_ref_lvl + 1 
ran = [mode_max_ref_lvl] 
for ref_lvl in ran:
    plotting = False
    plot_geom = False
    result = SolveProblem(ref_lvl,load_modes=load_modes,theta=99,plotting=plotting,plot_geom=plot_geom)

load_modes = True
it_lvl = len(Ns) 
ran = range(it_lvl)


for theta in [1,2]:

    errors = { "delta-t": [], 
               "B" : [ ],
              "B-complement": [],
              "omega" : [ ],  
              "Q_all" : [ ] 
              } 

    for ref_lvl in ran:
        plotting = False 
        plot_geom = False

        result = SolveProblem(ref_lvl,load_modes=load_modes, theta=theta, plotting=plotting,plot_geom=plot_geom)
        
        errors["delta-t"].append(result[0])
        errors["B"].append(result[1])
        errors["B-complement"].append(result[2])
        errors["omega"].append(result[3])
        errors["Q_all"].append(result[4])
        print(errors)
        
        name_str = "Cylinder" + "-"  + "q{0}".format(q)+"-qstar{0}".format(qstar)+"-k{0}".format(k)+"-kstar{0}".format(kstar)+"-noise-smooth-theta{0}".format(theta)+".dat"
        results = [np.array(errors["delta-t"],dtype=float), np.array( errors["B"],dtype=float), np.array( errors["B-complement"] ,dtype=float),
                   np.array( errors["omega"],dtype=float)  ]
        header_str = "deltat L2-err-B L2-err-Bcompl L2-err-omega"
        np.savetxt(fname ="../data/{0}".format(name_str), X = np.transpose(results),header = header_str,comments = '')

