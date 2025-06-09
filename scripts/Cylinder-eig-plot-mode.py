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

tol = 1e-7 #GMRes
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


sigma = 1e-4 
mode_max_ref_lvl = 4 
max_ref_lvl = 4

# Space finite element order
order_mode = 1

order_global = 1
k = order_global
kstar = order_global
q = order_global
qstar = order_global
time_order = 2*max(q,qstar)

# case of perturbed data 
perturbations = { }
perturb = True
if perturb:
    poly_t_order = 4
    perturbations["exponent"] = 1
    #perturbations["scal"] = 1e-14
    perturbations["scal"] = 1e0
    perturbations["maxh"] = maxh_init 

stabs = {"data": 1e4,
         "dual": 1,
         "primal": 1e-4,
         "primal-jump":1e1,
         "primal-jump-displ-grad":1e1,
         "Tikh": 1e-5
        }

Ns = [4,8,16,32,64,128]
noise_lvl = [ 1    for nn in Ns ] 
m_sol = 2

if perturb:
    nx = 50
    ny = 50
    values = np.array([-1 + 2*np.random.rand() for y in np.linspace(0,2*np.pi,ny) for x in np.linspace(0,2*np.pi, nx)])
    values = values.reshape(nx,ny)
    func = VoxelCoefficient((-1,-1), (0,1), values, linear=True)
    poly_t = [-1 + 2*np.random.rand() for porder in range(poly_t_order)]

meshes_list = [ ]
for i in range(max(max_ref_lvl,mode_max_ref_lvl)+1):
    mesh = GetMeshHalfCylinder(R_Omega=R_Omega,R_data=R_data,maxh = maxh_init,order=order_global)
    for n in range(i):
        mesh.Refine()
    mesh.Curve(order_global)
    meshes_list.append(mesh)

def SolveProblem(ref_lvl,plotting=True,plot_geom=False,create_video=False):
     
    N = Ns[ref_lvl]
    delta_t = 2*tend / N
    mesh = meshes_list[ref_lvl]

    # Level-set functions specifying the geoemtry
    told = Parameter(-tend)
    t = told + delta_t * tref

    def get_levelset(s):
        return s - (x-beta)**2 - y**2 + (1-epsilon)*t**2 

    def get_levelset_fix_t(s,tfix=0.0):
        return s - (x-beta)**2 - y**2 + (1-epsilon)*tfix**2 
    
    B_lset = get_levelset(rho)
    Q_lset = get_levelset(delta)
    data_lset = x**2 +  y**2  - R_data**2 
    levelset = B_lset
    #levelset = data_lset
    lset_Omega =  x**2 +  y**2  - R_Omega**2 
 
    t_slice = [ -tend + n*delta_t + delta_t*tref for n in range(N)]
    qpi = pi/4
    u_exact_slice = [  5*cos( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol  * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]
    #print(" cos( - sqrt(2) * m_sol * qpi * t_end  ) = ", cos( - sqrt(2) * m_sol * qpi * tend  ) ) 
    ut_exact_slice = [ -5 * sqrt(2) * m_sol * qpi * sin( sqrt(2) * m_sol * qpi * t_slice[n] ) * cos(m_sol * qpi * x) * cos(m_sol  * qpi * y) for n in range(N)]

    if load_modes: 
        q = order_global 
        qstar = order_global 
        k = order_global
        kstar = order_global
        #s_mode =  noise_lvl[ref_lvl] * np.loadtxt('singular/lvl{0}/mode{1}-ref_lvl{0}.out'.format(ref_lvl,0) ,delimiter=',')[:]
        
        s_mode = np.loadtxt('mode{1}-ref_lvl{0}.out'.format(mode_max_ref_lvl ,0) ,delimiter=',')[:]
            
        N_f = Ns[mode_max_ref_lvl  ]
        V_f = GetSpaceTimeFESpace(meshes_list[mode_max_ref_lvl  ],order_mode , order_mode ,bc=[])
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
                #print("n_c =" , n_c)
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
        

        mass_omega_ls = [ ]
        mass_B_ls = [ ]
        mass_complement_ls = [] 

        ttimes = [ ]


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

            noise_slab = GridFunction(st.W_slice_primal)

            vtk_out = [noise_slab ]
            vtk_out_names = ["noise"]

            vtk = SpaceTimeVTKOutput(ma=mesh, coefs=vtk_out, names=vtk_out_names,
                                     filename="2D-cylinder-noise-reflvl{0}-q{1}".format(ref_lvl,q), subdivision_x=3,
                                     subdivision_t=3)
            print("ploting ...")
            n = 0 
            told.Set(st.tstart)
            while tend - told.Get() > st.delta_t / 2:
                #print("n = ", n) 
                SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
                dfm = lset_adap_st.CalcDeformation(levelset, tref)
                ci.Update(lset_p1, time_order=0)
                
                noise_slab.vec.FV().NumPy()[:] = noise_slice[n].vec.FV().NumPy()[:]                

                vtk.Do(t_start=told.Get(), t_end=told.Get() + st.delta_t)
                told.Set(told.Get() + st.delta_t)
                n += 1


        if create_video:
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
            noise_slab = GridFunction(st.W_slice_primal)
            u_slab_node = GridFunction(st.V_space)
            noise_slab_node = GridFunction(st.V_space)
            #noise_slab_node = GridFunction(st.V_space)

            #vtk_out = [B_lset , lset_Omega, data_lset, Q_lset,u_slab, uh_slab, lset_p1, diff_slab, noise_slab  ]
            #vtk_out_names = ["B", "Omega", "data", "Q","u", "uh","lsetp1","diff" ,"noise"]
            
            vtk_out = [  noise_slab_node  ]
            vtk_out_names = [ "noise"]

            vtk =  VTKOutput(ma=mesh, coefs=vtk_out, names=vtk_out_names,
                                     filename="2D-cylinder-noise-time".format(ref_lvl,q), subdivision=2)

            bonus_intorder_error = 5 
            #dx_ho = dmesh(st.mesh, bonus_intorder = bonus_intorder_error) 
            #dx_omega = dmesh(st.mesh, bonus_intorder =  bonus_intorder_error , definedon=st.mesh.Materials("omega")) 
            print("ploting ...")
            n = 0 
            told.Set(st.tstart)
            while tend - told.Get() > st.delta_t / 2:
                #print("n = ", n) 
                SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
                dfm = lset_adap_st.CalcDeformation(levelset, tref)
                ci.Update(lset_p1, time_order=0)
                #uh_slab.vec.FV().NumPy()[:] = st.gfuX.components[0].components[n].vec.FV().NumPy()[:]
                noise_slab.vec.FV().NumPy()[:] = noise_slice[n].vec.FV().NumPy()[:]
                

                #times = [xi for xi in st.W_slice_primal.TimeFE_nodes()]
                times = [0.0,0.25,0.5,0.75,1.0]  
                for i,tau_i in enumerate(times):
                    lset_B_tfix = get_levelset_fix_t(rho,tfix=told.Get() + tau_i*st.delta_t)
                    chi = IfPos(-lset_B_tfix, 1.0, 0.0 )

                    
                    #u_slab_node.Set(fix_tref(st.u_exact_slice[n],ti ))
                    #u_slab.vec[i*st.V_space.ndof : (i+1)*st.V_space.ndof].data = u_slab_node.vec[:]
                    #if i == 0: 
                    #noise_slab_node.vec[:].data  = noise_slab.vec[i*st.V_space.ndof : (i+1)*st.V_space.ndof]  
                    noise_slab_node.vec[:].data  = (1-tau_i)*noise_slab.vec[ : st.V_space.ndof] + tau_i * noise_slab.vec[st.V_space.ndof : 2*st.V_space.ndof] 
                    
                    #mass_omega =  sqrt( Integrate(   noise_slab_node**2, st.mesh, definedon=st.mesh.Materials("omega"), bonus_intorder = bonus_intorder_error)  ) 
                    mass_omega =  sqrt( Integrate(   noise_slab_node**2, st.mesh, definedon=st.mesh.Materials("omega"), order=2*st.k+bonus_intorder_error  )  )
                    mass_B = sqrt( Integrate(  chi * noise_slab_node**2, st.mesh, order=2*st.k+bonus_intorder_error  )  )
                    mass_all = sqrt( Integrate(  noise_slab_node**2, st.mesh, order=2*st.k+bonus_intorder_error  )  )
                    mass_complement =  mass_all - mass_B 

                    mass_omega_ls.append( mass_omega ) 
                    mass_B_ls.append( mass_B )
                    mass_complement_ls.append( mass_complement ) 

                    ttimes.append( told.Get() + tau_i*st.delta_t ) 
                    #vtk.Do( time =  told.Get() + i*st.delta_t  )
                    vtk.Do( time =  told.Get() + tau_i*st.delta_t  )

                #diff_slab.vec.FV().NumPy()[:] = np.abs( u_slab.vec.FV().NumPy()[:] - uh_slab.vec.FV().NumPy()[:] )

                #vtk.Do(t_start=told.Get(), t_end=told.Get() + st.delta_t)
                
                told.Set(told.Get() + st.delta_t)
                n += 1 

        #input("")
        #return delta_t, l2_error_B, l2_errors_B_complement, l2_errors_omega, l2_errors_Q  

        return ttimes, mass_omega_ls,mass_B_ls,mass_complement_ls 

    #return l2_errors_omega 


errors = { "delta-t": [], 
           "B" : [ ],
          "B-complement": [],
          "omega" : [ ],  
          "Q_all" : [ ] 
          } 

masses = { "times": [],
           "omega": [],
           "B": [],
          "complement": [] }


if load_modes: 
    it_lvl = max_ref_lvl + 1
else:
    it_lvl = mode_max_ref_lvl + 1 

for ref_lvl in range(it_lvl):
    plotting = False
    plot_geom = False
    create_video = False

    if ref_lvl == 2:
        plotting = True
    if ref_lvl == 4:
        create_video = True

    if True:     
        #plotting = load_modes 
        plot_geom = False
        
    result = SolveProblem(ref_lvl,plotting = plotting,plot_geom = plot_geom, create_video = create_video)
    masses["times"] = result[0]
    masses["omega"] = result[1] 
    masses["B"] = result[2] 
    masses["complement"] = result[3] 

print(masses)

results = [np.array( masses["times"] ,dtype=float), np.array(masses["omega"],dtype=float), np.array(masses["B"],dtype=float) , np.array(masses["complement"],dtype=float)]
header_str = "t mass-omega mass-B mass-complement"
name_str = "Cylinder-bad-mode-mass.dat" 
np.savetxt(fname ="../data/{0}".format(name_str),
               X = np.transpose(results),
               header = header_str,
               comments = '')

import matplotlib.pyplot as plt 
plt.semilogy( masses["times"],  masses["omega"], label="omega") 
plt.semilogy( masses["times"],  masses["B"], label="B") 
plt.semilogy( masses["times"],  masses["complement"], label="complement") 
plt.legend()
plt.show()



