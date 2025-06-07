from ngsolve import *
from xfem import *
from xfem.lset_spacetime import *
from ngsolve.comp import GlobalSpace
solver = "pardiso"
#solver = "umfpack"
from space_time import LaplacianProxy, quad_rule

def GetSpaceTimeFESpace(mesh,order_space,order_time,bc=".*",return_base=False):
    V_space = H1(mesh, order=order_space, dirichlet=bc,dgjumps=True)
    if return_base:
        return V_space
    tfe = ScalarTimeFE(order_time)
    st_fes = tfe * V_space
    return st_fes


class space_time_trace:
    def __init__(self,q,qstar,k,kstar,N,T,delta_t,mesh,stabs,bnd_names,
                 t_slice,u_exact_slice,ut_exact_slice,trace_slice,trace_bottom,
                 tstart=None,told=None,perturbations=None,N_trace=1,
                 mesh_uncurved=None,
                 bonus_intorder_error = 5
                 ):
        self.q = q 
        self.qstar = qstar
        self.k = k 
        self.kstar = kstar
        self.N = N
        self.time_order = 2*max(q,qstar)
        self.tend = T 
        if not tstart:
            self.tstart = -self.tend 
        self.mesh = mesh
        #self.mesh_uncurved = mesh_uncurved 
        self.stabs = stabs
        if "trace-jumps" in self.stabs:
            pass
        else:
            self.stabs["trace-jumps"] = 1 
        

        self.bnd_name = bnd_names
        self.h = specialcf.mesh_size
        self.nF = specialcf.normal(self.mesh.dim)
        # P = I = nn*nn.T (for computing tangential derivative)
        self.P_xx = 1 - self.nF[0]*self.nF[0]
        self.P_xy = -self.nF[0]*self.nF[1]
        self.P_yy = 1 - self.nF[1]*self.nF[1]
        #self.data_boost = 1/self.h**2
        self.data_boost = 1.0
        self.bonus_intorder_error =  bonus_intorder_error 

        #self.delta_t = self.tend / self.N
        self.delta_t  = delta_t
        self.t_slice = t_slice
        self.u_exact_slice = u_exact_slice
        self.ut_exact_slice = ut_exact_slice
        self.trace_slice = trace_slice
        self.trace_bottom = trace_bottom
        self.qr = quad_rule("Gauss-Lobatto",5)
        self.perturbations = perturbations

        self.dxt = self.delta_t * dxtref(self.mesh, time_order=self.time_order)

        self.dxt_ho = self.delta_t * dxtref(self.mesh, time_order=self.time_order + self.bonus_intorder_error, order= 2*self.k + bonus_intorder_error)
    
        self.dxtF =  self.delta_t * dxtref(self.mesh, time_order=self.time_order, skeleton=True)
        self.dxt_omega = self.delta_t * dxtref(self.mesh, time_order=self.time_order,definedon=self.mesh.Materials("omega"))
        self.dmesh = dmesh(self.mesh)
        #self.dmesh = dx
        self.dmesh_BND = dmesh(self.mesh, definedon=mesh.Boundaries(self.bnd_name["trace_BND"]))
        #self.dmesh_BND_bottom = dmesh(self.mesh, definedon=mesh.Boundaries(self.bnd_name["trace_BND"]))
        
        #self.dmesh_BND = ds(definedon=mesh.Boundaries("bc_Omega"))
        
        #self.dst = self.delta_t * dxtref(self.mesh, time_order=self.time_order,skeleton=True, vb=BND)
        self.dst = self.delta_t * dxtref(self.mesh, time_order=self.time_order,skeleton=True, definedon=mesh.Boundaries(self.bnd_name["Omega_BND"])) 
        self.dst_BND = self.delta_t * dxtref(self.mesh, time_order=self.time_order,definedon=mesh.Boundaries(self.bnd_name["trace_BND"]), skeleton=True, vb=BND)
        #self.dst_BND = self.delta_t * dxtref(self.mesh, time_order=self.time_order+3,definedon=mesh.Boundaries(self.bnd_name["trace_BND"]), skeleton=True, vb=BND)
        #self.dst_BND = self.delta_t * dxtref(self.mesh,order=12, time_order=self.time_order+3,definedon=mesh.Boundaries(self.bnd_name["trace_BND"]), skeleton=True, vb=BND)


        self.W_slice_primal = [None for i in range(self.N)] 
        self.W_slice_dual = [None for i in range(self.N)] 
        self.X_slab = [None for i in range(self.N)]  
        self.V_trace_slab = [None for i in range(self.N)]  
        self.V_trace_bottom = [None for i in range(self.N)]  
        self.W_coupling = [None for i in range(self.N)]  
        self.W_forward_coupling = [None for i in range(self.N)]  
        self.W_primal = None
        self.V_trace = None 
        self.W_dual = None
        
        self.X = None

        self.f = None
        self.a_first_slab = None
        self.a_first_slab_inv = None
        #self.a_general_slab = None
        self.a_general_slab_inv = [None for i in range(self.N)]
        self.a_slab_nojumps = [None for i in range(self.N)]

        self.m_coupling = [None for i in range(self.N)]
        self.m_scaled_mass = [None for i in range(self.N)]

        self.gfuS_in = [None for i in range(self.N)]
        self.gfuS_out = [None for i in range(self.N)]
        self.gfuC_in = [None for i in range(self.N)]
        self.gfuC_out = [None for i in range(self.N)]
        self.gfuX = None 
        self.gfuX_in = None
        self.gfuX_out = None
        self.b_in = None
        self.gfu_pre = None
        self.b_rhs_pre = [None for i in range(self.N)]
        self.gfuS_pre = [None for i in range(self.N)]
        self.gfu_fixt_in  = [None for i in range(self.N)]
        self.gfu_fixt_out = [None for i in range(self.N)]

        self.told = told

    def dt(self,u):
        return 1.0 / self.delta_t * dtref(u)

    def SetupSpaceTimeFEs(self): 
        print("Setting up SpaceTimeFEs") 
        for n in range(self.N):
            self.W_slice_primal[n] = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=[])
            self.W_slice_dual[n] = GetSpaceTimeFESpace(self.mesh,self.kstar,self.qstar,bc=[])
            #self.W_slice_dual[n] = GetSpaceTimeFESpace(self.mesh,self.kstar,self.qstar,bc=self.bnd_name["Omega_BND"])

            shapes = CF( tuple(cfi for cfi in self.trace_slice[n]  )  )
            dxlist = [ff.Diff(x) for ff in self.trace_slice[n] ]
            dxshapes = CF( tuple(cfi for cfi in dxlist )  )
            dylist = [ff.Diff(y) for ff in self.trace_slice[n] ]
            dyshapes = CF( tuple(cfi for cfi in dylist )  )
            self.V_trace_slab[n] = GlobalSpace(self.mesh, definedon=self.mesh.Boundaries(self.bnd_name["trace_BND"]), basis=shapes) 
            self.V_trace_slab[n].AddOperator("gdx", BND, dxshapes)
            self.V_trace_slab[n].AddOperator("gdy", BND, dyshapes)
            self.X_slab[n] = FESpace([self.W_slice_primal[n],self.W_slice_primal[n], self.V_trace_slab[n],self.W_slice_dual[n],self.W_slice_dual[n]])

            #print("n = ", n) 
            if n > 0:
                V_space = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=[],return_base=True)
                V_space2 = FESpace([V_space,V_space])
                shapes_bottom = CF( tuple(cfi for cfi in self.trace_bottom[n]))
                self.V_trace_bottom[n] = GlobalSpace(self.mesh, definedon=self.mesh.Boundaries(self.bnd_name["trace_BND"]), basis=shapes_bottom)
                V_trace_bottom2 = FESpace([self.V_trace_bottom[n],self.V_trace_bottom[n]])
                self.W_coupling[n] = FESpace([V_space2,V_space2,V_trace_bottom2])
                #print("W_forward_coupling[{0}]".format(n)) 
                self.W_forward_coupling[n] = FESpace([V_space,V_space,self.V_trace_bottom[n]])


        self.W_primal = FESpace([self.W_slice_primal[n]  for n in range(self.N)])
        self.W_dual = FESpace([self.W_slice_dual[n]  for n in range(self.N)])
        self.V_Trace = FESpace([self.V_trace_slab[n]  for n in range(self.N)])
        self.X = FESpace([self.W_primal,self.W_primal,self.V_Trace,self.W_dual,self.W_dual]) 

        #self.V_space = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=".*",return_base=True)
        
        self.V_space = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=[],return_base=True)
        #self.V_space = GetSpaceTimeFESpace(self.mesh_uncurved,self.k,self.q,bc=[],return_base=True)
        
        #self.V_space2 = FESpace([self.V_space,self.V_space])
        #self.W_coupling = FESpace([self.V_space2,self.V_space2])
        
        #print("gfuS_in  = ", self.gfuS_in) 
        for n in range(self.N):
            self.gfuS_in[n] = GridFunction(self.X_slab[n])
            self.gfuS_out[n] = GridFunction(self.X_slab[n])
            if n > 0:
                self.gfuC_in[n] = GridFunction(self.W_coupling[n])
                self.gfuC_out[n] = GridFunction(self.W_coupling[n])

        self.gfuX = GridFunction(self.X)
        self.gfuX_in = GridFunction(self.X)
        self.gfuX_out = GridFunction(self.X)

    def SetupSlabMatrix(self,a,n):
        u1,u2,mu,z1,z2 = self.X_slab[n].TrialFunction() 
        w1,w2,eta,y1,y2 = self.X_slab[n].TestFunction()

        mu_x = mu.Operator("gdx")
        eta_x = eta.Operator("gdx")
        mu_y = mu.Operator("gdy")
        eta_y = eta.Operator("gdy")

        a += self.dt(w2) * z1 * self.dxt  
        a += grad(w1) * grad(z1) * self.dxt
        a += (self.dt(w1) - w2) * z2 * self.dxt
        a += (-1) * grad(w1) * self.nF * z1 * self.dst

        # (u_1,w_1)_omega
        a += self.data_boost * self.stabs["data"] * u1 * w1 * self.dxt_omega 
       
        # trace term  
        a += self.stabs["trace-proj"] * (u1-mu)*(w1-eta) * self.dst_BND
        #a += self.stabs["trace-proj"] * (1/self.h)*(u1-mu)*(w1-eta) * self.dst_BND
        #a += self.stabs["trace-proj"] * self.h*( grad(u1).Trace()[0] - self.P_xx*mu_x - self.P_xy*mu_y)*( grad(w1).Trace()[0] - self.P_xx*eta_x - self.P_xy*eta_y) * self.dst_BND 
        #a += self.stabs["trace-proj"] * self.h*( grad(u1).Trace()[1] - self.P_xy*mu_x - self.P_yy*mu_y)*( grad(w1).Trace()[1] - self.P_xy*eta_x - self.P_yy*eta_y) * self.dst_BND 

        # S(U_h,W_h) 
        a += self.stabs["primal"] * self.h * InnerProduct( ( grad(u1) - grad(u1).Other()) * self.nF , ( grad(w1) - grad(w1).Other()) * self.nF ) * self.dxtF
        a += self.stabs["primal"] * self.h**2 * InnerProduct(  self.dt(u2) - LaplacianProxy(u1,self.mesh.dim) , self.dt(w2) - LaplacianProxy(w1,self.mesh.dim) ) * self.dxt
        a += self.stabs["primal"] * InnerProduct( u2 - self.dt(u1) , w2 - self.dt(w1) ) * self.dxt
        a += self.stabs["Tikh"] * self.h**(2*min(self.q,self.k)) *  u1 *  w1 * self.dxt

        # A[U_h,Y_h]
        a += self.dt(u2) * y1 * self.dxt  
        a += grad(u1) * grad(y1) * self.dxt
        a += (self.dt(u1) - u2) * y2 * self.dxt
        a += (-1) * grad(u1) * self.nF * y1 * self.dst

        # S*(Y_h,Z_h)
        a += self.stabs["dual"] *  (-1)* y1 * z1 * self.dxt  
        a += self.stabs["dual"] * (-1)* grad(y1) * grad(z1) * self.dxt  
        a += self.stabs["dual"] *  (-1)* y2 * z2 * self.dxt 
        # boundary term
        a += self.stabs["dual"] *  (-1) * (40/self.h) * y1 * z1 * self.dst 
        #a += (-1)*self.stabs["primal"] * self.h * InnerProduct( ( grad(y1) - grad(y1).Other()) * self.nF , ( grad(z1) - grad(z1).Other()) * self.nF ) * self.dxtF
        #a += (-1)*self.stabs["primal"] * self.h**2 * InnerProduct(  self.dt(y2) - LaplacianProxy(y1,self.mesh.dim) , self.dt(z2) - LaplacianProxy(z1,self.mesh.dim) ) * self.dxt
        #a += (-1)*self.stabs["primal"] * InnerProduct( y2 - self.dt(y1) , z2 - self.dt(z1) ) * self.dxt
        #a += (-1)*self.stabs["Tikh"] * self.h**(2*min(self.q,self.k)) *  y1 *  z1 * self.dxt


    def SetupCouplingMatrixBetweenSlices(self,m,n):
        u1,u2,mu = self.W_coupling[n].TrialFunction()  
        w1,w2,eta = self.W_coupling[n].TestFunction()  
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u1[1] - u1[0], w1[1] - w1[0]) * self.dmesh
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u2[1] - u2[0], w2[1] - w2[0]) * self.dmesh
        m += self.stabs["primal-jump-displ-grad"] * self.delta_t * InnerProduct(grad(u1[1]) - grad(u1[0]), grad(w1[1]) - grad(w1[0])) * self.dmesh
        # Jump term for trace space 
        m += self.stabs["trace-jumps"] *  (1/self.delta_t) * (mu[1]-mu[0])*(eta[1]-eta[0]) * self.dmesh_BND 
        #m +=  (1/self.delta_t) * (mu[1]-mu[0])*eta[1] * self.dmesh_BND 
        #m +=  (1/self.delta_t) * (mu[1]-mu[0])*eta[1] * self.dmesh_BND 
         
 

    def SetupScaledMassMatrixBetweenSlices(self,m,n): 
        #print("huhu")
        u1,u2,mu = self.W_forward_coupling[n].TrialFunction() 
        w1,w2,eta = self.W_forward_coupling[n].TestFunction()  
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u1, w1 ) * self.dmesh
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u2, w2 ) * self.dmesh
        m += self.stabs["primal-jump-displ-grad"] * self.delta_t * InnerProduct(grad(u1) , grad(w1)) * self.dmesh
        m += self.stabs["trace-jumps"] * (1/self.delta_t) * mu * eta * self.dmesh_BND 

    def SetupRightHandSide(self):
        u1,u2,mu,z1,z2 = self.X.TrialFunction()
        w1,w2,eta,y1,y2 = self.X.TestFunction()

        f = LinearForm(self.X)
        for n in range(self.N):
            f += self.data_boost * self.stabs["data"] * self.u_exact_slice[n] * w1[n] * self.dxt_omega 
            if self.perturbations:
                f += self.data_boost * self.stabs["data"] * self.perturbations["scal"] * (self.h/self.perturbations["maxh"])**(self.perturbations["exponent"]) * self.perturbations["noise-slice"][n] * w1[n] * self.dxt_omega 
        f.Assemble()
        self.f = f

    def ApplySpaceTimeMat(self,x,y):

        self.gfuX_in.vec.data = x
        self.gfuX_out.vec[:] = 0.0
        
        for n in range(self.N):            
            # Multiplication with Slab Matrix (No DG jumps between slices)
            for j in range(len(self.gfuS_in[n].components)):
                self.gfuS_in[n].components[j].vec.data = self.gfuX_in.components[j].components[n].vec
            self.gfuS_out[n].vec.data = self.a_slab_nojumps[n].mat * self.gfuS_in[n].vec
            for j in range(len(self.gfuS_in[n].components)):
                self.gfuX_out.components[j].components[n].vec.data += self.gfuS_out[n].components[j].vec  
            # Multiplication with DG jump Matrix between slices
            if n > 0: 
                for j in range(len(self.gfuC_in[n].components)):
                    if j in [0,1]:
                        self.gfuC_in[n].components[j].components[0].vec.FV().NumPy()[:] = self.gfuX_in.components[j].components[n-1].vec.FV().NumPy()[self.q*self.V_space.ndof : (self.q+1)*self.V_space.ndof] 
                        self.gfuC_in[n].components[j].components[1].vec.FV().NumPy()[:] = self.gfuX_in.components[j].components[n].vec.FV().NumPy()[ : self.V_space.ndof]
                    elif j == 2:
                        self.gfuC_in[n].components[j].components[0].vec.FV().NumPy()[:] = self.gfuX_in.components[j].components[n-1].vec.FV().NumPy()[:] 
                        self.gfuC_in[n].components[j].components[1].vec.FV().NumPy()[:] = self.gfuX_in.components[j].components[n].vec.FV().NumPy()[:] 
                    else:
                        print("Attention, index j = {0} for gfuC_in")
                self.gfuC_out[n].vec.data =  self.m_coupling[n].mat * self.gfuC_in[n].vec
                for j in range(len(self.gfuC_out[n].components)):
                    if j in [0,1]:
                        self.gfuX_out.components[j].components[n-1].vec.FV().NumPy()[self.q*self.V_space.ndof : (self.q+1)*self.V_space.ndof] += self.gfuC_out[n].components[j].components[0].vec.FV().NumPy()[:]
                        self.gfuX_out.components[j].components[n].vec.FV().NumPy()[ : self.V_space.ndof] += self.gfuC_out[n].components[j].components[1].vec.FV().NumPy()[:]
                    elif j == 2:
                        self.gfuX_out.components[j].components[n-1].vec.FV().NumPy()[:] += self.gfuC_out[n].components[j].components[0].vec.FV().NumPy()[:]
                        self.gfuX_out.components[j].components[n].vec.FV().NumPy()[:] += self.gfuC_out[n].components[j].components[1].vec.FV().NumPy()[:]
                    else:
                        print("Attention, index j = {0} for gfuC_out")


        y.data = self.gfuX_out.vec


    def PreparePrecondGMRes(self):
        # Setup local matrices on time slab
        
        for n in range(self.N): 
            # first slab does not have dg jump terms
            a_first_slab = BilinearForm(self.X_slab[n], symmetric=False)
            #self.SetupSlabMatrix(a_first_slab,0)
            self.SetupSlabMatrix(a_first_slab,n)
            a_first_slab.Assemble() 
            self.a_slab_nojumps[n] = a_first_slab 
            #print("a")
            if n == 0:
                self.a_first_slab = a_first_slab
                self.a_first_slab_inv = self.a_first_slab.mat.Inverse(self.X_slab[n].FreeDofs(), inverse=solver) 
            
            if n > 0: 
                # general slab has them  
                a_general_slab = BilinearForm(self.X_slab[n], symmetric=False)
                self.SetupSlabMatrix(a_general_slab,n)
                u1s,u2s,mus,z1s,z2s = self.X_slab[n].TrialFunction()
                w1s,w2s,etas,y1s,y2s = self.X_slab[n].TestFunction()
                a_general_slab += self.stabs["primal-jump"] * (1/self.delta_t)  * u1s * w1s * dmesh(self.mesh, tref=0)  
                a_general_slab += self.stabs["primal-jump-displ-grad"] * self.delta_t * grad(u1s) * grad(w1s) * dmesh(self.mesh, tref=0)  
                a_general_slab += self.stabs["primal-jump"] * (1/self.delta_t)  * u2s * w2s * dmesh(self.mesh, tref=0)  
                a_general_slab +=  self.stabs["trace-jumps"] * (1/self.delta_t) * mus * etas * dmesh(self.mesh, definedon=self.mesh.Boundaries(self.bnd_name["trace_BND"]),  tref=0)  
                #a_general_slab +=   (1/self.delta_t) * mus * etas * dmesh(self.mesh, tref=0)  
                a_general_slab.Assemble() 
                #self.a_general_slab = a_general_slab
                self.a_general_slab_inv[n] = a_general_slab.mat.Inverse(self.X_slab[n].FreeDofs(), inverse=solver) 

                m_coupling = BilinearForm(self.W_coupling[n], symmetric=False)
                self.SetupCouplingMatrixBetweenSlices(m_coupling,n)
                m_coupling.Assemble()
                #print(" n = ", n )
                #print("m_coupling.mat = ", m_coupling.mat) 
                self.m_coupling[n] = m_coupling
                
                #print("m_scaled_mass, n = {0}".format(n))
                m_scaled_mass = BilinearForm(self.W_forward_coupling[n], symmetric=False)
                self.SetupScaledMassMatrixBetweenSlices(m_scaled_mass,n)
                m_scaled_mass.Assemble() 
                self.m_scaled_mass[n] = m_scaled_mass
        
        self.b_in = GridFunction(self.X)
        self.gfu_pre = GridFunction(self.X)
        #print("A")
        for n in range(self.N):
            self.b_rhs_pre[n] = GridFunction(self.X_slab[n])
            self.gfuS_pre[n] = GridFunction(self.X_slab[n])
            #print("A1")
            if n > 0:
                self.gfu_fixt_in[n]  = GridFunction(self.W_forward_coupling[n]) 
                self.gfu_fixt_out[n] = GridFunction(self.W_forward_coupling[n])  
            #print("A2")
        #print("B")

    def TimeMarching(self,x,y):
        
        self.b_in.vec.data = x
        for n in range(self.N):
            self.gfuS_pre[n].vec[:] = 0.0
            self.b_rhs_pre[n].vec[:] = 0.0
        self.gfu_pre.vec[:] = 0.0

        for n in range(self.N):
            for j in range(len(self.gfuS_in[n].components)):
                self.b_rhs_pre[n].components[j].vec.data += self.b_in.components[j].components[n].vec
            if n == 0:
                self.gfuS_pre[n].vec.data =  self.a_first_slab_inv * self.b_rhs_pre[n].vec
            else: 
                self.gfuS_pre[n].vec.data = self.a_general_slab_inv[n] * self.b_rhs_pre[n].vec
            # impose weak continuity between slices
            self.b_rhs_pre[n].vec[:] = 0.0

            if n != self.N-1:
                for j in range(len(self.gfu_fixt_in[n+1].components)):
                    if j in [0,1]:
                        self.gfu_fixt_in[n+1].components[j].vec.FV().NumPy()[:] = self.gfuS_pre[n].components[j].vec.FV().NumPy()[self.q*self.V_space.ndof : (self.q+1)*self.V_space.ndof] 
                    elif j == 2:
                        self.gfu_fixt_in[n+1].components[j].vec.FV().NumPy()[:] =  self.gfuS_pre[n].components[j].vec.FV().NumPy()[:]
                        #print("self.gfu_fixt_in[n+1].components[j].vec.FV().NumPy()[:] = ", self.gfu_fixt_in[n+1].components[j].vec.FV().NumPy()[:])  
                    else:
                        print("Index j = {0} in TimeMarching".format(j))

                self.gfu_fixt_out[n+1].vec.data =  self.m_scaled_mass[n+1].mat * self.gfu_fixt_in[n+1].vec
                #print("self.m_scaled_mass[n+1].mat  =", self.m_scaled_mass[n+1].mat)
                #print("self.m_coupling[n+1].mat = ", self.m_coupling[n+1].mat)  

            if n != self.N-1:
                for j in range(len(self.gfu_fixt_out[n+1].components)):
                    if j in [0,1]:
                        self.b_rhs_pre[n+1].components[j].vec.FV().NumPy()[0 : self.V_space.ndof] = self.gfu_fixt_out[n+1].components[j].vec.FV().NumPy()[:] 
                    elif j == 2: 
                        self.b_rhs_pre[n+1].components[j].vec.FV().NumPy()[:] = self.gfu_fixt_out[n+1].components[j].vec.FV().NumPy()[:]
                        #print("self.b_rhs_pre[n+1].components[j].vec.FV().NumPy()[:] = " , self.b_rhs_pre[n+1].components[j].vec.FV().NumPy()[:])  
                    else:
                        print("Index j = {0} in TimeMarching".format(j))
                #input("")
            for j in range(len(self.gfuS_in[n].components)):
                self.gfu_pre.components[j].components[n].vec.FV().NumPy()[:]  = self.gfuS_pre[n].components[j].vec.FV().NumPy()[:] 
    
        y.data = self.gfu_pre.vec
        #y.data = x

    def MeasureErrors(self,gfuh,levelset,domain_type=NEG,data_domain=False,Q_all=False): 
        el_type = HASNEG
        chi = IfPos(-levelset,1.0,0.0)
        if domain_type == POS:
            el_type = HASPOS
            chi = IfPos(levelset,1.0,0.0)
        elif domain_type == IF:
            el_type = IF

        fesp1 = H1(self.mesh, order=1, dgjumps=True)
        # Time finite element (nodal!)
        tfe = ScalarTimeFE(self.q)
        # Space-time finite element space
        st_fes = tfe * fesp1
        lset_p1 = GridFunction(st_fes)
        #lset_adap_st = LevelSetMeshAdaptation_Spacetime(self.mesh, order_space=self.k+2,
        #                                                order_time=self.q+2,
        #                                                threshold=0.1,
        #                                                discontinuous_qn=True)
        # 
        #lset_adap_st = LevelSetMeshAdaptation_Spacetime(self.mesh, order_space=1,
        #                                                order_time=1,
        #                                                threshold=0.1,
        #                                                discontinuous_qn=True)

        ci = CutInfo(self.mesh, time_order=0)
        if data_domain:
            dB = self.dxt_omega 
        elif Q_all: 
            #dB = self.dxt
            dB = self.dxt_ho 
        else: 
            #dB = self.delta_t * dCut(lset_adap_st.levelsetp1[INTERVAL], domain_type, time_order=self.time_order,
            #                deformation=lset_adap_st.deformation[INTERVAL],
            #                definedonelements=ci.GetElementsOfType(el_type))
            #dB = self.dxt
            dB = self.dxt_ho 

        l2_error_B_slab = []  
        n = 0 
        self.told.Set(self.tstart)

        while self.tend - self.told.Get() > self.delta_t / 2:
            #SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
            #dfm = lset_adap_st.CalcDeformation(levelset, tref)
            #dfm = lset_adap_st.CalcDeformation(levelset)
            #ci.Update(lset_adap_st.levelsetp1[INTERVAL], time_order=0)
            #ci.Update(lset_p1, time_order=0)
             
            #if n == 0:
            #    if data_domain or Q_all:
            #        l2_error_B_slab.append(Integrate((self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * dB, self.mesh))
            #    else: 
            #        l2_error_B_slab.append(Integrate(chi*(self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * dB, self.mesh))
            #
            #    #Draw(BitArrayCF(ci.GetElementsOfType(el_type)),self.mesh,"markedEL")
            #    #input("")
            #else:
                #tnn = (n+1)*self.delta_t - self.tend 
                #theta = (tnn - self.t_slice[n] ) / self.delta_t
                #uh_minus = CreateTimeRestrictedGF(gfuh.components[0].components[n-1], 1)
                #uh_minus.Set(fix_tref(gfuh.components[0].components[n-1], 1))
                #uh_plus = CreateTimeRestrictedGF(gfuh.components[0].components[n], 0)
                #uh_plus.Set(fix_tref(gfuh.components[0].components[n], 0))
                #l2_error_B_slab.append(Integrate((self.u_exact_slice[n]  - (gfuh.components[0].components[n] - theta * ( uh_plus - uh_minus  )  ) )**2 * dB, self.mesh))
                
                #l2_error_B_slab.append(Integrate((self.u_exact_slice[n]  - (gfuh.components[0].components[n]   ) )**2 * dB, self.mesh))
                
            if data_domain or Q_all:
                l2_error_B_slab.append(Integrate( (self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * dB, self.mesh))
            else: 
                l2_error_B_slab.append(Integrate(chi*(self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * dB, self.mesh))


            #print(" n = {0}, l2_error_B_slab = {1}".format(n,l2_error_B_slab[-1]))
            
            self.told.Set(self.told.Get() + self.delta_t)
            n += 1 
        
        return sqrt( sum( l2_error_B_slab)  )


    def MeasureL2Errors(self,gfuh): 
        l2_error_sum = 0.0
        for n in range(self.N):
            if n == 0:
                l2_error_sum += Integrate((self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * self.dxt, self.mesh)
            else:
                tnn = (n+1)*self.delta_t - self.tend 
                theta = (tnn - self.t_slice[n] ) / self.delta_t
                uh_minus = CreateTimeRestrictedGF(gfuh.components[0].components[n-1], 1)
                uh_minus.Set(fix_tref(gfuh.components[0].components[n-1], 1))
                uh_plus = CreateTimeRestrictedGF(gfuh.components[0].components[n], 0)
                uh_plus.Set(fix_tref(gfuh.components[0].components[n], 0))
                l2_error_sum += Integrate((self.u_exact_slice[n]  - (gfuh.components[0].components[n] - theta * ( uh_plus - uh_minus  )  ) )**2 * self.dxt, self.mesh)
                
        l2_error_sum = sqrt(l2_error_sum)  
        print("L2-L2-error = ", l2_error_sum  )
     
