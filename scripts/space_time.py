from ngsolve import *
from xfem import *
from xfem.lset_spacetime import *
from ngsolve.comp import GlobalSpace
solver = "pardiso"
#solver = "umfpack"

class quad_rule:

    def __init__(self,name,npoints):
        self.name = name
        self.npoints = npoints

        # available quadrature rules
        gauss_lobatto = {
            3: ( [ -1, 0, 1 ],
                 [ 1/3, 4/3, 1/3 ] ),
            4: ( [ -1, -sqrt(1/5), sqrt(1/5), 1],
                 [ 1/6, 5/6, 5/6, 1/6 ] ),
            5: ( [ -1, -(1/7)*sqrt(21),0.0, (1/7)*sqrt(21), 1.0 ],
                 [ 1/10,49/90,32/45, 49/90, 1/10  ] ),
            6: ( [ -1, -sqrt((1/21)*(7+2*sqrt(7))), -sqrt((1/21)*(7-2*sqrt(7))), sqrt((1/21)*(7-2*sqrt(7))), sqrt((1/21)*(7+2*sqrt(7))), 1.0 ],
                 [ 1/15, (1/30)*(14-sqrt(7)), (1/30)*(14+sqrt(7)), (1/30)*(14+sqrt(7)), (1/30)*(14-sqrt(7)),  1/15 ] ),
        }

        if name == "Gauss-Lobatto":
            self.points = gauss_lobatto[npoints][0]
            self.weights = gauss_lobatto[npoints][1]

    def current_pts(self,a,b):
        if self.name == "Gauss-Lobatto":
            return [0.5*(b-a) * pt + 0.5*(b+a)  for pt in self.points]
    
    def t_weights(self,delta_t):
        if self.name == "Gauss-Lobatto":
            return [0.5*delta_t*w for w in self.weights]

def GetSpaceTimeFESpace(mesh,order_space,order_time,bc=".*",return_base=False):
    V_space = H1(mesh, order=order_space, dirichlet=bc,dgjumps=True)
    if return_base:
        return V_space
    tfe = ScalarTimeFE(order_time)
    st_fes = tfe * V_space
    return st_fes

def LaplacianProxy(u,d):
    return sum([ u.Operator("hesse")[i,i] for i in range(d)])


class SpaceTimeMat(BaseMatrix):
    
    def __init__(self,ndof,mult):
        super().__init__() 
        self.ndof = ndof
        self.mult = mult

    def IsComplex(self):
        return False

    def Height(self):
        return self.ndof

    def Width(self):
        return self.ndof
    
    def Mult(self, x, y):
        self.mult(x,y)


class space_time:
    def __init__(self,q,qstar,k,kstar,N,T,delta_t,mesh,stabs,t_slice,u_exact_slice,ut_exact_slice,tstart=None,told=None,perturbations=None, bonus_intorder_error = 5, shift=None):
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
        self.stabs = stabs
        self.h = specialcf.mesh_size
        self.nF = specialcf.normal(self.mesh.dim)
        #self.delta_t = self.tend / self.N
        self.delta_t  = delta_t
        self.t_slice = t_slice
        self.u_exact_slice = u_exact_slice
        self.ut_exact_slice = ut_exact_slice
        self.qr = quad_rule("Gauss-Lobatto",5)
        self.perturbations = perturbations
        self.bonus_intorder_error =  bonus_intorder_error 
        self.shift = shift

        self.dxt = self.delta_t * dxtref(self.mesh, time_order=self.time_order)
        self.dxt_ho = self.delta_t * dxtref(self.mesh, time_order=self.time_order + self.bonus_intorder_error, order= 2*self.k + bonus_intorder_error)
        self.dxtF =  self.delta_t * dxtref(self.mesh, time_order=self.time_order, skeleton=True)
        self.dxt_omega = self.delta_t * dxtref(self.mesh, time_order=self.time_order,definedon=self.mesh.Materials("omega"))
        #self.dxt_omega = self.dxt # only for testing 
        self.dmesh = dmesh(self.mesh)
        #self.dst = self.delta_t * dxtref(self.mesh, time_order=self.time_order,skeleton=True, vb=BND)
        self.dst = self.delta_t * dxtref(self.mesh, time_order=self.time_order,skeleton=True, definedon=mesh.Boundaries("bc_Omega"))

        self.W_slice_primal = None 
        self.W_slice_dual = None
        self.X_slab = None 
        self.W_primal = None
        self.W_dual = None
        self.X = None

        self.f = None
        self.a_first_slab = None
        self.a_first_slab_inv = None
        self.a_general_slab = None
        self.a_general_slab_inv = None
        self.a_slab_nojumps = None

        self.m_coupling = None
        self.m_scaled_mass = None
        self.mass_slab = None 

        gfuS_in = None
        gfuS_out = None
        gfuC_in = None
        gfuC_out = None 
        self.gfuX = None 
        self.gfuX_in = None
        self.gfuX_out = None
        self.b_in = None
        self.gfu_pre = None
        self.b_rhs_pre = None
        self.gfuS_pre = None
        self.gfu_fixt_in  = None
        self.gfu_fixt_out = None

        self.told = told

    def dt(self,u):
        return 1.0 / self.delta_t * dtref(u)

    def SetupSpaceTimeFEs(self): 
        #self.W_slice_primal = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc="bc_Omega")
        self.W_slice_primal = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=[])
        #self.W_slice_dual = GetSpaceTimeFESpace(self.mesh,self.kstar,self.qstar,bc="bc_Omega")
        self.W_slice_dual = GetSpaceTimeFESpace(self.mesh,self.kstar,self.qstar,bc=[])
        self.X_slab = FESpace([self.W_slice_primal,self.W_slice_primal,self.W_slice_dual,self.W_slice_dual]) 
        self.W_primal = FESpace([self.W_slice_primal  for n in range(self.N)])
        self.W_dual = FESpace([self.W_slice_dual  for n in range(self.N)])
        self.X = FESpace([self.W_primal,self.W_primal,self.W_dual,self.W_dual]) 

        #self.V_space = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=".*",return_base=True)
        self.V_space = GetSpaceTimeFESpace(self.mesh,self.k,self.q,bc=[],return_base=True)
        self.V_space2 = FESpace([self.V_space,self.V_space])
        self.W_coupling = FESpace([self.V_space2,self.V_space2])

        self.gfuS_in = GridFunction(self.X_slab)
        self.gfuS_out = GridFunction(self.X_slab)

        self.gfuC_in = GridFunction(self.W_coupling)
        self.gfuC_out = GridFunction(self.W_coupling)

        self.gfuX = GridFunction(self.X)
        self.gfuX_in = GridFunction(self.X)
        self.gfuX_out = GridFunction(self.X)

    def SetupSlabMatrix(self,a):
        u1,u2,z1,z2 = self.X_slab.TrialFunction() 
        w1,w2,y1,y2 = self.X_slab.TestFunction()
    
        a += self.dt(w2) * z1 * self.dxt  
        a += grad(w1) * grad(z1) * self.dxt
        a += (self.dt(w1) - w2) * z2 * self.dxt
        a += (-1) * grad(w1) * self.nF * z1 * self.dst

        # (u_1,w_1)_omega
        a += self.stabs["data"] * u1 * w1 * self.dxt_omega 

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
        
        #a += self.stabs["dual"] *  (-1)* y1 * z1 * self.delta_t * dxtref(self.mesh, time_order=self.time_order, definedon=self.mesh.Boundaries("bc_Omega"))
        #a += self.stabs["dual"] *  (-1)* y1 * z1 * self.delta_t * dxtref(self.mesh, time_order=self.time_order,element_vb=BND)
        #for tau_i,omega_i in zip(self.qr.current_pts(0,1),self.qr.t_weights(1)):
        #    #a += (-1) * omega_i * self.delta_t * fix_tref(y1,tau_i).Trace() *  fix_tref(z1,tau_i).Trace() * self.dmesh(definedon=self.mesh.Boundaries("bc_Omega"),element_vb=BND)
        #    a += (-1) * omega_i * self.delta_t * fix_tref(y1,tau_i) *  fix_tref(z1,tau_i) * self.dmesh(definedon=self.mesh.Boundaries("bc_Omega"),element_vb=BND)
        #    #a += (-1) * omega_i * self.delta_t * fix_tref(y1,tau_i) *  fix_tref(z1,tau_i) * self.dmesh(definedon=self.mesh.Boundaries("bc_Omega"), skeleton=True, element_vb=BND)
        #    a += (-1) * omega_i * self.delta_t * y1 * z1 * dmesh(self.mesh, tref=tau_i,definedon=self.mesh.Boundaries("bc_Omega"),element_vb=BND)

        # only if shift (for eigenvalue computation) 
        if self.shift:
            print("Setting up shifted system for eigenvalue computation")
            a += (-1) *  self.shift * u1 *  w1 * self.dxt
            a += (-1) *  self.shift * u2 *  w2 * self.dxt
            a += (-1) *  self.shift * y1 *  z1 * self.dxt
            a += (-1) *  self.shift * y2 *  z2 * self.dxt

        # remove (only for testing)
        #a += 1.0 * (40/self.h) * u1 * w1 * self.dst

    def SetupSlabMassMatrix(self,scale = 1.0, set_z_zero=False):
        m = BilinearForm(self.X_slab, symmetric=False)
        u1,u2,z1,z2 = self.X_slab.TrialFunction() 
        w1,w2,y1,y2 = self.X_slab.TestFunction()
        m += scale * u1 *  w1 * self.dxt
        m += scale * u2 *  w2 * self.dxt
        if not set_z_zero:
            m += scale *  y1 *  z1 * self.dxt
            m += scale *  y2 *  z2 * self.dxt
        m.Assemble() 
        self.mass_slab = m 
        
    
    def SetupCouplingMatrixBetweenSlices(self,m):
        u1,u2 = self.W_coupling.TrialFunction()  
        w1,w2 = self.W_coupling.TestFunction()  
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u1[1] - u1[0], w1[1] - w1[0]) * self.dmesh
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u2[1] - u2[0], w2[1] - w2[0]) * self.dmesh
        m += self.stabs["primal-jump-displ-grad"] * self.delta_t * InnerProduct(grad(u1[1]) - grad(u1[0]), grad(w1[1]) - grad(w1[0])) * self.dmesh

    def SetupScaledMassMatrixBetweenSlices(self,m): 
        u1,u2 = self.V_space2.TrialFunction() 
        w1,w2 = self.V_space2.TestFunction() 
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u1, w1 ) * self.dmesh
        m += self.stabs["primal-jump"] * (1/self.delta_t) * InnerProduct( u2, w2 ) * self.dmesh
        m += self.stabs["primal-jump-displ-grad"] * self.delta_t * InnerProduct(grad(u1) , grad(w1)) * self.dmesh
     
    def SetupRightHandSide(self):
        u1,u2,z1,z2 = self.X.TrialFunction()
        w1,w2,y1,y2 = self.X.TestFunction()

        f = LinearForm(self.X)
        for n in range(self.N):
            f +=  self.stabs["data"] * self.u_exact_slice[n] * w1[n] * self.dxt_omega 
            if self.perturbations:
                print("Adding noise to rhs")
                print("factor = ", self.stabs["data"] * self.perturbations["scal"] ) 
                #f +=  self.stabs["data"] * self.perturbations["scal"] * (self.h/self.perturbations["maxh"])**(self.perturbations["exponent"]) * self.perturbations["noise-slice"][n] * w1[n] * self.dxt_omega 
                f +=  self.stabs["data"] * self.perturbations["scal"] * (self.delta_t)**(self.perturbations["exponent"]) * self.perturbations["noise-slice"][n] * w1[n] * self.dxt_omega 
        f.Assemble()
        self.f = f

    def ApplySpaceTimeMat(self,x,y):

        self.gfuX_in.vec.data = x
        self.gfuX_out.vec[:] = 0.0
        
        for n in range(self.N):            
            # Multiplication with Slab Matrix (No DG jumps between slices)
            for j in range(len(self.gfuS_in.components)):
                self.gfuS_in.components[j].vec.data = self.gfuX_in.components[j].components[n].vec
            self.gfuS_out.vec.data = self.a_slab_nojumps.mat * self.gfuS_in.vec
            for j in range(len(self.gfuS_in.components)):
                self.gfuX_out.components[j].components[n].vec.data += self.gfuS_out.components[j].vec  
            # Multiplication with DG jump Matrix between slices
            if n > 0: 
                for j in range(len(self.gfuC_in.components)):
                    self.gfuC_in.components[j].components[0].vec.FV().NumPy()[:] = self.gfuX_in.components[j].components[n-1].vec.FV().NumPy()[self.q*self.V_space.ndof : (self.q+1)*self.V_space.ndof] 
                    self.gfuC_in.components[j].components[1].vec.FV().NumPy()[:] = self.gfuX_in.components[j].components[n].vec.FV().NumPy()[ : self.V_space.ndof] 
                self.gfuC_out.vec.data =  self.m_coupling.mat * self.gfuC_in.vec
                for j in range(len(self.gfuC_out.components)):
                    self.gfuX_out.components[j].components[n-1].vec.FV().NumPy()[self.q*self.V_space.ndof : (self.q+1)*self.V_space.ndof] += self.gfuC_out.components[j].components[0].vec.FV().NumPy()[:]
                    self.gfuX_out.components[j].components[n].vec.FV().NumPy()[ : self.V_space.ndof] += self.gfuC_out.components[j].components[1].vec.FV().NumPy()[:] 

        y.data = self.gfuX_out.vec



    def ApplySpaceTimeMassMat(self,x,y):

        self.gfuX_in.vec.data = x
        self.gfuX_out.vec[:] = 0.0
        
        for n in range(self.N):            
            # Multiplication with Slab Matrix (No DG jumps between slices)
            for j in range(len(self.gfuS_in.components)):
                self.gfuS_in.components[j].vec.data = self.gfuX_in.components[j].components[n].vec
            self.gfuS_out.vec.data = self.mass_slab.mat * self.gfuS_in.vec
            for j in range(len(self.gfuS_in.components)):
                self.gfuX_out.components[j].components[n].vec.data += self.gfuS_out.components[j].vec  

        y.data = self.gfuX_out.vec



    def PreparePrecondGMRes(self):
        # Setup local matrices on time slab
        # first slab does not have dg jump terms
        a_first_slab = BilinearForm(self.X_slab, symmetric=False)
        self.SetupSlabMatrix(a_first_slab)
        a_first_slab.Assemble() 
        self.a_first_slab = a_first_slab
        self.a_slab_nojumps = self.a_first_slab 
        self.a_first_slab_inv = self.a_first_slab.mat.Inverse(self.X_slab.FreeDofs(), inverse=solver) 
 
        # general slab has them 
        a_general_slab = BilinearForm(self.X_slab, symmetric=False)
        self.SetupSlabMatrix(a_general_slab)
        u1s,u2s,z1s,z2s = self.X_slab.TrialFunction()
        w1s,w2s,y1s,y2s = self.X_slab.TestFunction()
        a_general_slab += self.stabs["primal-jump"] * (1/self.delta_t)  * u1s * w1s * dmesh(self.mesh, tref=0)  
        a_general_slab += self.stabs["primal-jump-displ-grad"] * self.delta_t * grad(u1s) * grad(w1s) * dmesh(self.mesh, tref=0)  
        a_general_slab += self.stabs["primal-jump"] * (1/self.delta_t)  * u2s * w2s * dmesh(self.mesh, tref=0)  
        a_general_slab.Assemble() 
        self.a_general_slab = a_general_slab
        self.a_general_slab_inv = self.a_general_slab.mat.Inverse(self.X_slab.FreeDofs(), inverse=solver) 

        m_coupling = BilinearForm(self.W_coupling, symmetric=False)
        self.SetupCouplingMatrixBetweenSlices(m_coupling)
        m_coupling.Assemble() 
        self.m_coupling = m_coupling

        m_scaled_mass = BilinearForm(self.V_space2 , symmetric=False)
        self.SetupScaledMassMatrixBetweenSlices(m_scaled_mass)
        m_scaled_mass.Assemble() 
        self.m_scaled_mass = m_scaled_mass

        self.b_in = GridFunction(self.X)
        self.gfu_pre = GridFunction(self.X)
        self.b_rhs_pre = GridFunction(self.X_slab)
        self.gfuS_pre = GridFunction(self.X_slab)
        self.gfu_fixt_in  = GridFunction(self.V_space2) 
        self.gfu_fixt_out = GridFunction(self.V_space2) 

    def TimeMarching(self,x,y):
        
        self.b_in.vec.data = x
        self.gfuS_pre.vec[:] = 0.0
        self.gfu_pre.vec[:] = 0.0
        self.b_rhs_pre.vec[:] = 0.0

        for n in range(self.N):
            for j in range(len(self.gfuS_in.components)):
                self.b_rhs_pre.components[j].vec.data += self.b_in.components[j].components[n].vec
            if n == 0:
                self.gfuS_pre.vec.data =  self.a_first_slab_inv * self.b_rhs_pre.vec
            else: 
                self.gfuS_pre.vec.data = self.a_general_slab_inv * self.b_rhs_pre.vec
            # impose weak continuity between slices
            self.b_rhs_pre.vec[:] = 0.0
            for j in range(len(self.gfu_fixt_in.components)):
                self.gfu_fixt_in.components[j].vec.FV().NumPy()[:] = self.gfuS_pre.components[j].vec.FV().NumPy()[self.q*self.V_space.ndof : (self.q+1)*self.V_space.ndof] 
            self.gfu_fixt_out.vec.data =  self.m_scaled_mass.mat * self.gfu_fixt_in.vec
            
            for j in range(len(self.gfu_fixt_out.components)):
                self.b_rhs_pre.components[j].vec.FV().NumPy()[0 : self.V_space.ndof] = self.gfu_fixt_out.components[j].vec.FV().NumPy()[:] 

            for j in range(len(self.gfuS_in.components)):
                self.gfu_pre.components[j].components[n].vec.FV().NumPy()[:]  = self.gfuS_pre.components[j].vec.FV().NumPy()[:] 
    
        y.data = self.gfu_pre.vec
        #y.data = x
    
    '''
    def MeasureErrors(self,gfuh,levelset,domain_type=NEG,data_domain=False): 
        el_type = HASNEG
        if domain_type == POS:
            el_type = HASPOS
        elif domain_type == IF:
            el_type = IF

        fesp1 = H1(self.mesh, order=1, dgjumps=True)
        # Time finite element (nodal!)
        tfe = ScalarTimeFE(self.q)
        # Space-time finite element space
        st_fes = tfe * fesp1
        lset_p1 = GridFunction(st_fes)
        lset_adap_st = LevelSetMeshAdaptation_Spacetime(self.mesh, order_space=self.k,
                                                        order_time=self.q,
                                                        threshold=0.1,
                                                        discontinuous_qn=True)
        ci = CutInfo(self.mesh, time_order=0)
        if data_domain:
            dB = self.dxt_omega 
        else:
            dB = self.delta_t * dCut(lset_adap_st.levelsetp1[INTERVAL], domain_type, time_order=self.time_order,
                            deformation=lset_adap_st.deformation[INTERVAL],
                            definedonelements=ci.GetElementsOfType(el_type))
        l2_error_B_slab = []  
        n = 0 
        self.told.Set(self.tstart)

        while self.tend - self.told.Get() > self.delta_t / 2:
            SpaceTimeInterpolateToP1(levelset, tref, lset_p1)
            dfm = lset_adap_st.CalcDeformation(levelset, tref)
            ci.Update(lset_p1, time_order=0)
             
            if n == 0:
                l2_error_B_slab.append(Integrate((self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * dB, self.mesh))
                #Draw(BitArrayCF(ci.GetElementsOfType(el_type)),self.mesh,"markedEL")
                #input("")
            else:
                tnn = (n+1)*self.delta_t - self.tend 
                theta = (tnn - self.t_slice[n] ) / self.delta_t
                uh_minus = CreateTimeRestrictedGF(gfuh.components[0].components[n-1], 1)
                uh_minus.Set(fix_tref(gfuh.components[0].components[n-1], 1))
                uh_plus = CreateTimeRestrictedGF(gfuh.components[0].components[n], 0)
                uh_plus.Set(fix_tref(gfuh.components[0].components[n], 0))
                l2_error_B_slab.append(Integrate((self.u_exact_slice[n]  - (gfuh.components[0].components[n] - theta * ( uh_plus - uh_minus  )  ) )**2 * dB, self.mesh))
            
            self.told.Set(self.told.Get() + self.delta_t)
            n += 1 
        
        return sqrt( sum( l2_error_B_slab)  )
    '''

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
            #dB = self.dxt_omega 
            #self.dxt_ho = self.delta_t * dxtref(self.mesh, time_order=self.time_order + self.bonus_intorder_error, order= 2*self.k + bonus_intorder_error)
            dB =  self.delta_t * dxtref(self.mesh, time_order=self.time_order + self.bonus_intorder_error , order= 2*self.k + self.bonus_intorder_error, definedon=self.mesh.Materials("omega"))
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
     
    '''
    def MeasureL2ErrorsLset(self,gfuh): 
        l2_error_sum = 0.0
        for n in range(self.N):
            if n == 0:
                l2_error_sum += Integrate((self.u_exact_slice[n]  - gfuh.components[0].components[n])**2 * self.dxt, self.mesh)
            else:
                tnn = (n+1)*self.delta_t 
                theta = (tnn - self.t_slice[n] ) / self.delta_t
                uh_minus = CreateTimeRestrictedGF(gfuh.components[0].components[n-1], 1)
                uh_minus.Set(fix_tref(gfuh.components[0].components[n-1], 1))
                uh_plus = CreateTimeRestrictedGF(gfuh.components[0].components[n], 0)
                uh_plus.Set(fix_tref(gfuh.components[0].components[n], 0))
                l2_error_sum += Integrate((self.u_exact_slice[n]  - (gfuh.components[0].components[n] - theta * ( uh_plus - uh_minus  )  ) )**2 * self.dxt, self.mesh)
                
        l2_error_sum = sqrt(l2_error_sum)  
        print("L2-L2-error = ", l2_error_sum  )
    '''

