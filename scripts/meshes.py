from netgen.geom2d import CSG2d, Circle, Rectangle, SplineGeometry
from ngsolve import Mesh
from ngsolve.webgui import Draw

def GetMeshHalfCylinder(R_Omega,R_data,maxh,rename_bnd=False,order=3):
    geo = CSG2d()
    # define some primitives
    circle = Circle( center=(0,0), radius=R_Omega, bc="bc_Omega" )
    data_circle = Circle( center=(0,0), radius=R_data)
    if rename_bnd:
        rect_bnd = "bc_axis"
    else:
        rect_bnd = "bc_Omega"
    rect = Rectangle( pmin=(0,-R_Omega), pmax=(R_Omega,R_Omega), bc=rect_bnd)
    small_rect = Rectangle( pmin=(0,-R_data), pmax=(R_data,R_data), bc=rect_bnd)

    # use operators +, - and * for union, difference and intersection operations
    big_half_ball = circle - rect
    small_half_ball = data_circle - small_rect 

    data_domain = big_half_ball - small_half_ball 
    nodata_domain = small_half_ball 

    data_domain.Mat("omega")
    nodata_domain.Mat("unknown")
    #nodata_domain.Maxh(maxh/2)
    #domain2 = circle * rect
    #domain2.Mat("mat3").Maxh(0.1) # change domain name and maxh
    #domain3 = rect-circle

    # add top level objects to geometry 
    geo.Add(nodata_domain)
    geo.Add(data_domain)
    #geo.Add(domain2)
    #geo.Add(domain3)

    # generate mesh
    m = geo.GenerateMesh(maxh=maxh)
    # use NGSolve just for visualization
    mesh = Mesh(m)
    #mesh.Curve(order)
    #Draw(mesh)
    #cf = mesh.RegionCF(VOL, dict(mat1=0, mat2=4, mat3=7))
    #Draw(cf, mesh)
    return mesh


def GetMeshDataAllAround(maxh):
    geo = SplineGeometry() 
    # data domain
    p1 = geo.AppendPoint (0.0,0.0)
    p2 = geo.AppendPoint (1.0,0.0)
    p3 = geo.AppendPoint (1.0,1.0)
    p4 = geo.AppendPoint (0.0,1.0)

    p5 = geo.AppendPoint (0.25,0.25)
    p6 = geo.AppendPoint (0.75,0.25)
    p7 = geo.AppendPoint (0.75,0.75)
    p8 = geo.AppendPoint (0.25,0.75)

    # omega
    geo.Append (["line", p1, p2], leftdomain=1, rightdomain=0,bc="bc_Omega")
    geo.Append (["line", p2, p3], leftdomain=1, rightdomain=0,bc="bc_Omega")
    geo.Append (["line", p3, p4], leftdomain=1, rightdomain=0,bc="bc_Omega")
    geo.Append (["line", p4, p1], leftdomain=1, rightdomain=0,bc="bc_Omega")

    # only B 
    geo.Append (["line", p5, p6], leftdomain=2, rightdomain=1)
    geo.Append (["line", p6, p7], leftdomain=2, rightdomain=1)
    geo.Append (["line", p7, p8], leftdomain=2, rightdomain=1)
    geo.Append (["line", p8, p5], leftdomain=2, rightdomain=1)

    geo.SetMaterial(1, "omega")
    geo.SetMaterial(2, "only_B")
    nmesh = geo.GenerateMesh(maxh=maxh)
    mesh = Mesh(nmesh)
    return mesh
