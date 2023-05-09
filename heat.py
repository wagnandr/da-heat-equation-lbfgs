import numpy as np
import scipy as sp
# due to internal initializations in scipy, 
# this has to be imported before dolfin and dolfin_adjoint!
from scipy.optimize import minimize
import dolfin as df
import dolfin_adjoint as dfa


def forward(V, dt, kappa, lanbda, phase, num_steps, annotate, dx):
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    u_prev = dfa.Function(V, name='u_prev', annotate=annotate)
    #u_prev = dfa.project(dfa.Expression('sin(2*pi*x[0]) * sin(2*pi*x[1])', degree=2), V, annotate=True)
    x = df.SpatialCoordinate(V.mesh())
    u_prev = dfa.project(df.sin(2 * df.pi * x[0] ) * df.sin(2 * df.pi * phase * x[1] ), V, annotate=annotate)

    a = df.inner(u, v)*dx 
    a += dt * dfa.Constant(0.5) * kappa[0] * df.inner(df.grad(u), df.grad(v)) * dx(0)
    a += dt * dfa.Constant(0.5) * kappa[1] * df.inner(df.grad(u), df.grad(v)) * dx(1)
    L = df.inner(u_prev, v)*dx 
    L += - dt * dfa.Constant(0.5) * kappa[0] * df.inner(df.grad(u_prev), df.grad(v)) * dx(0)
    L += - dt * dfa.Constant(0.5) * kappa[1] * df.inner(df.grad(u_prev), df.grad(v)) * dx(1)
    L += lanbda * dt * u_prev * (1 - u_prev) * v * dx

    u = dfa.Function(V, name='u', annotate=annotate)

    for i in range(num_steps):
        dfa.solve(a == L, u, annotate=annotate)
        u_prev.assign(u, annotate=annotate)
    
    return u


class OptimizationProblem:
    def __init__(self, dt, num_steps, V, u_sol,dx):
        self.dt = dt
        self.num_steps = num_steps
        self.V = V
        self.u_sol = u_sol
        self.dx=dx
    
    def _run(self, x):
        dt = dfa.Constant(self.dt)
        kappa1 = dfa.Constant(x[0])
        kappa2 = dfa.Constant(x[1])
        kappa = [kappa1, kappa2]
        landa = dfa.Constant(x[2])
        phase = dfa.Constant(x[3])
        return forward(self.V, dt, kappa, landa, phase, self.num_steps, True, dx=self.dx)

    def _eval(self, x):
        dt = dfa.Constant(self.dt)
        kappa1 = dfa.Constant(x[0])
        kappa2 = dfa.Constant(x[1])
        kappa = [kappa1, kappa2]
        landa = dfa.Constant(x[2])
        phase = dfa.Constant(x[3])
        u_approx = forward(self.V, dt, kappa, landa, phase, self.num_steps, True, dx=self.dx)
        J = dfa.assemble((0.5 * df.inner(self.u_sol - u_approx, self.u_sol - u_approx)) * df.dx)
        return J, [kappa1, kappa2, landa, phase]
    
    def _jacobian(self, x):
        J, params = self._eval(x)
        res = dfa.compute_gradient(J, [dfa.Control(p) for p in params])
        res = np.array([x.values()[0] for x in res])
        return res 
    
    def create_evaluation(self):
        return lambda *args, **kwargs: self._eval(*args, **kwargs)[0]

    def create_jacobian(self):
        return lambda *args, **kwargs: self._jacobian(*args, **kwargs)


if __name__ == '__main__':
    df.set_log_active(False)

    n = 40
    mesh = dfa.UnitSquareMesh(n, n)
    #mesh = dfa.BoxMesh(df.Point(0,0,0), df.Point(1,1,1), self.n, self.n, self.n)
    V = df.FunctionSpace(mesh, "CG", 1)

    # create an artificial test problem to compare true:
    kappa_true = [dfa.Constant(2.5), dfa.Constant(4.5)]
    landa_true = dfa.Constant(3.75)
    phase_true = dfa.Constant(2.)
    dt = 1e-2
    num_steps = 10

    cf = df.MeshFunction("size_t", mesh, mesh.geometric_dimension())
    subdomain = df.CompiledSubDomain('std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
    cf.set_all(0)
    subdomain.mark(cf, 1)
    dx = df.Measure('dx', domain=mesh, subdomain_data=cf)

    u_sol = forward(V, dfa.Constant(dt), kappa_true, landa_true, phase_true, num_steps, False, dx=dx)

    # initialize optimization problem
    problem = OptimizationProblem(dt=dt,num_steps=num_steps,V=V,u_sol=u_sol,dx=dx)

    # initial guess
    x0 = np.array([1.0, 1.0, 0.0, 1.0])

    # execute the scipy LBFGSB implementation
    res = sp.optimize.minimize(
        fun=problem.create_evaluation(),
        jac=problem.create_jacobian(),
        method = 'L-BFGS-B',
        # initial guess
        x0=x0,
        options={
            # 'disp': True, 
            'maxiter': 1000,
            'gtol': 1e-16,
            'ftol': 1e-16,
            #'maxcor': 1,
            'maxls': 100
        },
        bounds=sp.optimize.Bounds(np.array([0.01, 0.01, 0., 0.]), np.array([100, 100, 100, 10]), True),
        callback=lambda *args: print(f'intermediate result {args}')
    )

    print(f'inferred {res.x}')
    print(f'value {problem._eval(x0)[0]}')
    print(f'value {problem._eval(res.x)[0]}')

    u_approx = problem._run(res.x) 
    u_init = problem._run(x0) 
    df.File('output/u_exact.pvd') << u_sol, 0
    df.File('output/u_approx.pvd') << u_approx, 0
    df.File('output/u_init.pvd') << u_init, 0
