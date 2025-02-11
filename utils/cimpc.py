from tqdm import tqdm
import crocoddyl
import pinocchio
import numpy as np

from .models import DAD_contact, DAM_contact, IAM_contact, IAM_shoot
from .costs import set_la_cost

def cimpc(total_time, x0, actionmodels, DT=2.5e-2):
    steps = np.floor(total_time/DT).astype(int)
    maxiter = 4
    is_feasible = False
    init_reg = 0.1 # 重要
    
    traj = []
    inp_traj = []

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverFDDP(problem)
    xs = [x0] * (solver.problem.T + 1) # list of array即可
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    # us = [np.zeros(12)]*solver.problem.T
    print('Initialization success !')
    solver.solve(xs, us, maxiter, is_feasible, init_reg)
    
    print(f'Initial cost: {solver.cost}')
    
    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])
    
    for i in tqdm(range(steps)):
        
        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12)) # 巨重要
        # ui.append(ui[-1])
        
        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverFDDP(problem)
        
        solver.solve(xi, ui, maxiter, is_feasible, init_reg)
        
        print(f'Step {i+1}, cost: {solver.cost}')
        
        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])
        
    return (traj,inp_traj)

def cimpc_adaptive(total_time, x0, action_component, air_eps=3e-2):
    N, state, actuation, costs, contact_model, DT, rho = action_component
    costs_r, costs_t=costs
    
    steps = np.floor(total_time/DT).astype(int)
    maxiter = 4
    is_feasible = False
    init_reg = 0.1
    
    traj = []
    inp_traj = []
    
    actionmodels = IAM_shoot(N,state, actuation, [costs_r,costs_t], contact_model, DT, rho)

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverFDDP(problem)
    xs = [x0] * (solver.problem.T + 1) # list of array即可
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    # us = [np.zeros(12)]*solver.problem.T
    print('Initialization success !')
    solver.solve(xs, us, maxiter, is_feasible, init_reg)
    
    print(f'Initial cost: {solver.cost}')
    
    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])
    
    swing_count = [0] * len(contact_model.contact_ids)
    la_active = [False] * len(contact_model.contact_ids)
    data = pinocchio.Model.createData(state.pinocchio)
    
    for i in tqdm(range(steps)):
        
        q,v = traj[-1][:state.nq], traj[-1][state.nv:]
        pinocchio.forwardKinematics(state.pinocchio, data, q)
        pinocchio.updateFramePlacements(state.pinocchio, data)
        for k, contact_id in enumerate(contact_model.contact_ids):
            oMf = data.oMf[contact_id]
            height = oMf.translation[2]
            if height>air_eps:
                swing_count[k] += 1
            elif not la_active[k]:
                swing_count[k] = 0
            if swing_count[k]>=12:
                la_active[k] = True
                set_la_cost(costs_r, state, actuation, contact_id, True)
        
        actionmodels = IAM_shoot(N,state, actuation, [costs_r,costs_t], contact_model, DT, rho)
        
        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12))
        
        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverFDDP(problem)
        
        solver.solve(xi, ui, maxiter, is_feasible, init_reg)
        
        print(f'Step {i+1}, cost: {solver.cost}')
        
        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])
        
        # print(swing_count)
        
    return (traj,inp_traj)