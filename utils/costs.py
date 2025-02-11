import crocoddyl
from .transforms import *
import numpy as np
import pinocchio

alpha = 30
beta = 10

wq = np.array([20, 20, 80] + [10]*3 + [1]*12)
wv = wq/alpha
wx_r = np.concat([wq,wv])
# wx_r *= 10
wx_t = beta*wx_r
# wx_t[6:18] += 20
wu = 2e-4

wf = 1
wa = 2e3
ws = 1e-2

D=np.hstack([np.zeros((2,1)),np.eye(2)])
c_trot = np.array([[1,0,0,-1],
                   [0,1,-1,0]])
c_pace = np.array([[1,0,-1,0],
                   [0,1,0,-1]])
c_bounding = np.array([[1,-1,0,0],
                    [0,0,1,-1]])
C_trot = np.kron(c_trot,D)
C_pace = np.kron(c_pace,D)
C_bounding = np.kron(c_bounding,D)

# 太小会发散, 太大在空中会乱动
x_act_r = crocoddyl.ActivationModelWeightedQuad(2*wx_r**(1.7))
x_act_t = crocoddyl.ActivationModelWeightedQuad(2*wx_t**(1.7))

ls_act = crocoddyl.ActivationModelQuad(4)

wf_act = np.array([1,1,0,0,0,0])
wa_act = np.array([0,0,1])

u_bound = 50
wu_bound =1e6



class ControlSymmCostModel(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation, Csymm, nu):
        self.C = Csymm
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u=None):
        if u is None:
            data.cost = 0.0
        else:
            data.residual.r = self.C@u
            self.activation.calc(data.activation, data.residual.r)
            data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        if u is not None:
            self.activation.calcDiff(data.activation, data.residual.r)
            data.Lu[:] = self.C.T@data.activation.Ar
            data.Luu[:, :] = self.C.T@data.activation.Arr@self.C
            
    def createData(self, collector):
        data = crocoddyl.CostDataAbstract(self, collector)
        return data


def handstand(rmodel,state,actuation):
    lr = crocoddyl.CostModelSum(state, actuation.nu)
    lt = crocoddyl.CostModelSum(state, actuation.nu)
    
    x0 = rmodel.defaultState
    xtarget = x0.copy()
    quat = rpy_to_quaternion(np.array([0,0.9,0]))
    xtarget[2] += 0.1
    xtarget[3] = quat.x
    xtarget[4] = quat.y
    xtarget[5] = quat.z
    xtarget[6] = quat.w
    # xtarget[0] += 0.1
    

    lr_x = crocoddyl.CostModelResidual(
        state, x_act_r, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lt_x = crocoddyl.CostModelResidual(
        state, x_act_t, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lr_u = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    ls = ControlSymmCostModel(state,ls_act,C_bounding,actuation.nu)
    
    u_bound_act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-np.ones(actuation.nu)*u_bound,np.ones(actuation.nu)*u_bound)
    )
    lu_bound = crocoddyl.CostModelResidual(
        state, u_bound_act, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    
    lr.addCost("xGoal", lr_x, 2)
    lr.addCost("uReg", lr_u, 2*wu)
    lr.addCost('ls',ls,2*ws)
    # lr.addCost('lu_bound',lu_bound,wu_bound)
    
    lt.addCost("xGoal", lt_x, 2)
    return lr, lt

def upright(rmodel,state,actuation):
    lr = crocoddyl.CostModelSum(state, actuation.nu)
    lt = crocoddyl.CostModelSum(state, actuation.nu)
    
    x0 = rmodel.defaultState
    xtarget = x0.copy()
    quat = rpy_to_quaternion(np.array([0,-3.14/2,0]))
    xtarget[2] += 0.3
    xtarget[3] = quat.x
    xtarget[4] = quat.y
    xtarget[5] = quat.z
    xtarget[6] = quat.w
    xtarget[8] = 3.14/2
    xtarget[11] = 3.14/2
    xtarget[14] = 3.14/2
    xtarget[15] = 0
    xtarget[17] = 3.14/2
    xtarget[18] = 0
    

    lr_x = crocoddyl.CostModelResidual(
        state, x_act_r, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lt_x = crocoddyl.CostModelResidual(
        state, x_act_t, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lr_u = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    ls = ControlSymmCostModel(state,ls_act,C_bounding,actuation.nu)
    
    u_bound_act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-np.ones(actuation.nu)*u_bound,np.ones(actuation.nu)*u_bound)
    )
    lu_bound = crocoddyl.CostModelResidual(
        state, u_bound_act, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    
    lr.addCost("xGoal", lr_x, 2)
    lr.addCost("uReg", lr_u, 2*wu)
    lr.addCost('ls',ls,2*ws)
    lr.addCost('lu_bound',lu_bound,wu_bound)
    
    lt.addCost("xGoal", lt_x, 2)
    return lr, lt

def crouch(rmodel,state,actuation):
    lr = crocoddyl.CostModelSum(state, actuation.nu)
    lt = crocoddyl.CostModelSum(state, actuation.nu)
    
    x0 = rmodel.defaultState
    xtarget = x0.copy()
    quat = rpy_to_quaternion(np.array([0,0,0]))
    xtarget[2] -= 0.1
    xtarget[3] = quat.x
    xtarget[4] = quat.y
    xtarget[5] = quat.z
    xtarget[6] = quat.w
    
    lr_x = crocoddyl.CostModelResidual(
        state, x_act_r, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lt_x = crocoddyl.CostModelResidual(
        state, x_act_t, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lr_u = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    ls = ControlSymmCostModel(state,ls_act,C_bounding,actuation.nu)
    
    u_bound_act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-np.ones(actuation.nu)*u_bound,np.ones(actuation.nu)*u_bound)
    )
    lu_bound = crocoddyl.CostModelResidual(
        state, u_bound_act, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    
    lr.addCost("xGoal", lr_x, 2)
    lr.addCost("uReg", lr_u, 2*wu)
    lr.addCost('ls',ls,2*ws)
    lr.addCost('lu_bound',lu_bound,wu_bound)
    
    lt.addCost("xGoal", lt_x, 2)
    return lr, lt
  
def walking(rmodel,state,actuation):
    DT=2.5e-2
    
    lr = crocoddyl.CostModelSum(state, actuation.nu)
    lt = crocoddyl.CostModelSum(state, actuation.nu)
    
    x0 = rmodel.defaultState
    xdr = x0.copy()
    xdr[19] = 1
    xdr[2] += 0.05
    xdr[0] = 100*DT
    
    xdt = x0.copy()
    xdt[2] += 0.05
    xdt[0] = 100*DT
    
    lr_act = crocoddyl.ActivationModelWeightedQuad(
        2*np.array([80, 20, 20] + [1]*3 + [1]*12 + [0,0,0]+[0]*15)
    )
    lr_x = crocoddyl.CostModelResidual(
        state, lr_act, crocoddyl.ResidualModelState(state, xdr, actuation.nu)
    )
    lt_x = crocoddyl.CostModelResidual(
        state, x_act_r, crocoddyl.ResidualModelState(state, xdt, actuation.nu)
    )
    lr_u = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    ls = ControlSymmCostModel(state,ls_act,C_trot,actuation.nu)
    
    u_bound_act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-np.ones(actuation.nu)*u_bound,np.ones(actuation.nu)*u_bound)
    )
    lu_bound = crocoddyl.CostModelResidual(
        state, u_bound_act, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    
    lr.addCost("xGoal", lr_x, 2)
    # lr.addCost("uReg", lr_u, 2*wu)
    lr.addCost('ls',ls,2*ws)
    # lr.addCost('lu_bound',lu_bound,wu_bound)
    
    lt.addCost("xGoal", lt_x, 2)
    return lr, lr

def jumping(rmodel,state,actuation):
    lr = crocoddyl.CostModelSum(state, actuation.nu)
    lt = crocoddyl.CostModelSum(state, actuation.nu)
    
    x0 = rmodel.defaultState
    xtarget = x0.copy()
    xtarget[2] += 0.5

    lr_x = crocoddyl.CostModelResidual(
        state, x_act_r, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lt_x = crocoddyl.CostModelResidual(
        state, x_act_t, crocoddyl.ResidualModelState(state, xtarget, actuation.nu)
    )
    lr_u = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    ls = ControlSymmCostModel(state,ls_act,C_bounding,actuation.nu)
    
    u_bound_act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-np.ones(actuation.nu)*u_bound,np.ones(actuation.nu)*u_bound)
    )
    lu_bound = crocoddyl.CostModelResidual(
        state, u_bound_act, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    
    lr.addCost("xGoal", lr_x, 2)
    lr.addCost("uReg", lr_u, 2*wu*1e3) # 这一项大可以算的更快
    lr.addCost('ls',ls,2*ws)
    lr.addCost('lu_bound',lu_bound,wu_bound)
    
    lt.addCost("xGoal", lt_x, 2)
    return lr, lt


def add_lf_cost(costs, state, actuation, contact_ids, wf_act=wf_act):
    for i in contact_ids:
        lf_res = crocoddyl.ResidualModelFrameVelocity(
        state,i,pinocchio.Motion.Zero(),
        pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        actuation.nu
        )
        lf_act = crocoddyl.ActivationModelWeightedQuad(wf_act**2)
        lf = crocoddyl.CostModelResidual(
            state, lf_act, lf_res
        )
        costname = f'lf_{i}'
        if not costname in (costs.active_set.toset() | costs.inactive_set.toset()):
            costs.addCost(f'lf_{i}',lf,2*wf)
        else:
            costs.removeCost(costname)
            costs.addCost(costname,lf,2*wf)
        
def set_lf_cost(costs, state, actuation, contact_id, height):
    i=contact_id
    costname = f'lf_{i}'
    if not costname in (costs.active_set.toset() | costs.inactive_set.toset()):
        return
    
    # wf_act_height = wf_act/(1+np.exp(30*height))
    wf_act_height = wf_act*0.5*(1+np.tanh(-15*height))
    lf_res = crocoddyl.ResidualModelFrameVelocity(
    state,i,pinocchio.Motion.Zero(),
    pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    actuation.nu
    )
    lf_act = crocoddyl.ActivationModelWeightedQuad(wf_act_height**2)
    lf = crocoddyl.CostModelResidual(
        state, lf_act, lf_res
    )
    costs.removeCost(costname)
    costs.addCost(costname,lf,2*wf)
          
def set_la_cost(costs, state, actuation, contact_id, active=True):
    i=contact_id
    costname = f'la_{i}'
    
    if not active:
        if costname in (costs.active_set.toset() | costs.inactive_set.toset()):
            costs.removeCost(costname)
        return
        
    la_res = crocoddyl.ResidualModelFrameTranslation(
        state,i,np.zeros(3),actuation.nu
    )
    la_act = crocoddyl.ActivationModelWeightedQuad(wa_act**2)
    la = crocoddyl.CostModelResidual(
        state, la_act, la_res
    )
    
    if not costname in (costs.active_set.toset() | costs.inactive_set.toset()):
        costs.addCost(costname,la,2*wa)
    else:
        costs.removeCost(costname)
        costs.addCost(costname,la,2*wa)
    
    