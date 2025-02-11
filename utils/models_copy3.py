import numpy as np
import pinocchio
import crocoddyl

import coal
from coal import CollisionObject, DynamicAABBTreeCollisionManager as CollisionManager
ground_width = 100.0
ground_height = 100.0
ground_depth = 50
ground_geo = coal.Box(ground_width, ground_height, ground_depth)
ground_pos = np.array([0., 0., -ground_depth/2])
ground_transform = coal.Transform3s(np.eye(3), ground_pos)
ground_obj = CollisionObject(ground_geo, ground_transform)

import pinocchio.casadi as cpin
from casadi import SX, Function, jacobian, vertcat, MX

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds
from scipy.linalg import block_diag

from line_profiler import profile

from .transforms import *

class DAD_contact(crocoddyl.DifferentialActionDataAbstract):
    @profile
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        
        self.cmodel = cpin.Model(model.state.pinocchio)
        self.cdata = self.cmodel.createData()
        self.qsym = SX.sym('q', model.state.pinocchio.nq-1) # cpin只允许SX
        # self.vsym = SX.sym('v', model.state.nv)
        # self.tausym = SX.sym('tau', model.state.nv)
        # self.varsym = SX.sym('qvtau', 3*model.state.nv)
        
        self.contact_ids = model.contact_ids
        
        
class DAM_contact(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel, contact_ids, dt, rho=0):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuationModel.nu, costModel.nr
        )
        self.actuation = actuationModel
        self.costs = costModel
        
        self.contact_ids = contact_ids
        self.contact_radius = 0.02  # 脚部半径
        geo = coal.Sphere(self.contact_radius)
        trans = coal.Transform3s(np.eye(3), np.zeros(3))
        self.contact_objects = [CollisionObject(geo, trans) for i in range(len(contact_ids))]
        self.dt = dt
        self.control_bound = 100
        self.M_reg = 0e-3
        self.friction = 0.8
        self.rho = rho
        self.contact_maxiter = 20
        
        self.collision_time = 0
        self.differentiation_time = 0
        self.count = 0
    
    @profile
    def collision_test(self,data,tau,v):
        for i, contact_id in enumerate(self.contact_ids):
            oMf = data.pinocchio.oMf[contact_id]
            pos = oMf.translation
            rot = oMf.rotation
            self.contact_objects[i].setTransform(coal.Transform3s(rot, pos))
        collision_ids = []
        normal_trans = []
        height = []
        for i in range(len(self.contact_objects)):
            result = coal.CollisionResult() # 必须清空...
            req = coal.CollisionRequest()   
            if coal.collide(self.contact_objects[i], ground_obj, req, result):
                contact = result.getContacts()[0]
                collision_ids.append(self.contact_ids[i])
                # current = data.pinocchio.oMf[self.contact_ids[i]].rotation[:,2]/np.linalg.norm(data.pinocchio.oMf[self.contact_ids[i]].rotation)
                current = np.array([0,0,1])
                target = -contact.normal
                normal_trans.append(pinocchio.Quaternion.FromTwoVectors(current,target).matrix())
                
                # height.append(contact.pos[2])
                height.append(data.pinocchio.oMf[self.contact_ids[i]].translation[2])
                
        height = np.array(height)
        
        if collision_ids == []:
            return False
            
        col_num = len(collision_ids)
        Jk = [
            # normal_trans[i]@
            pinocchio.getFrameJacobian(
                    self.state.pinocchio,
                    data.pinocchio,
                    collision_ids[i],
                    pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    # pinocchio.ReferenceFrame.LOCAL
            )[:3,:] for i in range(col_num)
        ]
        
        # Mtest=data.pinocchio.M
        # if np.linalg.inv(Mtest).max() > 70:
        #     data.pinocchio.M += self.M_reg*np.eye(data.pinocchio.M.shape[0])
        data.M = data.pinocchio.M
        # data.Minv = np.linalg.inv(data.M)
        data.Minv = data.pinocchio.Minv
    
        
        ddqf = np.dot(data.Minv, ( tau - data.pinocchio.nle ))
        
        Ak = [Jk[i]@data.Minv@Jk[i].T for i in range(col_num)]
        Mk = [np.linalg.inv(Ak[i]) for i in range(col_num)]
        
        J = np.vstack(Jk)
        
        iter = 0
        # repeat = 1
        # converge = np.array([False]*col_num*repeat)
        impulse_last = [np.array([1,1,1])*1]*col_num
        impulse = [np.array([1,1,1])*0]*col_num
        bound = Bounds(lb=np.array([-np.inf,-np.inf,0]))
        cone = lambda force: self.friction*force[2]-np.sqrt(force[0]**2+force[1]**2)
        cases = [None] * col_num
        # inits = [1,10,0]
        # methods = ['SLSQP','trust-constr'] # trust-constr多一位数时间
        opts = {'maxiter':50}
        while (not np.allclose(impulse,impulse_last)) and iter < self.contact_maxiter:
            impulse_last = impulse.copy()
            
            for i in range(col_num):
                contact_cases = ['sep','clamp','slide']
                cases[i] = {case:[np.array([0,0,0]),0] for case in contact_cases}
                ck = Jk[i]@(ddqf*self.dt+v)
                for j in range(col_num):
                    if j==i:
                        continue
                    ck += Jk[i]@data.Minv@Jk[j].T@impulse_last[j]
                contact_vk = lambda force: ck + Ak[i] @ force
                jack = lambda force: 2*Ak[i].T@Mk[i]@contact_vk(force)
                objk = lambda force: contact_vk(force).T @ Mk[i] @ contact_vk(force)
                cases[i]['sep'][1] = objk(cases[i]['sep'][0])
                cases[i]['clamp'][0] = Mk[i]@(np.array([0,0,-height[i]/self.dt])-ck)
                if cone(cases[i]['clamp'][0])<0:
                    cases[i]['clamp'][1] = np.inf
                else:
                    cases[i]['clamp'][1] = objk(cases[i]['clamp'][0])
                
                cons = (LinearConstraint(Ak[i][2,:],lb=-ck[2]-height[i]/self.dt,ub=-ck[2]-height[i]/self.dt),# -height[i]/self.dt),
                    NonlinearConstraint(cone,lb=0,ub=0),
                    NonlinearConstraint(lambda force: contact_vk(force)[0]*force[1]-contact_vk(force)[1]*force[0],lb=0,ub=0))
                # impulsek_init = -Mk[i]@ck
                result = minimize(objk,impulse_last[i],constraints=cons,jac=jack,bounds=bound,method='SLSQP',options=opts)
                # result = minimize(objk,impulsek_init,constraints=cons,bounds=bound,method='SLSQP',options=opts)
                cases[i]['slide'][0] = result.x
                cases[i]['slide'][1] = objk(cases[i]['slide'][0])
                
                _, (impulse[i], _) = min(cases[i].items(), key=lambda item: item[1][1])
                
            iter += 1
        if iter == self.contact_maxiter:
            raise 'contact dynamics fails'
            pass
        impulse = np.hstack(impulse)
        
        # classify
        slide_ids = []
        Es = []
        clamping_ids = []
        for i in range(col_num):
            if np.allclose(impulse[3*i+2],0): # separate
                continue
            if np.allclose(self.friction*impulse[3*i+2]-np.sqrt(impulse[3*i]**2+impulse[3*i+1]**2),0):
                slide_ids.append(i)
                Es.append(np.array([impulse[3*i]/impulse[3*i+2],impulse[3*i+1]/impulse[3*i+2],1])[:,np.newaxis])
            else:
                clamping_ids.append(i)
        
        J_ = []
        contact_impulse = []
        Jleft = []
        Jright = []
        h = []
        if clamping_ids != []:
            Jc = np.vstack([J[(3*i):(3*i+3),:] for i in clamping_ids])
            impulsec = np.hstack([impulse[(3*i):(3*i+3)] for i in clamping_ids])
            Jleft.append(Jc)
            Jright.append(Jc)
            J_.append(Jc)
            contact_impulse.append(impulsec)
            h.append(np.hstack( [np.array([0,0,height[i]]) for i in clamping_ids] ))
            
            data.Jc = Jc
            data.impulsec = impulsec
        if slide_ids != []:
            Es = block_diag(*Es)
            data.Es = Es
            Jsn = np.vstack([J[(3*i+2):(3*i+3),:] for i in slide_ids])
            Jst = np.vstack([J[(3*i):(3*i+2),:] for i in slide_ids])
            Js = np.vstack([J[(3*i):(3*i+3),:] for i in slide_ids])
            impulsesn = np.array([impulse[(3*i+2)] for i in slide_ids])
            impulsest = np.hstack([impulse[(3*i):(3*i+2)] for i in slide_ids])
            Jleft.append(Jsn)
            Jright.append(Es.T@Js)
            J_.append(Jsn)
            J_.append(Jst)
            contact_impulse += [impulsesn.flatten(),impulsest] # 要和J_对齐
            
            h.append(np.hstack( [np.array([height[i]]) for i in slide_ids] ))
            
            data.Jsn = Jsn
            data.Jst = Jst
            data.impulsesn = impulsesn
            data.impulsest = impulsest
        data.h = np.hstack(h)
        
        if slide_ids+clamping_ids == []:
            return False
        
        Jleft = np.vstack(Jleft)        
        Jright = np.vstack(Jright)
        
        A = Jleft@data.Minv@Jright.T
        b = Jleft@(ddqf*self.dt+v)
        
        data.contactJleft = Jleft
        data.contactJright = Jright
        data.contactJ = np.vstack(J)
        data.impulse = np.hstack(contact_impulse)
        
        # Ainv = np.linalg.inv(A)
        D = data.impulse.copy()
        if slide_ids == []:
            D[::3] = 1
            D[1::3] = 1
            D = self.rho/D**2
            D[::3] = 0
            D[1::3] = 0
        else:
            snum = len(slide_ids)
            D = D[:(-2*snum)]
            D[:(-snum):3] = 1
            D[1:(-snum):3] = 1
            D = self.rho/D**2
            D[:(-snum):3] = 0
            D[1:(-snum):3] = 0
        Ainv = np.linalg.inv(A + np.diag(D))
        
        data.slide_ids = slide_ids
        data.clamping_ids = clamping_ids
        data.collision_ids = collision_ids
        data.contactAinv = Ainv
        data.contactb = b
        data.contactpreb = (ddqf*self.dt+v)
        data.effect = data.contactJ.T@data.impulse/self.dt
        return True
    @profile
    def calc(self, data, x, u=None):
        if u is None:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            if v[2] < -q[2]/self.dt:
                v[2] = -q[2]/self.dt

            # u = np.clip(u,-self.control_bound*np.ones(self.actuation.nu),self.control_bound*np.ones(self.actuation.nu))
            self.actuation.calc(data.actuation, x, u)
            tau = data.actuation.tau
                        
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.computeMinverse(self.state.pinocchio, data.pinocchio, q)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
            
            collision = self.collision_test(data, tau, v)
            
            # Computing the dynamics using ABA
            if collision:
                data.xout[:] = pinocchio.aba(
                self.state.pinocchio, data.pinocchio, q, v, tau+data.effect
                )
            else:
                data.xout[:] = pinocchio.aba(
                self.state.pinocchio, data.pinocchio, q, v, tau+data.effect
                )
            
            # Computing the cost value and residuals
            pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
            
            self.costs.calc(data.costs, x, u)
            data.cost = data.costs.cost
    # @profile
    def calcDiff(self, data, x, u=None):
        if u is None:
            self.costs.calcDiff(data.costs, x)
        else:
        
            # u = np.clip(u,-self.control_bound*np.ones(self.actuation.nu),self.control_bound*np.ones(self.actuation.nu))
            nq, nv = self.state.nq, self.state.nv
            q, v = x[:nq], x[-nv:]
            # Computing the actuation derivatives
            self.actuation.calcDiff(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics derivatives
            collision_ids = data.collision_ids
            if collision_ids == [] or (data.slide_ids+data.clamping_ids == []):
                # Computing the cost derivatives
                pinocchio.computeABADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
                ddq_dq = data.pinocchio.ddq_dq
                ddq_dv = data.pinocchio.ddq_dv
                # data.pinocchio.Minv = data.Minv
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + np.dot(
                    data.Minv, data.actuation.dtau_dx
                )
                data.Fu[:, :] = np.dot(data.Minv, data.actuation.dtau_du)
                self.costs.calcDiff(data.costs, x, u)
                return
            
            pinocchio.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, tau+data.effect
            )
            ddq_dq = data.pinocchio.ddq_dq
            ddq_dv = data.pinocchio.ddq_dv
            # data.pinocchio.Minv = data.Minv
            # data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + np.dot(
            #     data.Minv, data.actuation.dtau_dx
            # )
            data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + data.Minv@data.actuation.dtau_dx
            data.Fu[:, :] = data.Minv@data.actuation.dtau_du
            # data.Fu[:,:] = np.zeros_like(data.Fu)
            
            # for db_dq db_dv
            pinocchio.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, tau
            )
            ddq_dq = data.pinocchio.ddq_dq + data.Minv@data.actuation.dtau_dx[:,:nv]
            ddq_dv = data.pinocchio.ddq_dv + data.Minv@data.actuation.dtau_dx[:,-nv:]
            
            qrpy = quat_to_rpy(q[3:7])
            qrpy = np.concat([q[:3],qrpy,q[7:]],axis=0)
            cmodel, cdata = data.cmodel, data.cdata
            contactJleft = data.contactJleft
            contactJright = data.contactJright
            Minv = data.Minv
            contactAinv = data.contactAinv
            contactJ = data.contactJ
            impulse = data.impulse
            qsym = data.qsym
            
            slide_ids = data.slide_ids
            clamping_ids = data.clamping_ids
            
            dJc_dq = []
            dJsn_dq = []
            dJst_dq = []
            dJsright_dq = []
            dh_dq = []
            def Jacobians_h_Minv(q_sym):
                qquat = csrpy_to_quat(q_sym[3:6])
                inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
                cpin.forwardKinematics(cmodel,cdata,inp_q) # 这行必须
                cpin.computeJointJacobians(cmodel,cdata,inp_q)
                cpin.computeMinverse(cmodel, cdata, inp_q)
                J = []
                h = []
                for fid in clamping_ids+slide_ids:
                    Ji = cpin.getFrameJacobian(
                        cmodel, cdata, collision_ids[fid], cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    )[:3,:]
                    hi = cdata.oMf[collision_ids[fid]].translation[2]
                    
                    J.append(Ji)
                    if fid in clamping_ids:
                        h.append(vertcat(SX(0),SX(0),hi))
                    else:
                        h.append(hi)
                J = vertcat(*J)
                h = vertcat(*h)
                return J, h, cdata.Minv
            frameJsym, hsym, Minvsym = Jacobians_h_Minv(qsym)
            
            frameJ_dq = jacobian(frameJsym, qsym)                
            frameJ_dq_fun = Function('jacobian', [qsym], [frameJ_dq])
            dJ_dq_ = np.array(frameJ_dq_fun(qrpy)).reshape((nv,3*len(clamping_ids+slide_ids),-1)).transpose((2,1,0))
            
            height_dq = jacobian(hsym, qsym)
            dh_dq_fun = Function('height_dq', [qsym], [height_dq])
            dh_dq = np.array(dh_dq_fun(qrpy))
            
            dMinv_dq = jacobian(Minvsym,qsym)
            dMinv_dq_fun = Function('jacobian', [qsym], [dMinv_dq])
            dMinv_dq = np.array(dMinv_dq_fun(qrpy)).reshape((nv,nv,-1)).transpose((2,1,0))
                
            for fid in clamping_ids+slide_ids:
                i = (clamping_ids+slide_ids).index(fid)
                dJ_dqi = dJ_dq_[:,(3*i):(3*i+3),:]
                               
                if fid in clamping_ids:
                    dJc_dq.append(dJ_dqi)
                else:
                    k = slide_ids.index(fid)
                    dJsn_dq.append(dJ_dqi[:,2:3,:])
                    dJst_dq.append(dJ_dqi[:,:2,:])
                    dJsright_dq.append(
                        data.Es[(3*k):(3*k+3),k:(k+1)].T[np.newaxis,:] @ dJ_dqi
                    )
                
            dJ_dq_left = np.concat(dJc_dq+dJsn_dq,axis=1)
            dJ_dq_right = np.concat(dJc_dq+dJsright_dq,axis=1)
            dJ_dq = np.concat(dJc_dq+dJsn_dq+dJst_dq,axis=1)
            
            dA_dq = dJ_dq_left@((Minv@contactJright.T)[np.newaxis,:]) + (contactJleft@Minv)[np.newaxis,:]@dJ_dq_right.transpose((0,2,1)) + (contactJleft[np.newaxis,:])@dMinv_dq@(contactJright.T[np.newaxis,:])
            
            db_dq = (dJ_dq_left@(data.contactpreb[np.newaxis,:,np.newaxis])).squeeze().T + contactJleft@ddq_dq*self.dt
            db_dv = contactJleft@(ddq_dv*self.dt+np.eye(nv))
            db_dtau = contactJleft@Minv*self.dt
            
            if slide_ids == []:
                dlambda_dq = -contactAinv@( (dA_dq@impulse[np.newaxis,:,np.newaxis]).squeeze().T )
            else:
                dlambda_dq = -contactAinv@( (dA_dq@impulse[np.newaxis,:-(2*len(slide_ids)),np.newaxis]).squeeze().T )
            dlambda_dq += -contactAinv@(db_dq+dh_dq/self.dt)
                
            dlambda_dv = -contactAinv@db_dv
            dlambda_dtau = -contactAinv@db_dtau
            if slide_ids != []:
                Es = data.Es
                snum = len(slide_ids)
                Est = np.concat([Es[(3*i):(3*i+2),:] for i in range(snum)])
                dlambda_dq = np.vstack([dlambda_dq, Est@dlambda_dq[-snum:,:]])
                dlambda_dv = np.vstack([dlambda_dv, Est@dlambda_dv[-snum:,:]])
                dlambda_dtau = np.vstack([dlambda_dtau, Est@dlambda_dtau[-snum:,:]])
            
            Fq = Minv@ ( (dJ_dq.transpose(0,2,1)@(impulse[np.newaxis,:,np.newaxis])).squeeze().T + contactJ.T@dlambda_dq )/self.dt
            Fv = Minv@contactJ.T@dlambda_dv/self.dt
            Ftau = Minv@contactJ.T@dlambda_dtau/self.dt
            
            data.Fx[:, :] += np.hstack([Fq, Fv])
            data.Fu[:, :] += Ftau@data.actuation.dtau_du
            
            self.costs.calcDiff(data.costs, x, u)


    def createData(self):
        data = DAD_contact(self)
        return data
    
def IAM_contact(N, state, actuation, costs, contact_ids, DT):
    assert N>1
    dmodel = DAM_contact(
        state, actuation, costs, contact_ids, DT
    )
    dmodels = [dmodel] * N
    actionmodels = [
            crocoddyl.IntegratedActionModelEuler(
                m, DT
            ) for m in dmodels
        ]
    return actionmodels

def IAM_ccd(N, state, actuation, costs, contact, DT):
    assert N>1
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, costs, contact
    )
    dmodels = [dmodel] * N
    actionmodels = [
            crocoddyl.IntegratedActionModelEuler(
                m, DT
            ) for m in dmodels
        ]
    return actionmodels

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    WITHDISPLAY = "display"

    import example_robot_data
    # Loading the anymal model
    a1 = example_robot_data.load("a1")
    rmodel = a1.model

    lims = rmodel.effortLimit
    lims *= 0.5  # reduced artificially the torque limits
    rmodel.effortLimit = lims

    lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"
    lfCalf, rfCalf, lhCalf, rhCalf = "FL_calf", "FR_calf", "RL_calf", "RR_calf"
    body = "trunk"

    rdata = rmodel.createData()
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    lf_id = rmodel.getFrameId(lfFoot)
    rf_id = rmodel.getFrameId(rfFoot)
    lh_id = rmodel.getFrameId(lhFoot)
    rh_id = rmodel.getFrameId(rhFoot)
    body_id = rmodel.getFrameId(body)
    lfcalf_id = rmodel.getFrameId(lfCalf)
    rfcalf_id = rmodel.getFrameId(rfCalf)
    lhcalf_id = rmodel.getFrameId(lhCalf)
    rhcalf_id = rmodel.getFrameId(rhCalf)

    _integrator = 'euler'
    _control = 'one'

    q0 = rmodel.referenceConfigurations["standing"]
    rmodel.defaultState = np.concatenate([q0, np.zeros(rmodel.nv)])
    x0 = rmodel.defaultState
    u0 = np.zeros(actuation.nu)+1
    
    
    from costs import handstand, crouch
    costs = handstand(rmodel, state, actuation)

    DT = 0.25
    contact_ids = [lf_id, rf_id, lh_id, rh_id,
                ]
    DAM = DAM_contact(state, actuation,costs,contact_ids,DT)
    x0 = rmodel.defaultState
    x0[8] = -3.14/2
    u0 = np.ones(12)*0.1
    u0 = np.ones(12)*0.1
    DAD = DAM.createData()
    DAM.calc(DAD,x0,u0)
    DAM.calcDiff(DAD,x0,u0)