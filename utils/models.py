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
# from scipy import sparse

from line_profiler import profile

from .transforms import *
from .costs import set_lf_cost

class ContactModel:
    def __init__(self, state, contact_ids):
        self.contact_ids = contact_ids
        self.cmodel = cpin.Model(state.pinocchio)
        self.cdata = self.cmodel.createData()
        self.qsym = SX.sym('q', state.pinocchio.nq-1) # cpin只允许SX
        # self.vsym = SX.sym('v', model.state.nv)
        # self.tausym = SX.sym('tau', model.state.nv)
        # self.varsym = SX.sym('qvtau', 3*model.state.nv)
        
        self.contact_ids = contact_ids
        
        # data在每个N初始化, ddpiter增加不会再创建
        self.auto_diff()
        
    @profile    
    def auto_diff(self):
        cmodel, cdata = self.cmodel, self.cdata
        qsym = self.qsym
        self.dJ_dq_funs = []
        self.dh_dq_funs = []
        
        for i in self.contact_ids:
            def frameJacobian(q_sym):
                qquat = csrpy_to_quat(q_sym[3:6])
                inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
                cpin.forwardKinematics(cmodel,cdata,inp_q)
                cpin.computeJointJacobians(cmodel,cdata,inp_q)
                J = cpin.getFrameJacobian(
                    cmodel, cdata, i, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                return J[:3,:]
            frameJsym = frameJacobian(qsym)
            frameJ_dq = jacobian(frameJsym, qsym)
            frameJ_dq_fun = Function('jacobian', [qsym], [frameJ_dq])
            self.dJ_dq_funs.append(frameJ_dq_fun)
            
            def frameheight(q_sym):
                qquat = csrpy_to_quat(q_sym[3:6])
                inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
                cpin.forwardKinematics(cmodel,cdata,inp_q)
                return cdata.oMf[i].translation[2]
            heightsym = frameheight(qsym)
            height_dq = jacobian(heightsym, qsym)
            dh_dq_fun = Function('height_dq', [qsym], [height_dq])
            self.dh_dq_funs.append(dh_dq_fun)
            
        def Minvfun(q_sym):
            qquat = csrpy_to_quat(q_sym[3:6])
            inp_q = vertcat(q_sym[:3],qquat,q_sym[6:])
            cpin.computeMinverse(cmodel, cdata, inp_q)
            return cdata.Minv
        Minvfunsym = Minvfun(qsym)
        dMinv_dq = jacobian(Minvfunsym,qsym)        
        self.dMinv_dq_fun = Function('jacobian', [qsym], [dMinv_dq])
 

class DAD_contact(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
       
        
class DAM_contact(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel, contactModel, dt, rho=0, friction=0.8):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuationModel.nu, costModel.nr
        )
        self.actuation = actuationModel
        self.costs = costModel
        self.contact = contactModel
        
        self.contact_ids = contactModel.contact_ids
        self.contact_radius = 0.02  # 脚部半径
        geo = coal.Sphere(self.contact_radius)
        trans = coal.Transform3s(np.eye(3), np.zeros(3))
        self.contact_objects = [CollisionObject(geo, trans) for i in range(len(self.contact_ids))]
        self.dt = dt
        self.control_bound = 100
        self.M_reg = 0e-3
        self.friction = friction
        self.rho = rho
        
        self.contact_maxiter = 1
        self.contact_eps = 1e-5
        self.contact_energyeps = 10
        
    @profile
    def collision_test(self,data,tau,v):
        
        for i, contact_id in enumerate(self.contact_ids):
            oMf = data.pinocchio.oMf[contact_id]
            pos = oMf.translation
            rot = oMf.rotation
            self.contact_objects[i].setTransform(coal.Transform3s(rot, pos))
            
            set_lf_cost(self.costs, self.state, self.actuation, contact_id ,pos[2])
            
        collision_ids = []
        normal_trans = []
        height = []
        for i in range(len(self.contact_objects)):
            result = coal.CollisionResult() # 必须清空...
            req = coal.CollisionRequest()   
            if coal.collide(self.contact_objects[i], ground_obj, req, result):
                # contact = result.getContacts()[0]
                collision_ids.append(self.contact_ids[i])
                
                # current = np.array([0,0,1])
                # target = -contact.normal
                # normal_trans.append(pinocchio.Quaternion.FromTwoVectors(current,target).matrix())
                
                # height.append(contact.pos[2])
                height.append(data.pinocchio.oMf[self.contact_ids[i]].translation[2])
                
        height = np.array(height)
        data.collision_ids = collision_ids
        
        if collision_ids == []:
            data.real_collision = False
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
        
        Aij = [[ Jk[i]@data.Minv@Jk[j].T for j in range(col_num)] for i in range(col_num)]
        Mk = [np.linalg.inv(Aij[i][i]) for i in range(col_num)]
        
        J = np.vstack(Jk)
        
        iter = 0
        impulse_last = [np.array([1,1,1])*1]*col_num
        impulse = [np.array([1,1,1])*0]*col_num
        bound = ((None,None),(None,None),(0,None))
        cone = lambda force: (self.friction*force[2])**2-force[0]**2-force[1]**2
        cases = [None] * col_num
        cases_result = [None] * col_num
        contact_cases = ['sep','clamp','slide']
        cases_init = {case:[np.array([0,0,0]),0] for case in contact_cases}
        # methods = ['SLSQP','trust-constr'] # trust-constr多一位数时间
        opts = {'maxiter':1}
        impulse_eps = 1
        while impulse_eps > self.contact_eps and iter < self.contact_maxiter:
            impulse_last = impulse.copy()
            
            for i in range(col_num):
                cases[i] = cases_init.copy()
                ck = Jk[i]@(ddqf*self.dt+v)
                for j in range(col_num):
                    if j==i:
                        continue
                    ck += Aij[i][j]@impulse_last[j]
                
                cases[i]['clamp'][0] = Mk[i]@(np.array([0,0,-height[i]/self.dt])-ck)
                if cone(cases[i]['clamp'][0])<0:
                    cases[i]['clamp'][1] = np.inf
                else:
                    # cases[i]['clamp'][1] = objk(cases[i]['clamp'][0])
                    impulse[i] = cases[i]['clamp'][0]
                    cases_result[i] = 'clamp'
                    continue
                
                contact_vk = lambda force: ck + Aij[i][i] @ force
                jack = lambda force: 2*Aij[i][i].T@Mk[i]@contact_vk(force)
                objk = lambda force: np.sqrt( contact_vk(force).T @ Mk[i] @ contact_vk(force) )
                cases[i]['sep'][1] = objk(cases[i]['sep'][0])
                
                cons = (LinearConstraint(Aij[i][i][2,:],lb=-ck[2]-height[i]/self.dt,ub=-ck[2]-height[i]/self.dt),
                    NonlinearConstraint(cone,lb=0,ub=0),
                    NonlinearConstraint(lambda force: contact_vk(force)[0]*force[1]-contact_vk(force)[1]*force[0],lb=0,ub=0))
                result = minimize(objk,impulse_last[i],constraints=cons,jac=jack,bounds=bound,method='SLSQP',options=opts)
                cases[i]['slide'][0] = result.x
                cases[i]['slide'][1] = objk(cases[i]['slide'][0])
                
                cases_result[i], (impulse[i], _) = min(cases[i].items(), key=lambda item: item[1][1])
                
                if impulse[i][2] < self.contact_eps: # 数值误差，其实就是separate
                    cases_result[i], impulse[i] = 'sep', np.array([0,0,0])
                
            impulse_eps = np.sum([np.abs(np.array(impulse)-np.array(impulse_last))])
            iter += 1
        if iter == self.contact_maxiter:
            # raise 'contact dynamics fails'
            # 问题不大，硬算
            pass
        # impulse = np.hstack(impulse)
        impulse = np.concat(impulse)
        
        # classify
        slide_ids = []
        Es = []
        clamping_ids = []
        for i in range(col_num):
            if cases_result[i]=='sep': # separate
                continue
            if cases_result[i]=='slide':
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
            mask = sum([list(range(3*i, 3*i + 3)) for i in clamping_ids],[])
            Jc = J[mask,:]
            # Jc = np.vstack([J[(3*i):(3*i+3),:] for i in clamping_ids])
            impulsec = impulse[mask]
            # impulsec = np.hstack([impulse[(3*i):(3*i+3)] for i in clamping_ids])
            Jleft.append(Jc)
            Jright.append(Jc)
            J_.append(Jc)
            contact_impulse.append(impulsec)
            # h.append(np.hstack( [np.array([0,0,height[i]]) for i in clamping_ids] ))
            h += [np.array([0,0,height[i]]) for i in clamping_ids]
            
            data.Jc = Jc
            data.impulsec = impulsec
        if slide_ids != []:
            mask1 = sum([list(range(3*i+2, 3*i + 3)) for i in slide_ids],[])
            mask2 = sum([list(range(3*i, 3*i + 2)) for i in slide_ids],[])
            mask3 = sum([list(range(3*i, 3*i + 3)) for i in slide_ids],[])
            Es = block_diag(*Es)
            data.Es = Es
            Jsn = J[mask1,:]
            Jst = J[mask2,:]
            Js = J[mask3,:]
            impulsesn = impulse[mask1]
            impulsest = impulse[mask2]
            # Jsn = np.vstack([J[(3*i+2):(3*i+3),:] for i in slide_ids])
            # Jst = np.vstack([J[(3*i):(3*i+2),:] for i in slide_ids])
            # Js = np.vstack([J[(3*i):(3*i+3),:] for i in slide_ids])
            # impulsesn = np.array([impulse[(3*i+2)] for i in slide_ids])
            # impulsest = np.hstack([impulse[(3*i):(3*i+2)] for i in slide_ids])
            Jleft.append(Jsn)
            Jright.append(Es.T@Js)
            J_.append(Jsn)
            J_.append(Jst)
            contact_impulse += [impulsesn.flatten(),impulsest] # 要和J_对齐
            
            # h.append(np.hstack( [np.array([height[i]]) for i in slide_ids] ))
            h += [np.array([height[i]]) for i in slide_ids]
            
            data.Jsn = Jsn
            data.Jst = Jst
            data.impulsesn = impulsesn
            data.impulsest = impulsest
        
        if slide_ids+clamping_ids == []:
            data.real_collision = False
            return False
        
        # data.h = np.hstack(h)
        data.h = np.concat(h)
        Jleft = np.vstack(Jleft)        
        Jright = np.vstack(Jright)
        
        A = Jleft@data.Minv@Jright.T
        b = Jleft@(ddqf*self.dt+v)
        
        data.contactJleft = Jleft
        data.contactJright = Jright
        data.contactJ = np.vstack(J_)
        # data.impulse = np.hstack(contact_impulse)
        data.impulse = np.concat(contact_impulse)
        
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
        data.contactAinv = Ainv
        data.contactb = b
        data.contactpreb = (ddqf*self.dt+v)
        data.effect = data.contactJ.T@data.impulse/self.dt
        data.real_collision = True
        return True
    @profile
    def calc(self, data, x, u=None):
        if u is None: # 最后那一步默认u=None
            q, v = x[: self.state.nq], x[-self.state.nv :]
            if v[2] < -q[2]/self.dt:
                v[2] = -q[2]/self.dt
                
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
                self.state.pinocchio, data.pinocchio, q, v, tau
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
            if not data.real_collision:
                # Computing the cost derivatives
                pinocchio.computeABADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
                ddq_dq = data.pinocchio.ddq_dq
                ddq_dv = data.pinocchio.ddq_dv
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + data.pinocchio.Minv@data.actuation.dtau_dx
                data.Fu[:, :] = data.pinocchio.Minv@data.actuation.dtau_du
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
            contactJleft = data.contactJleft
            contactJright = data.contactJright
            Minv = data.Minv
            contactAinv = data.contactAinv
            contactJ = data.contactJ
            impulse = data.impulse
            collision_ids = data.collision_ids
            slide_ids = data.slide_ids
            clamping_ids = data.clamping_ids
            
            dJc_dq = []
            dJsn_dq = []
            dJst_dq = []
            dJsright_dq = []
            dh_dq = []
            for fid in clamping_ids+slide_ids:
                i = self.contact_ids.index(collision_ids[fid])
                frameJ_dq_fun = self.contact.dJ_dq_funs[i]
                # dJ_dqi = np.array(frameJ_dq_fun(qrpy)).reshape((nv,3,-1)).transpose((2,1,0))
                # dJ_dqi = np.array(frameJ_dq_fun.call([qrpy])[0]).reshape((nv,3,-1)).transpose((2,1,0))
                dJ_dqi = (frameJ_dq_fun.call([qrpy])[0]).full().reshape((nv,3,-1)).transpose((2,1,0))
                
                dh_dq_fun = self.contact.dh_dq_funs[i]
                # dh_dq_i = np.array(dh_dq_fun(qrpy)).reshape((1,-1))
                dh_dq_i = (dh_dq_fun.call([qrpy])[0]).full().reshape((1,-1))
                
                if fid in clamping_ids:
                    dJc_dq.append(dJ_dqi)
                    dh_dq_zero = np.zeros_like(dh_dq_i)
                    dh_dq.append(np.concat([dh_dq_zero,dh_dq_zero,dh_dq_i],axis=0))
                else:
                    k = slide_ids.index(fid)
                    dJsn_dq.append(dJ_dqi[:,2:3,:])
                    dJst_dq.append(dJ_dqi[:,:2,:])
                    dJsright_dq.append(
                        data.Es[(3*k):(3*k+3),k:(k+1)].T[np.newaxis,:] @ dJ_dqi
                    )
                    dh_dq.append(dh_dq_i)
                
            dJ_dq_left = np.concat(dJc_dq+dJsn_dq,axis=1)
            dJ_dq_right = np.concat(dJc_dq+dJsright_dq,axis=1)
            dJ_dq = np.concat(dJc_dq+dJsn_dq+dJst_dq,axis=1)
            dh_dq = np.concat(dh_dq,axis=0)
            
            dMinv_dq_fun = self.contact.dMinv_dq_fun
            # dMinv_dq = np.array(dMinv_dq_fun(qrpy)).reshape((nv,nv,-1)).transpose((2,1,0))
            dMinv_dq = (dMinv_dq_fun.call([qrpy])[0]).full().reshape((nv,nv,-1)).transpose((2,1,0))
            
            dA_dq = dJ_dq_left@((Minv@contactJright.T)[np.newaxis,:]) + (contactJleft@Minv)[np.newaxis,:]@dJ_dq_right.transpose((0,2,1)) + (contactJleft[np.newaxis,:])@dMinv_dq@(contactJright.T[np.newaxis,:])
            
            db_dq = (dJ_dq_left@(data.contactpreb[np.newaxis,:,np.newaxis])).squeeze(2).T + contactJleft@ddq_dq*self.dt
            db_dv = contactJleft@(ddq_dv*self.dt+np.eye(nv))
            db_dtau = contactJleft@Minv*self.dt
            
            if slide_ids == []:
                dlambda_dq = -contactAinv@( (dA_dq@impulse[np.newaxis,:,np.newaxis]).squeeze(2).T )
            else:
                dlambda_dq = -contactAinv@( (dA_dq@impulse[np.newaxis,:-(2*len(slide_ids)),np.newaxis]).squeeze(2).T )
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
            
            Fq = Minv@ ( (dJ_dq.transpose(0,2,1)@(impulse[np.newaxis,:,np.newaxis])).squeeze(2).T + contactJ.T@dlambda_dq )/self.dt
            Fv = Minv@contactJ.T@dlambda_dv/self.dt
            Ftau = Minv@contactJ.T@dlambda_dtau/self.dt
            
            data.Fx[:, :] += np.hstack([Fq, Fv])
            data.Fu[:, :] += Ftau@data.actuation.dtau_du
            
            self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DAD_contact(self)
        return data
    
def IAM_contact(N, state, actuation, costs, contact, DT, rho=0):
    assert N>1
    dmodel = DAM_contact(
        state, actuation, costs, contact, DT, rho
    )
    dmodels = [dmodel] * N
    actionmodels = [
            crocoddyl.IntegratedActionModelEuler(
                m, 1e-3
            ) for m in dmodels
        ]
    return actionmodels

def IAM_shoot(N, state, actuation, costs, contact, DT, rho=0):
    assert N>1
    dmodelr = DAM_contact(
        state, actuation, costs[0], contact, DT, rho
    )
    dmodelt = DAM_contact(
        state, actuation, costs[1], contact, DT, rho
    )
    actionmodels = [crocoddyl.IntegratedActionModelEuler(dmodelr, DT)] * N + [crocoddyl.IntegratedActionModelEuler(dmodelt, 0.0)]
    return actionmodels

def IAM_ccdContactFwd(N, state, actuation, costs, contact3D, DT):
    assert N>1
    dmodelr = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contact3D, costs[0]
    )
    dmodelt = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contact3D, costs[1]
    )
    actionmodels = [crocoddyl.IntegratedActionModelEuler(dmodelr, DT)] * N + [crocoddyl.IntegratedActionModelEuler(dmodelt, 0.0)]
    return actionmodels

def IAM_ccdFreeFwd(N, state, actuation, costs, DT):
    assert N>1
    dmodelr = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, costs[0]
    )
    dmodelt = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, costs[1]
    )
    actionmodels = [crocoddyl.IntegratedActionModelEuler(dmodelr, DT)] * N + [crocoddyl.IntegratedActionModelEuler(dmodelt, 0.0)]
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
    _, costs = handstand(rmodel, state, actuation)

    DT = 0.025
    contact_ids = [lf_id, rf_id, lh_id, rh_id,
                ]
    contact_model = ContactModel(state,contact_ids)
    DAM = DAM_contact(state, actuation,costs,contact_model,DT)
    x0 = rmodel.defaultState
    x0[2] += 0.2
    # x0[8] = -3.14/2
    u0 = np.ones(12)*0.1
    DAD = DAM.createData()
    DAM.calc(DAD,x0,u0)
    DAM.calcDiff(DAD,x0,u0)