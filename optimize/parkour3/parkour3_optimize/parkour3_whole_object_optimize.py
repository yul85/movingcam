import numpy as np
import pydart2 as pydart
import time
from optimize.parkour3.parkour3_param_box1.parkour3_param_box1_ppo_mp import PPO as PPO1
from optimize.parkour3.parkour3_param_box2.parkour3_param_box2_ppo_mp import PPO as PPO2
from optimize.parkour3.parkour3_param_box3.parkour3_param_box3_ppo_mp import PPO as PPO3
import torch
from scipy.optimize import minimize
import torch.nn.functional as F
from itertools import count


def get_optimized_box_pos():
    env_name = 'parkour3_param_box'
    ppo1 = PPO1(env_name, 0, visualize_only=True)
    ppo2 = PPO2(env_name, 0, visualize_only=True)
    ppo3 = PPO3(env_name, 0, visualize_only=True)

    ppo1.LoadModel('../parkour3_param_box1/parkour3_param_box1_model/max.pt')
    ppo1.model.eval()

    ppo2.LoadModel('../parkour3_param_box2/parkour3_param_box2_model/max.pt')
    ppo2.model.eval()

    ppo3.LoadModel('../parkour3_param_box3/parkour3_param_box3_model/max.pt')
    ppo3.model.eval()

    def f(_x):
        box_pos_list = [np.array([0, 0., 0.]), np.array([0., 0.0, 0.]),
                        np.array([0., 0., 0.])]

        box_pos_list[0][1] = _x[0]
        box_pos_list[0][2] = _x[1]
        box_pos_list[1][1] = _x[2]
        box_pos_list[1][2] = _x[3]
        box_pos_list[2][1] = _x[4]
        box_pos_list[2][2] = _x[5]

        state = ppo1.env.reset()
        ppo1.env.flag_rsi(False)
        ppo1.env.reset_optimized_box(None, None, box_pos_list)
        ppo1.env.ref_skel.set_positions(ppo1.env.ref_motion.get_q(ppo1.env.current_frame))
        state0 = state.copy()

        _, v0 = ppo1.model(torch.tensor(state0.reshape(1, -1)).float())
        values0 = v0.detach().numpy().reshape(-1)

        state1, q, dq = None, None, None
        while True:
            action_dist, _ = ppo1.model(torch.tensor(state.reshape(1, -1)).float())
            action = action_dist.loc.detach().numpy()
            state, reward, is_done, _ = ppo1.env.step(action[0])
            if is_done:
                _, v1 = ppo1.model(torch.tensor(state.reshape(1, -1)).float())
                values1 = v1.detach().numpy().reshape(-1)
                return -(values0[0] - (0.99 ** ppo1.env.current_frame) * values1[0])
            elif ppo1.env.current_frame == ppo1.env.rf_contact_start_frame1:
                # print("ppo1 fff")
                state1 = state.copy()
                q = ppo1.env.skel.q
                dq = ppo1.env.skel.dq
                break

        _, v1 = ppo1.model(torch.tensor(state1.reshape(1, -1)).float())
        values1 = v1.detach().numpy().reshape(-1)

        state2 = ppo2.env.reset_optimized_box(q, dq, box_pos_list)
        _, v2 = ppo2.model(torch.tensor(state2.reshape(1, -1)).float())
        values2 = v2.detach().numpy().reshape(-1)
        state = state2.copy()

        while True:
            action_dist, _ = ppo2.model(torch.tensor(state.reshape(1, -1)).float())
            action = action_dist.loc.detach().numpy()
            state, reward, is_done, _ = ppo2.env.step(action[0])
            if is_done:
                _, v3 = ppo2.model(torch.tensor(state.reshape(1, -1)).float())
                values3 = v3.detach().numpy().reshape(-1)
                return -(values0[0] - (0.99 ** ppo1.env.rf_contact_start_frame1) * values1[0]) \
                       - ((0.99 ** ppo2.env.rf_contact_start_frame1) * values2[0] - (0.99 ** ppo2.env.current_frame) *
                          values3[0])
            elif ppo2.env.current_frame == ppo2.env.lf_contact_start_frame2:
                # print("ppo2 fff")
                state3 = state.copy()
                q = ppo2.env.skel.q
                dq = ppo2.env.skel.dq
                break

        _, v3 = ppo2.model(torch.tensor(state3.reshape(1, -1)).float())
        values3 = v3.detach().numpy().reshape(-1)

        state4 = ppo3.env.reset_optimized_box(q, dq, box_pos_list)
        _, v4 = ppo3.model(torch.tensor(state4.reshape(1, -1)).float())
        values4 = v4.detach().numpy().reshape(-1)

        return -(values0[0] - (0.99 ** ppo1.env.rf_contact_start_frame1) * values1[0]) - (
                (0.99 ** ppo2.env.rf_contact_start_frame1) * values2[0] - (0.99 ** ppo3.env.lf_contact_start_frame2) *
                values3[0]) - (
                       0.99 ** ppo3.env.current_frame) * values4[0]

    def g(_x):
        grad_range = np.array([-8, -7, -5, -4, -2, -1])
        box_pos_list = [np.array([0, 0., 0.]), np.array([0., 0.0, 0.]),
                        np.array([0., 0., 0.])]

        box_pos_list[0][1] = _x[0]
        box_pos_list[0][2] = _x[1]
        box_pos_list[1][1] = _x[2]
        box_pos_list[1][2] = _x[3]
        box_pos_list[2][1] = _x[4]
        box_pos_list[2][2] = _x[5]

        state = ppo1.env.reset()
        ppo1.env.flag_rsi(False)
        # ppo.env.reset()
        ppo1.env.reset_optimized_box(None, None, box_pos_list)
        ppo1.env.ref_skel.set_positions(ppo1.env.ref_motion.get_q(ppo1.env.current_frame))
        state0 = state.copy()

        xx0 = torch.autograd.Variable(torch.tensor(state0.reshape(1, -1)).float(), requires_grad=True)
        v0 = F.relu(ppo1.model.value_fc1(xx0))
        v0 = F.relu(ppo1.model.value_fc2(v0))
        v0 = F.relu(ppo1.model.value_fc3(v0))
        v0 = ppo1.model.value_fc4(v0)
        v0.backward()
        grad0 = xx0.grad.detach().numpy().reshape(-1)

        while True:
            action_dist, _ = ppo1.model(torch.tensor(state.reshape(1, -1)).float())
            action = action_dist.loc.detach().numpy()
            state, reward, is_done, _ = ppo1.env.step(action[0])
            if is_done:
                xx1 = torch.autograd.Variable(torch.tensor(state.reshape(1, -1)).float(), requires_grad=True)
                v1 = F.relu(ppo1.model.value_fc1(xx1))
                v1 = F.relu(ppo1.model.value_fc2(v1))
                v1 = F.relu(ppo1.model.value_fc3(v1))
                v1 = ppo1.model.value_fc4(v1)
                v1.backward()
                grad1 = xx1.grad.detach().numpy().reshape(-1)
                total_grad = (grad0[grad_range] - (0.99 ** ppo1.env.current_frame) * grad1[grad_range])
                return -total_grad
            elif ppo1.env.current_frame == ppo1.env.rf_contact_start_frame1:
                state1 = state.copy()
                q = ppo1.env.skel.q
                dq = ppo1.env.skel.dq
                break

        xx1 = torch.autograd.Variable(torch.tensor(state1.reshape(1, -1)).float(), requires_grad=True)
        v1 = F.relu(ppo1.model.value_fc1(xx1))
        v1 = F.relu(ppo1.model.value_fc2(v1))
        v1 = F.relu(ppo1.model.value_fc3(v1))
        v1 = ppo1.model.value_fc4(v1)
        v1.backward()
        grad1 = xx1.grad.detach().numpy().reshape(-1)

        state2 = ppo2.env.reset_optimized_box(q, dq, box_pos_list)
        xx2 = torch.autograd.Variable(torch.tensor(state2.reshape(1, -1)).float(), requires_grad=True)
        v2 = F.relu(ppo2.model.value_fc1(xx2))
        v2 = F.relu(ppo2.model.value_fc2(v2))
        v2 = F.relu(ppo2.model.value_fc3(v2))
        v2 = ppo2.model.value_fc4(v2)
        v2.backward()
        grad2 = xx2.grad.detach().numpy().reshape(-1)
        state = state2.copy()

        while True:
            action_dist, _ = ppo2.model(torch.tensor(state.reshape(1, -1)).float())
            action = action_dist.loc.detach().numpy()
            state, reward, is_done, _ = ppo2.env.step(action[0])
            if is_done:
                xx3 = torch.autograd.Variable(torch.tensor(state.reshape(1, -1)).float(), requires_grad=True)
                v3 = F.relu(ppo2.model.value_fc1(xx3))
                v3 = F.relu(ppo2.model.value_fc2(v3))
                v3 = F.relu(ppo2.model.value_fc3(v3))
                v3 = ppo2.model.value_fc4(v3)
                v3.backward()
                grad3 = xx3.grad.detach().numpy().reshape(-1)
                total_grad = (grad0[grad_range] - (0.99 ** ppo1.env.rf_contact_start_frame1) * grad1[grad_range]) + \
                             ((0.99 ** ppo2.env.rf_contact_start_frame1) * grad2[grad_range] - (
                                         0.99 ** ppo2.env.current_frame) * grad3[grad_range])
                return -total_grad
            elif ppo2.env.current_frame == ppo2.env.lf_contact_start_frame2:
                state3 = state.copy()
                q = ppo2.env.skel.q
                dq = ppo2.env.skel.dq
                break

        xx3 = torch.autograd.Variable(torch.tensor(state3.reshape(1, -1)).float(), requires_grad=True)
        v3 = F.relu(ppo2.model.value_fc1(xx3))
        v3 = F.relu(ppo2.model.value_fc2(v3))
        v3 = F.relu(ppo2.model.value_fc3(v3))
        v3 = ppo2.model.value_fc4(v3)
        v3.backward()
        grad3 = xx3.grad.detach().numpy().reshape(-1)

        state4 = ppo3.env.reset_optimized_box(q, dq, box_pos_list)
        xx4 = torch.autograd.Variable(torch.tensor(state4.reshape(1, -1)).float(), requires_grad=True)
        v4 = F.relu(ppo3.model.value_fc1(xx4))
        v4 = F.relu(ppo3.model.value_fc2(v4))
        v4 = F.relu(ppo3.model.value_fc3(v4))
        v4 = ppo3.model.value_fc4(v4)
        v4.backward()
        grad4 = xx4.grad.detach().numpy().reshape(-1)

        total_grad = (grad0[grad_range] - (0.99 ** ppo1.env.rf_contact_start_frame1) * grad1[grad_range]) + \
                     ((0.99 ** ppo2.env.rf_contact_start_frame1) * grad2[grad_range] - (
                                 0.99 ** ppo3.env.lf_contact_start_frame2) * grad3[grad_range]) + \
                     (0.99 ** ppo3.env.current_frame) * grad4[grad_range]
        return -total_grad

    # optimization
    tic = time.time()

    x0 = np.array([0.075, 1.5, 0.1, 3.5, 0.4, 5.5])

    bnds = ([0.075 - 0.03, 0.075 + 0.03], [1.5 - 0.3, 1.5 + 0.3], [0.1 - 0.04, 0.1 + 0.04], [3.5 - 0.4, 3.5 + 0.4],
            [0.4 - 0.16, 0.4 + 0.16], [5.5 - 0.4, 5.5 + 0.4])

    # optimization
    res_box_pos = minimize(f, x0, jac=g, method='SLSQP', bounds=bnds)
    return res_box_pos.x


def test_box_pos(optimized_x):
    env_name = 'parkour3_vibe'
    ppo1 = PPO1(env_name, 0, visualize_only=True)
    ppo2 = PPO2(env_name, 0, visualize_only=True)
    ppo3 = PPO3(env_name, 0, visualize_only=True)

    ppo1.LoadModel('../parkour3_param_box1/parkour3_param_box1_model/max.pt')
    ppo1.model.eval()

    ppo2.LoadModel('../parkour3_param_box2/parkour3_param_box2_model/max.pt')
    ppo2.model.eval()

    ppo3.LoadModel('../parkour3_param_box3/parkour3_param_box3_model/max.pt')
    ppo3.model.eval()

    ppo = ppo1

    box_pos_list = [np.array([0, 0., 0.]), np.array([0., 0.0, 0.]),
                    np.array([0., 0., 0.])]

    box_pos_list[0][1] = optimized_x[0]
    box_pos_list[0][2] = optimized_x[1]
    box_pos_list[1][1] = optimized_x[2]
    box_pos_list[1][2] = optimized_x[3]
    box_pos_list[2][1] = optimized_x[4]
    box_pos_list[2][2] = optimized_x[5]

    ppo.env.flag_rsi(False)
    ppo.env.reset_optimized_box(None, None, box_pos_list)
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(ppo.env.current_frame))
    done_frame = 0

    for frame in count(1):
        if frame == ppo1.env.rf_contact_start_frame1:
            q = ppo1.env.skel.q
            dq = ppo1.env.skel.dq
            ppo2.env.reset_optimized_box(q, dq, box_pos_list)

        if frame == ppo2.env.lf_contact_start_frame2:
            q = ppo2.env.skel.q
            dq = ppo2.env.skel.dq
            ppo3.env.reset_optimized_box(q, dq, box_pos_list)

        if ppo1.env.rf_contact_start_frame1 <= frame < ppo1.env.lf_contact_start_frame2:
            ppo = ppo2
        elif ppo1.env.lf_contact_start_frame2 <= frame:
            ppo = ppo3
        else:
            ppo = ppo1
        if frame == 0:
            return
        state = ppo.env.state()
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()

        res = ppo.env.step(action[0])
        q = ppo.env.ref_motion.get_q(frame)
        q[:6] = ppo.env.skel.q[:6]
        ppo.env.ref_skel.set_positions(q)
        if frame < ppo1.env.rf_contact_start_frame1:
            ppo2.env.skel.set_positions(ppo1.env.skel.q)

        if frame < ppo1.env.lf_contact_start_frame2:
            ppo3.env.skel.set_positions(ppo2.env.skel.q)
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(frame))
        if res[2]:
            done_frame = frame
            break

    return done_frame


if __name__ == "__main__":
    pydart.init()

    box_pos = np.zeros(0)
    end_frame = 0
    while end_frame < 70:
        box_pos = get_optimized_box_pos()
        end_frame = test_box_pos(box_pos)
    print(end_frame, box_pos)

    with open("parkour3_parameter.txt", 'a') as f:
        f.write(f'{end_frame} optimized_x = np.array([{", ".join(list(map(str, box_pos)))}])\n')
        f.flush()
