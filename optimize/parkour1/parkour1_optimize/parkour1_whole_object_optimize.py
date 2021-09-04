import numpy as np
import pydart2 as pydart
import time
from optimize.parkour1.parkour1_param_box1.parkour1_param_box1_ppo_mp import PPO as PPO1
from optimize.parkour1.parkour1_param_box2.parkour1_param_box2_ppo_mp import PPO as PPO2
from optimize.parkour1.parkour1_param_box3.parkour1_param_box3_ppo_mp import PPO as PPO3
from optimize.parkour1.parkour1_param_box4.parkour1_param_box4_ppo_mp import PPO as PPO4
import torch
from scipy.optimize import minimize
import torch.nn.functional as F
from itertools import count


def get_optimized_box_pos():
    env_name = 'parkour1_param_box'
    # optimize box
    ppo1 = PPO1(env_name, 0, visualize_only=True)
    ppo2 = PPO2(env_name, 0, visualize_only=True)
    ppo3 = PPO3(env_name, 0, visualize_only=True)
    ppo4 = PPO4(env_name, 0, visualize_only=True)

    # box1
    ppo1.LoadModel('../parkour1_param_box1/parkour1_param_box1_model/max.pt')
    ppo1.model.eval()

    # box2
    ppo2.LoadModel('../parkour1_param_box2/parkour1_param_box2_model/max.pt')
    ppo2.model.eval()

    # box3
    ppo3.LoadModel('../parkour1_param_box3/parkour1_param_box3_model/max.pt')
    ppo3.model.eval()

    # box4
    ppo4.LoadModel('../parkour1_param_box4/parkour1_param_box4_model/max.pt')
    ppo4.model.eval()

    def f(_x):
        box_pos_list = [np.array([0, 0.5, 2.]),
                        np.array([0., 0.4, 3.]),
                        np.array([-0.5, 0.6, 4.2]),
                        np.array([0., 0.4, 6.2]),
                        np.array([0., 0.25, 8.2])]

        box_pos_list[0][1] = _x[0]
        box_pos_list[0][2] = _x[1]
        box_pos_list[1][1] = _x[2]
        box_pos_list[1][2] = _x[3]
        box_pos_list[2][1] = _x[4]
        box_pos_list[2][2] = _x[5]
        box_pos_list[3][1] = _x[6]
        box_pos_list[3][2] = _x[7]
        box_pos_list[4][1] = _x[8]
        box_pos_list[4][2] = _x[9]

        state = ppo1.env.reset_optimized_box(None, None, box_pos_list)
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
            elif ppo1.env.current_frame == ppo1.env.lf_contact_start_frame2:
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
                return -(values0[0] - (0.99 ** ppo1.env.lf_contact_start_frame2) * values1[0]) \
                       - ((0.99 ** ppo2.env.lf_contact_start_frame2) * values2[0] - (0.99 ** ppo2.env.current_frame) *
                          values3[0])
            elif ppo2.env.current_frame == ppo2.env.rf_contact_start_frame2:
                state3 = state.copy()
                q = ppo2.env.skel.q
                dq = ppo2.env.skel.dq
                break

        _, v3 = ppo2.model(torch.tensor(state3.reshape(1, -1)).float())
        values3 = v3.detach().numpy().reshape(-1)

        state4 = ppo3.env.reset_optimized_box(q, dq, box_pos_list)
        _, v4 = ppo3.model(torch.tensor(state4.reshape(1, -1)).float())
        values4 = v4.detach().numpy().reshape(-1)
        state = state4.copy()
        #--------------------------------------------------------
        while True:
            action_dist, _ = ppo3.model(torch.tensor(state.reshape(1, -1)).float())
            action = action_dist.loc.detach().numpy()
            state, reward, is_done, _ = ppo3.env.step(action[0])
            if is_done:
                _, v5 = ppo3.model(torch.tensor(state.reshape(1, -1)).float())
                values5 = v4.detach().numpy().reshape(-1)
                return -(values0[0] - (0.99 ** ppo1.env.lf_contact_start_frame2) * values1[0]) - \
                        ((0.99 ** ppo2.env.lf_contact_start_frame2) * values2[0] - (0.99 ** ppo3.env.rf_contact_start_frame2) * values3[0]) - \
                        ((0.99 ** ppo3.env.rf_contact_start_frame2) * values4[0] - (0.99 ** ppo4.env.current_frame) * values5[0])
            elif ppo3.env.current_frame == ppo3.env.both_f_contact_start_frame1:
                state5 = state.copy()
                q = ppo3.env.skel.q
                dq = ppo3.env.skel.dq
                break

        _, v5 = ppo3.model(torch.tensor(state5.reshape(1, -1)).float())
        values5 = v5.detach().numpy().reshape(-1)

        state6 = ppo4.env.reset_optimized_box(q, dq, box_pos_list)
        _, v6 = ppo4.model(torch.tensor(state6.reshape(1, -1)).float())
        values6 = v6.detach().numpy().reshape(-1)
        #-------------------------------------------------------

        return -(values0[0] - (0.99 ** ppo1.env.lf_contact_start_frame2) * values1[0]) - \
               ((0.99 ** ppo2.env.lf_contact_start_frame2) * values2[0] - (0.99 ** ppo3.env.rf_contact_start_frame2) * values3[0]) - \
               ((0.99 ** ppo3.env.rf_contact_start_frame2) * values4[0] - (0.99 ** ppo4.env.both_f_contact_start_frame1) * values5[0]) - \
               (0.99 ** ppo4.env.current_frame) * values6[0]

    def g(_x):
        grad_range = np.array([-14, -13, -11, -10, -8, -7, -5, -4, -2, -1])
        box_pos_list = [np.array([0, 0.5, 2.]),
                        np.array([0., 0.4, 3.]),
                        np.array([-0.5, 0.6, 4.2]),
                        np.array([0., 0.4, 6.2]),
                        np.array([0., 0.25, 8.2])]

        box_pos_list[0][1] = _x[0]
        box_pos_list[0][2] = _x[1]
        box_pos_list[1][1] = _x[2]
        box_pos_list[1][2] = _x[3]
        box_pos_list[2][1] = _x[4]
        box_pos_list[2][2] = _x[5]
        box_pos_list[3][1] = _x[6]
        box_pos_list[3][2] = _x[7]
        box_pos_list[4][1] = _x[8]
        box_pos_list[4][2] = _x[9]

        state = ppo1.env.reset_optimized_box(None, None, box_pos_list)
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
            elif ppo1.env.current_frame == ppo1.env.lf_contact_start_frame2:
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
                total_grad = (grad0[grad_range] - (0.99 ** ppo1.env.lf_contact_start_frame2) * grad1[grad_range]) + \
                             ((0.99 ** ppo2.env.lf_contact_start_frame2) * grad2[grad_range] - (
                                         0.99 ** ppo2.env.current_frame) * grad3[grad_range])
                return -total_grad
            elif ppo2.env.current_frame == ppo2.env.rf_contact_start_frame2:
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
        state = state4.copy()

        while True:
            action_dist, _ = ppo3.model(torch.tensor(state.reshape(1, -1)).float())
            action = action_dist.loc.detach().numpy()
            state, reward, is_done, _ = ppo3.env.step(action[0])
            if is_done:
                xx5 = torch.autograd.Variable(torch.tensor(state.reshape(1, -1)).float(), requires_grad=True)
                v5 = F.relu(ppo3.model.value_fc1(xx5))
                v5 = F.relu(ppo3.model.value_fc2(v5))
                v5 = F.relu(ppo3.model.value_fc3(v5))
                v5 = ppo3.model.value_fc4(v5)
                v5.backward()
                grad5 = xx5.grad.detach().numpy().reshape(-1)
                total_grad = (grad0[grad_range] - (0.99 ** ppo1.env.lf_contact_start_frame2) * grad1[grad_range]) - \
                        ((0.99 ** ppo2.env.lf_contact_start_frame2) * grad2[grad_range] - (0.99 ** ppo3.env.rf_contact_start_frame2) * grad3[grad_range]) - \
                        ((0.99 ** ppo3.env.rf_contact_start_frame2) * grad4[grad_range] - (0.99 ** ppo4.env.current_frame) * grad5[grad_range])
                return -total_grad
            elif ppo3.env.current_frame == ppo3.env.both_f_contact_start_frame1:
                state5 = state.copy()
                q = ppo3.env.skel.q
                dq = ppo3.env.skel.dq
                break

        xx5 = torch.autograd.Variable(torch.tensor(state5.reshape(1, -1)).float(), requires_grad=True)
        v5 = F.relu(ppo3.model.value_fc1(xx5))
        v5 = F.relu(ppo3.model.value_fc2(v5))
        v5 = F.relu(ppo3.model.value_fc3(v5))
        v5 = ppo3.model.value_fc4(v5)
        v5.backward()
        grad5 = xx5.grad.detach().numpy().reshape(-1)

        state6 = ppo4.env.reset_optimized_box(q, dq, box_pos_list)
        xx6 = torch.autograd.Variable(torch.tensor(state6.reshape(1, -1)).float(), requires_grad=True)
        v6 = F.relu(ppo4.model.value_fc1(xx6))
        v6 = F.relu(ppo4.model.value_fc2(v6))
        v6 = F.relu(ppo4.model.value_fc3(v6))
        v6 = ppo4.model.value_fc4(v6)
        v6.backward()
        grad6 = xx6.grad.detach().numpy().reshape(-1)
        total_grad = (grad0[grad_range] - (0.99 ** ppo1.env.lf_contact_start_frame2) * grad1[grad_range]) - \
               ((0.99 ** ppo2.env.lf_contact_start_frame2) * grad2[grad_range] - (0.99 ** ppo3.env.rf_contact_start_frame2) * grad3[grad_range]) - \
               ((0.99 ** ppo3.env.rf_contact_start_frame2) * grad4[grad_range] - (0.99 ** ppo4.env.both_f_contact_start_frame1) * grad5[grad_range]) - \
               (0.99 ** ppo4.env.current_frame) * grad6[grad_range]
        return -total_grad

    # optimization
    tic = time.time()

    x0 = np.array([0.5, 2.0, 0.4, 3.0, 0.6, 4.2, 0.4, 6.2, 0.25, 8.2])

    bnds = ([0.5 - 0.1, 0.5 + 0.1], [2.0 - 0.4, 2.0 + 0.4], [0.4 - 0.08, 0.4 + 0.08], [3.0 - 0.2, 3.0 + 0.2],
            [0.6 - 0.06, 0.6 + 0.06], [4.2 - 0.12, 4.2 + 0.12], [0.4 - 0.04, 0.4 + 0.04], [6.2 - 0.2, 6.2 + 0.2],
            [0.25 - 0.025, 0.25 + 0.025], [8.2 - 0.2, 8.2 + 0.2])

    res_box_pos = minimize(f, x0, jac=g, method='SLSQP', bounds=bnds)  # optimization

    print(res_box_pos)
    return res_box_pos.x


def test_box_pos(optimized_x):
    env_name = 'parkour1_param_box'
    # optimize box
    ppo1 = PPO1(env_name, 0, visualize_only=True)
    ppo2 = PPO2(env_name, 0, visualize_only=True)
    ppo3 = PPO3(env_name, 0, visualize_only=True)
    ppo4 = PPO4(env_name, 0, visualize_only=True)

    # box1
    ppo1.LoadModel('../parkour1_param_box1/parkour1_param_box1_model/max.pt')
    ppo1.model.eval()

    # box2
    ppo2.LoadModel('../parkour1_param_box2/parkour1_param_box2_model/max.pt')
    ppo2.model.eval()

    # box3
    ppo3.LoadModel('../parkour1_param_box3/parkour1_param_box3_model/max.pt')
    ppo3.model.eval()

    # box4
    ppo4.LoadModel('../parkour1_param_box4/parkour1_param_box4_model/max.pt')
    ppo4.model.eval()

    ppo = ppo1

    box_pos_list = [np.array([0, 0.5, 2.]),
                    np.array([0., 0.4, 3.]),
                    np.array([-0.5, 0.6, 4.2]),
                    np.array([0., 0.4, 6.2]),
                    np.array([0., 0.25, 8.2])]

    box_pos_list[0][1] = optimized_x[0]
    box_pos_list[0][2] = optimized_x[1]
    box_pos_list[1][1] = optimized_x[2]
    box_pos_list[1][2] = optimized_x[3]
    box_pos_list[2][1] = optimized_x[4]
    box_pos_list[2][2] = optimized_x[5]
    box_pos_list[3][1] = optimized_x[6]
    box_pos_list[3][2] = optimized_x[7]
    box_pos_list[4][1] = optimized_x[8]
    box_pos_list[4][2] = optimized_x[9]

    ppo.env.flag_rsi(False)
    ppo.env.reset_optimized_box(None, None, box_pos_list)
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(ppo.env.current_frame))
    done_frame = 0

    for frame in count(1):
        if frame == ppo1.env.lf_contact_start_frame2:
            q = ppo1.env.skel.q
            dq = ppo1.env.skel.dq
            ppo2.env.reset_optimized_box(q, dq, box_pos_list)

        if frame == ppo2.env.rf_contact_start_frame2:
            q = ppo2.env.skel.q
            dq = ppo2.env.skel.dq
            ppo3.env.reset_optimized_box(q, dq, box_pos_list)

        if frame == ppo3.env.both_f_contact_start_frame1:
            q = ppo3.env.skel.q
            dq = ppo3.env.skel.dq
            ppo4.env.reset_optimized_box(q, dq, box_pos_list)

        if ppo1.env.lf_contact_start_frame2 <= frame < ppo1.env.rf_contact_start_frame2:
            ppo = ppo2
        elif ppo1.env.rf_contact_start_frame2 <= frame < ppo1.env.both_f_contact_start_frame1:
            ppo = ppo3
        elif ppo1.env.both_f_contact_start_frame1 <= frame:
            ppo = ppo4
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
        if frame < ppo1.env.lf_contact_start_frame2:
            ppo2.env.skel.set_positions(ppo1.env.skel.q)

        if frame < ppo1.env.rf_contact_start_frame2:
            ppo3.env.skel.set_positions(ppo2.env.skel.q)

        if frame < ppo1.env.both_f_contact_start_frame1:
            ppo4.env.skel.set_positions(ppo2.env.skel.q)

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
    print(f'{end_frame} optimized_x = np.array([{", ".join(list(map(str, box_pos)))}])\n')

    with open("parkour1_parameter.txt", 'a') as f:
        f.write(f'{end_frame} optimized_x = np.array([{", ".join(list(map(str, box_pos)))}])\n')
        f.flush()
