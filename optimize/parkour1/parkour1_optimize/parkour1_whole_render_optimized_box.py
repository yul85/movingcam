from fltk import Fl
import pydart2 as pydart
import torch
from optimize.parkour1.parkour1_param_box1.parkour1_param_box1_ppo_mp import PPO as PPO1
from optimize.parkour1.parkour1_param_box2.parkour1_param_box2_ppo_mp import PPO as PPO2
from optimize.parkour1.parkour1_param_box3.parkour1_param_box3_ppo_mp import PPO as PPO3
from optimize.parkour1.parkour1_param_box4.parkour1_param_box4_ppo_mp import PPO as PPO4
from PyCommon.modules.GUI import DartViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np
import os
from SkateUtils.DartMotionEdit import DartSkelMotion
from SkateUtils.Dart2bvh import dart2bvh

def main():
    MOTION_ONLY = False
    np.set_printoptions(precision=5)

    pydart.init()
    env_name = 'parkour1_param_box'
    ppo1 = PPO1(env_name, 0, visualize_only=True)
    ppo2 = PPO2(env_name, 0, visualize_only=True)
    ppo3 = PPO3(env_name, 0, visualize_only=True)
    ppo4 = PPO4(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        ppo1.LoadModel('../parkour1_param_box1/parkour1_param_box1_model/max.pt')
        ppo1.model.eval()

        ppo2.LoadModel('../parkour1_param_box2/parkour1_param_box2_model/max.pt')
        ppo2.model.eval()

        ppo3.LoadModel('../parkour1_param_box3/parkour1_param_box3_model/max.pt')
        ppo3.model.eval()

        ppo4.LoadModel('../parkour1_param_box4/parkour1_param_box4_model/max.pt')
        ppo4.model.eval()

    ppo = ppo1

    box_pos_list = [np.array([0,  0.5, 2.]),
                    np.array([0., 0.4, 3.]),
                    np.array([-0.5, 0.6, 4.2]),
                    np.array([0., 0.4, 6.2]),
                    np.array([0., 0.25, 8.2])]

    optimized_x = np.array([0.46068540076285386, 1.8427416030177972, 0.43145167937042705, 3.0782543364149784, 0.5764112417756978, 4.245464644915539, 0.4157258398836194, 6.277762249382121, 0.2598286499309781, 8.277981255535778])

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

    bvh_qs = []
    file_index = 1
    bvh_file_name = 'ppo_parkour1_param_optimized.bvh'
    box_pos_file_name = 'ppo_parkour1_param.txt'

    # viewer settings
    rd_contact_positions = [None]
    rd_contact_forces = [None]
    rd_COM = [None]
    rd_ext_force = [None]
    rd_ext_force_pos = [None]
    rd_traj = []
    ref_traj = []
    rd_frame_text_label = ['frame: ']
    rd_frame_text_label_pos = [(-0.95, 0.95)]
    rd_frame_text = ['0']
    rd_frame_text_pos = [(-0.9, 0.95)]
    rd_cotact_label_text = ['[lf rf lh rh]']
    rd_cotact_label_text_pos = [(-0.95, 0.9)]

    rd_contact_inf_text = ['', '']
    rd_contact_inf_text_pos = [(-0.95, 0.85), (-0.95, 0.8)]


    dart_world = ppo.env.world
    viewer_w, viewer_h = 1920, 1080
    viewer = hsv.DartViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55))
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (194,207,245), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('ppo1Model', yr.DartRenderer(ppo1.env.world, (255, 255, 255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('ppo2Model', yr.DartRenderer(ppo2.env.world, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('ppo3Model', yr.DartRenderer(ppo3.env.world, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('ppo4Model', yr.DartRenderer(ppo4.env.world, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    if not MOTION_ONLY:
        viewer.doc.addRenderer('contact', yr.VectorsRenderer(rd_contact_forces, rd_contact_positions, (255,0,0)))
        viewer.doc.addRenderer('COM projection', yr.PointsRenderer(rd_COM))
        viewer.doc.addRenderer('ext force', yr.WideArrowRenderer(rd_ext_force, rd_ext_force_pos, lineWidth=.1, fromPoint=False))
        viewer.doc.addRenderer('trajectory', yr.LinesRenderer(rd_traj))
        viewer.doc.addRenderer('ref_trajectory', yr.LinesRenderer(ref_traj, (0, 0, 255)))

        viewer.doc.addRenderer('frame_label',
                               yr.TextRenderer(rd_frame_text_label, rd_frame_text_label_pos, text_size=30))
        viewer.doc.addRenderer('frame_text', yr.TextRenderer(rd_frame_text, rd_frame_text_pos, text_size=30))
        viewer.doc.addRenderer('contact_label',
                               yr.TextRenderer(rd_cotact_label_text, rd_cotact_label_text_pos, text_size=30))
        viewer.doc.addRenderer('contact_info',
                               yr.TextRenderer(rd_contact_inf_text, rd_contact_inf_text_pos, text_size=30))


    def postCallback(frame):
        ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(frame))
        if ppo1.env.lf_contact_start_frame2 <= frame < ppo1.env.rf_contact_start_frame2:
            viewer.doc.setRendererVisible('ppo1Model', False)
            viewer.doc.setRendererVisible('ppo2Model', True)
            viewer.doc.setRendererVisible('ppo3Model', False)
            viewer.doc.setRendererVisible('ppo4Model', False)
        elif ppo1.env.rf_contact_start_frame2 <= frame < ppo1.env.both_f_contact_start_frame1:
            viewer.doc.setRendererVisible('ppo1Model', False)
            viewer.doc.setRendererVisible('ppo2Model', False)
            viewer.doc.setRendererVisible('ppo3Model', True)
            viewer.doc.setRendererVisible('ppo4Model', False)
        elif ppo1.env.both_f_contact_start_frame1 <= frame:
            viewer.doc.setRendererVisible('ppo1Model', False)
            viewer.doc.setRendererVisible('ppo2Model', False)
            viewer.doc.setRendererVisible('ppo3Model', False)
            viewer.doc.setRendererVisible('ppo4Model', True)
        else:
            viewer.doc.setRendererVisible('ppo1Model', True)
            viewer.doc.setRendererVisible('ppo2Model', False)
            viewer.doc.setRendererVisible('ppo3Model', False)
            viewer.doc.setRendererVisible('ppo4Model', False)

    contact_result = []
    def simulateCallback(frame):
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

        rd_frame_text[0] = f'{frame}'
        if ppo.env.p_fc is not None:
            rd_contact_inf_text[0] = f'des: {ppo.env.p_fc_hat}'
            rd_contact_inf_text[1] = f'cur: {ppo.env.p_fc}'
            contact_result.append(sum(ppo.env.p_fc_hat == ppo.env.p_fc)-2)

        if res[2]:
            pass

        q = [np.asarray(ppo.env.skel.q)]
        dq = [np.asarray(ppo.env.skel.dq)]

        # make bvh file
        bvh_qs.append(ppo.env.skel.q)

        # contact rendering
        contacts = ppo.env.world.collision_result.contacts
        del rd_contact_forces[:]
        del rd_contact_positions[:]
        for contact in contacts:
            if contact.skel_id1 == 0:
                rd_contact_forces.append(-contact.f/1000.)
            else:
                rd_contact_forces.append(contact.f/1000.)
            rd_contact_positions.append(contact.p)

        # com rendering
        del rd_COM[:]
        com = ppo.env.skel.com()
        com[1] = 0.
        rd_COM.append(com)

        rd_traj.append(ppo.env.skel.com())
        ref_traj.append(ppo.env.ref_skel.com())

        # ext force rendering
        del rd_ext_force[:]
        del rd_ext_force_pos[:]
        if ppo.env.ext_force_duration > 0.:
            rd_ext_force.append(ppo.env.ext_force/500.)
            rd_ext_force_pos.append(ppo.env.skel.body('h_spine').to_world())

    if MOTION_ONLY:
        viewer.setPostFrameCallback_Always(postCallback)
        viewer.setMaxFrame(len(ppo.env.ref_motion)-2)
    else:
        viewer.setSimulateCallback(simulateCallback)
        viewer.setMaxFrame(ppo.env.motion_len-10)
        CAMERA_TRACKING = False
        if CAMERA_TRACKING:
            cameraTargets = [None] * (viewer.getMaxFrame()+1)

        def postFrameCallback_Always(frame):
            if CAMERA_TRACKING:
                if cameraTargets[frame] is None:
                    cameraTargets[frame] = ppo.env.skel.body(0).com()
                viewer.setCameraTarget(cameraTargets[frame])

        viewer.setPostFrameCallback_Always(postCallback)
    viewer.startTimer(1./25.)
    viewer.show()

    Fl.run()
    # dart2bvh(bvh_file_name, ppo.env.skel, bvh_qs, 25)


if __name__ == '__main__':
    main()