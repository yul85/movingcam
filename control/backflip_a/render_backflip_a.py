from fltk import Fl
import pydart2 as pydart
import torch
from control.backflip_a.backflip_a_ppo_mp import PPO
from PyCommon.modules.GUI import DartViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
import numpy as np

from SkateUtils.Dart2bvh import dart2bvh

def main():
    MOTION_ONLY = False

    # MOTION_ONLY = True
    np.set_printoptions(precision=5)

    pydart.init()
    env_name = 'backflip_a'
    ppo = PPO(env_name, 0, visualize_only=True)
    if not MOTION_ONLY:
        ppo.LoadModel('backflip_a_model/' + 'max.pt')
        ppo.model.eval()

    ppo.env.flag_rsi(False)
    ppo.env.reset()
    ppo.env.ref_skel.set_positions(ppo.env.ref_motion.get_q(ppo.env.current_frame))

    # for bvh file
    bvh_qs = []
    bvh_file_name = f'{env_name}_output.bvh'

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

    rd_contact_inf_text = ['']
    rd_contact_inf_text_pos = [(-0.95, 0.85)]

    dart_world = ppo.env.world
    viewer_w, viewer_h = 1920, 1080
    viewer = hsv.DartViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55))
    viewer.doc.addRenderer('MotionModel', yr.DartRenderer(ppo.env.ref_world, (194,207,245), yr.POLYGON_FILL), visible=False)
    if not MOTION_ONLY:
        viewer.doc.addRenderer('controlModel', yr.DartRenderer(dart_world, (255,255,255), yr.POLYGON_FILL))
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

    def simulateCallback(frame):
        rd_frame_text[0] = str(frame)
        if ppo.env.p_fc is not None:
            rd_contact_inf_text[0] = str(ppo.env.p_fc_hat)

        state = ppo.env.state()
        action_dist, _ = ppo.model(torch.tensor(state.reshape(1, -1)).float())
        action = action_dist.loc.detach().numpy()
        res = ppo.env.step(action[0])

        q = [np.asarray(ppo.env.skel.q)]
        dq = [np.asarray(ppo.env.skel.dq)]

        # make bvh file
        bvh_qs.append(ppo.env.skel.q)

        if res[2]:
            viewer.motionViewWnd.pause()

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
        viewer.setMaxFrame(ppo.env.motion_len-2)
        CAMERA_TRACKING = False
        if CAMERA_TRACKING:
            cameraTargets = [None] * (viewer.getMaxFrame()+1)

        def postFrameCallback_Always(frame):
            if CAMERA_TRACKING:
                if cameraTargets[frame] is None:
                    cameraTargets[frame] = ppo.env.skel.body(0).com()
                viewer.setCameraTarget(cameraTargets[frame])

        viewer.setPostFrameCallback_Always(postFrameCallback_Always)
    viewer.startTimer(1./30.)
    viewer.show()

    Fl.run()

    dart2bvh(bvh_file_name, ppo.env.skel, bvh_qs, 30)


if __name__ == '__main__':
    main()
