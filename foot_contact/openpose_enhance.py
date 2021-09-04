import os
import numpy as np
import json
import math
import argparse
import subprocess


if __name__ == '__main__':
    lower_body_indices = np.asarray([8, 9, 10, 11, 12, 13, 14])
    upper_body_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    body_indices = np.arange(15)

    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_name', type=str, default='')
    parser.add_argument('--video_dir_path', type=str, default='/home/user/works/foot_contact/data')
    parser.add_argument('--output_path', type=str, default='/home/user/works/foot_contact/enhance_test')
    parser.add_argument('--write_video', action='store_true', default=True)
    parser.add_argument('--delete_temp_files', action='store_false', default=False)
    args = parser.parse_args()

    if len(args.motion_name) > 0:
        motion_name = args.motion_name
    else:
        motion_name = input("motion name? ")

    # motion_name = "parkour14"
    motion_path = args.video_dir_path
    output_path = args.output_path + '/' + motion_name

    video_info = subprocess.check_output(f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=s=x:p=0 {motion_path}/{motion_name}.mp4", shell=True)
    width, height, frame_rate = video_info.decode().strip().split('x')
    width, height = int(width), int(height)
    frame_rate = list(map(float, frame_rate.split('/')))
    fps = int(frame_rate[0] / frame_rate[1] + 0.1)
    print(width, height, fps)
    # fps_dict = {
    #     "parkour10_whole": 25,
    #     "high_jump4": 25,
    #     "gangnam_style": 25,
    #     "dance": 24,
    #     "spiderman": 60,
    #     "hurdles": 18
    # }
    # fps = fps_dict[motion_name] if motion_name in fps_dict.keys() else 30
    # width, height = (1920, 1080) if "whole" in motion_name or "high_jump" in motion_name else (1280, 720)

    if not os.path.exists(output_path) and args.write_video:
        os.system(f'mkdir -p {output_path}')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_0   --write_video /openpose/output/{motion_name}_0.mp4   --write_video_fps {fps}                    --display 0 --number_people_max 1 --net_resolution -1x512')
        os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_0")
        os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_0.mp4 {output_path}/{motion_name}_png_openpose_0/%03d.png")

        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_90  --write_video /openpose/output/{motion_name}_90.mp4  --write_video_fps {fps} --frame_rotate 90  --display 0 --number_people_max 1 --net_resolution -1x512')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_180 --write_video /openpose/output/{motion_name}_180.mp4 --write_video_fps {fps} --frame_rotate 180 --display 0 --number_people_max 1 --net_resolution -1x512')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_270 --write_video /openpose/output/{motion_name}_270.mp4 --write_video_fps {fps} --frame_rotate 270 --display 0 --number_people_max 1 --net_resolution -1x512')

        os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_90")
        os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_90.mp4 {output_path}/{motion_name}_png_openpose_90/%03d.png")
        for png_file in [f for f in os.listdir(os.path.join(output_path, motion_name+'_png_openpose_90')) if os.path.isfile(os.path.join(os.path.join(output_path, motion_name+'_png_openpose_90'), f)) and '.png' in f]:
            os.system(f'mogrify -rotate "90" {output_path}/{motion_name}_png_openpose_90/{png_file}')
        os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_180")
        os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_180.mp4 {output_path}/{motion_name}_png_openpose_180/%03d.png")
        for png_file in [f for f in os.listdir(os.path.join(output_path, motion_name+'_png_openpose_180')) if os.path.isfile(os.path.join(os.path.join(output_path, motion_name+'_png_openpose_180'), f)) and '.png' in f]:
            os.system(f'mogrify -rotate "180" {output_path}/{motion_name}_png_openpose_180/{png_file}')
        os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_270")
        os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_270.mp4 {output_path}/{motion_name}_png_openpose_270/%03d.png")
        for png_file in [f for f in os.listdir(os.path.join(output_path, motion_name+'_png_openpose_270')) if os.path.isfile(os.path.join(os.path.join(output_path, motion_name+'_png_openpose_270'), f)) and '.png' in f]:
            os.system(f'mogrify -rotate "270" {output_path}/{motion_name}_png_openpose_270/{png_file}')

    elif not os.path.exists(output_path):
        os.system(f'mkdir -p {output_path}')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_0                      --display 0 --render_pose 0 --number_people_max 1 --net_resolution -1x512')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_90  --frame_rotate 90  --display 0 --render_pose 0 --number_people_max 1 --net_resolution -1x512')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_180 --frame_rotate 180 --display 0 --render_pose 0 --number_people_max 1 --net_resolution -1x512')
        os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_270 --frame_rotate 270 --display 0 --render_pose 0 --number_people_max 1 --net_resolution -1x512')

    if args.write_video:
        os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_enhance")
    os.system(f"mkdir -p {output_path}/{motion_name}_enhance")
    pose_keypoints_2d = [[] for _ in range(4)]
    json_postfix = ['_0', '_90', '_180', '_270']
    
    for postfix_idx, postfix in enumerate(json_postfix):
        json_dir = os.path.join(output_path, motion_name + postfix)
        json_list = [f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f)) and '.json' in f]
        json_list.sort()

        for json_idx, json_file_name in enumerate(json_list):
            json_path = os.path.join(json_dir, json_file_name)
            with open(json_path, 'r') as json_file:
                data = json.load(json_file, encoding='utf-8')
                if len(data['people']) == 0:
                    keypoint_data = np.zeros((25, 3))
                else:
                    keypoint_data = np.asarray(data['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
                    for joint_idx in range(len(keypoint_data)):
                        if postfix_idx == 1:
                            keypoint_data[joint_idx, 0], keypoint_data[joint_idx, 1] = width - keypoint_data[joint_idx, 1], keypoint_data[joint_idx, 0]
                        elif postfix_idx == 2:
                            keypoint_data[joint_idx, 0], keypoint_data[joint_idx, 1] = width - keypoint_data[joint_idx, 0], height - keypoint_data[joint_idx, 1]
                        elif postfix_idx == 3:
                            keypoint_data[joint_idx, 0], keypoint_data[joint_idx, 1] = keypoint_data[joint_idx, 1], height - keypoint_data[joint_idx, 0]
                pose_keypoints_2d[postfix_idx].append(keypoint_data)

    w_confidence = 0.5
    w_complete = 0.5
    w_pelvis = 0.01
    w_joint_move = 0.001
    w_joint = 1.2
    w_joint_btw = 1.
    # w_ee = 0.02
    w_ee = 0.0
    prev_prev_postfix_idx = 0
    prev_postfix_idx = 0
    prev_frame = 0
    enhance_result = [0]
    if args.write_video:
        os.system(f'cp -f {output_path}/{motion_name}_png_openpose{json_postfix[0]}/001.png {output_path}/{motion_name}_png_openpose_enhance/001.png')
    os.system(f'cp -f {output_path}/{motion_name}{json_postfix[0]}/{motion_name}_00000000{0:04d}_keypoints.json {output_path}/{motion_name}_enhance/')
    for frame in range(1, len(pose_keypoints_2d[0])):
        # print(frame+1, prev_frame+1)
        this_frame_compare_skip = False
        f_confidence_objectives = [w_confidence * np.sum(pose_keypoints_2d[postfix_idx][frame][body_indices, 2]) for postfix_idx in range(4)]
        f_complete_objectives = [w_complete * np.sum(pose_keypoints_2d[postfix_idx][frame][body_indices, 2] > 0.1) for postfix_idx in range(4)]
        f_pelvis_objectives = [-w_pelvis * np.linalg.norm(pose_keypoints_2d[postfix_idx][frame][8, :2] - pose_keypoints_2d[prev_postfix_idx][prev_frame][8, :2]) for postfix_idx in range(4)]
        f_joint_move_objectives = [0., 0., 0., 0.]
        for joint_idx in body_indices:
            if pose_keypoints_2d[prev_postfix_idx][prev_frame][joint_idx, 2] > 0.05:
                joint_move_min = 10000000.
                skipped_postfix_indices = []
                for postfix_idx in range(4):
                    pelvis_offset = pose_keypoints_2d[postfix_idx][frame][8, :2] - pose_keypoints_2d[prev_postfix_idx][prev_frame][8, :2]
                    if pose_keypoints_2d[postfix_idx][frame][joint_idx, 2] > 0.05:
                        joint_move = w_joint_move * (np.linalg.norm(pose_keypoints_2d[postfix_idx][frame][joint_idx, :2] - pose_keypoints_2d[prev_postfix_idx][prev_frame][joint_idx, :2] - pelvis_offset))
                        joint_move_min = min(joint_move_min, joint_move)
                        f_joint_move_objectives[postfix_idx] -= joint_move
                    else:
                        skipped_postfix_indices.append(postfix_idx)
                if len(skipped_postfix_indices) < 4:
                    for postfix_idx in skipped_postfix_indices:
                        f_joint_move_objectives[postfix_idx] -= joint_move_min

        objectives = [f_confidence_objectives[postfix_idx] + f_complete_objectives[postfix_idx] + f_pelvis_objectives[postfix_idx] + f_joint_move_objectives[postfix_idx] for postfix_idx in range(4)]

        # find body line flip (upside-down)
        for postfix_idx in range(4):
            curr_body_vec = pose_keypoints_2d[postfix_idx][frame][1, :2] - pose_keypoints_2d[postfix_idx][frame][8, :2]
            prev_body_vec = pose_keypoints_2d[prev_postfix_idx][prev_frame][1, :2] - pose_keypoints_2d[prev_postfix_idx][prev_frame][8, :2]
            if math.acos(np.clip(np.dot(curr_body_vec, prev_body_vec)/np.linalg.norm(curr_body_vec)/np.linalg.norm(prev_body_vec), -0.9999, 0.9999)) > math.pi/2.:
                objectives[postfix_idx] -= 100.

        # find limbs which are switched
        f_joint_list = [(9, 10), (12, 13)]
        joint_angle_diff = [[-0.000001 for _ in range(4)] for _ in f_joint_list]
        if prev_frame == frame - 1:
            for f_joint_pair_idx, f_joint_pair in enumerate(f_joint_list):
                if pose_keypoints_2d[prev_postfix_idx][prev_frame][f_joint_pair[0], 2] < 0.05 or pose_keypoints_2d[prev_postfix_idx][prev_frame][f_joint_pair[1], 2] < 0.05:
                    continue
                for postfix_idx in range(4):
                    prev_vec = pose_keypoints_2d[prev_postfix_idx][prev_frame][f_joint_pair[1], :2] - pose_keypoints_2d[prev_postfix_idx][prev_frame][f_joint_pair[0], :2]
                    curr_vec = pose_keypoints_2d[postfix_idx][frame][f_joint_pair[1], :2] - pose_keypoints_2d[postfix_idx][frame][f_joint_pair[0], :2]
                    prev_joint_angle = math.atan2(prev_vec[1], prev_vec[0])
                    curr_joint_angle = math.atan2(curr_vec[1], curr_vec[0])
                    _joint_angle_diff = curr_joint_angle - prev_joint_angle
                    _joint_angle_diff = np.min([abs(_joint_angle_diff - 2 * math.pi), abs(_joint_angle_diff), abs(_joint_angle_diff + 2 * math.pi)])
                    joint_angle_diff[f_joint_pair_idx][postfix_idx] = _joint_angle_diff

            for f_joint_pair_idx, f_joint_pair in enumerate(f_joint_list):
                for postfix_idx in range(4):
                    if joint_angle_diff[f_joint_pair_idx][postfix_idx] > math.pi/180. * 15.:
                        objectives[postfix_idx] -= 2.

        max_postfix_idx = int(np.argmax(objectives))
        if (motion_name == "parkourv2_03") and (104 <= frame+1 <= 122 or frame+1 in [126, 145, 146, 147, 148, 149]):
            max_postfix_idx = int(np.argmax(objectives[1:]) + 1)
        prev_prev_postfix_idx = prev_postfix_idx
        enhance_result.append(max_postfix_idx)

        curr_body_vec = pose_keypoints_2d[max_postfix_idx][frame][1, :2] - pose_keypoints_2d[max_postfix_idx][frame][8, :2]
        prev_body_vec = pose_keypoints_2d[prev_postfix_idx][prev_frame][1, :2] - pose_keypoints_2d[prev_postfix_idx][prev_frame][8, :2]
        if math.acos(np.clip(np.dot(curr_body_vec, prev_body_vec)/np.linalg.norm(curr_body_vec)/np.linalg.norm(prev_body_vec), -0.9999, 0.9999)) > math.pi/2.:
            this_frame_compare_skip = True
        elif sum([pose_keypoints_2d[max_postfix_idx][frame][1, 2], pose_keypoints_2d[max_postfix_idx][frame][8, 2]]) < 0.001:
            this_frame_compare_skip = True

        if prev_frame == frame - 1:
            for f_joint_pair_idx, f_joint_pair in enumerate(f_joint_list):
                if joint_angle_diff[f_joint_pair_idx][max_postfix_idx] > math.pi/180. * 60. and prev_frame == frame - 1:
                    this_frame_compare_skip = True
                    break

        if not this_frame_compare_skip:
            prev_postfix_idx = max_postfix_idx
            prev_frame = frame

        if args.write_video:
            os.system(f'cp -f {output_path}/{motion_name}_png_openpose{json_postfix[max_postfix_idx]}/{frame+1:03d}.png {output_path}/{motion_name}_png_openpose_enhance/{frame+1:03d}.png')

    for json_idx, postfix_idx in enumerate(enhance_result):
        postfix = json_postfix[postfix_idx]
        json_dir = os.path.join(output_path, motion_name + postfix)
        json_file_name = f'{motion_name}_0000{json_idx:08d}_keypoints.json'
        json_path = os.path.join(json_dir, json_file_name)
        with open(json_path, 'r') as json_file:
            data = json.load(json_file, encoding='utf-8')
            if data['people']:
                data['people'][0]['pose_keypoints_2d'] = pose_keypoints_2d[postfix_idx][json_idx].reshape((-1,)).tolist()
            with open(f'{output_path}/{motion_name}_enhance/{json_file_name}', 'w') as json_output_file:
                json.dump(data, json_output_file)

    if args.write_video:
        os.system(f'ffmpeg -loglevel fatal -y -r {fps} -i {output_path}/{motion_name}_png_openpose_enhance/%03d.png -c:v libx264 -pix_fmt yuv420p -an {output_path}/{motion_name}_enhance.mp4')

    if args.delete_temp_files:
        os.system(f"rm -rf {output_path}/{motion_name}_0")
        os.system(f"rm -rf {output_path}/{motion_name}_90")
        os.system(f"rm -rf {output_path}/{motion_name}_180")
        os.system(f"rm -rf {output_path}/{motion_name}_270")
        os.system(f"mv {output_path}/{motion_name}_enhance/*.json {output_path}")
        os.system(f"rm -rf {output_path}/{motion_name}_enhance")
        os.system(f"rm -rf {output_path}/{motion_name}_png_openpose_0")
        os.system(f"rm -rf {output_path}/{motion_name}_png_openpose_90")
        os.system(f"rm -rf {output_path}/{motion_name}_png_openpose_180")
        os.system(f"rm -rf {output_path}/{motion_name}_png_openpose_270")
        os.system(f"rm -rf {output_path}/{motion_name}_png_openpose_enhance")
