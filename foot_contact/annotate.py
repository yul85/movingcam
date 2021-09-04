import torch
from network import Net
import os
import numpy as np
import copy
import json
import cv2
import argparse

lower_body_indices = np.asarray([8, 9, 10, 11, 12, 13, 14])
upper_body_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8])


def check_json_valid(_data, _previous_fail_indices, _previous_pelvis_pixel, _check_body_indices):
    _current_fail_indices = []
    _people_id = 0
    if len(_data['people']) < 1:
        return False, copy.deepcopy(_check_body_indices), _previous_pelvis_pixel, 0

    current_pelvis_pixel = np.asarray(_data['people'][_people_id]['pose_keypoints_2d']).reshape((-1, 3))[8, :2]

    # using pose_keypoints_2d
    pose_keypoints_2d = _data['people'][_people_id]['pose_keypoints_2d']
    pose_keypoints_2d = np.asarray(pose_keypoints_2d).reshape((-1, 3))
    for index in _check_body_indices:
        if pose_keypoints_2d[index][2] < 0.05 or (pose_keypoints_2d[index][0] == 0 and pose_keypoints_2d[index][1] == 0):
            _current_fail_indices.append(index)
            continue

    if len(set(_previous_fail_indices).intersection(set(_current_fail_indices))) > 0:
        return False, copy.deepcopy(_check_body_indices), current_pelvis_pixel, 0

    return True, _current_fail_indices, current_pelvis_pixel, _people_id


def make_one_frame_vector(_data, _people_id):
    one_frame_vector = np.zeros(0)
    keypoints_2d = np.asarray(_data['people'][_people_id]['pose_keypoints_2d']).reshape((-1, 3))
    for openpose_index in range(25):
        if keypoints_2d[openpose_index, 2] >= 0.05:
            one_frame_vector = np.append(one_frame_vector, keypoints_2d[openpose_index, :])
        else:
            one_frame_vector = np.append(one_frame_vector, np.zeros(3))

    return one_frame_vector.reshape((-1, 3))


if __name__ == '__main__':
    motion_name = input("motion name? ")
    # motion_name = 'parkour14_whole'
    motion_path = '/home/user/works/foot_contact/data'
    json_path = f'/home/user/works/foot_contact/enhance_test/{motion_name}/{motion_name}_enhance'
    output_path = '/home/user/works/foot_contact/test_rempe/' + motion_name

    os.system(f'mkdir -p {output_path}')

    # os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}     --write_video /openpose/output/{motion_name}.mp4                         --display 0 --part_candidates --number_people_max 1 --net_resolution -1x512')
    # os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_90  --write_video /openpose/output/{motion_name}_90.mp4  --frame_rotate 90   --display 0 --part_candidates --number_people_max 1 --net_resolution -1x512')
    # os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_180 --write_video /openpose/output/{motion_name}_180.mp4 --frame_rotate 180  --display 0 --part_candidates --number_people_max 1 --net_resolution -1x512')
    # os.system(f'docker run --rm --gpus all -v {output_path}:/openpose/output -v {motion_path}:/openpose/input openpose /openpose/build/examples/openpose/openpose.bin --video /openpose/input/{motion_name}.mp4 --write_json /openpose/output/{motion_name}_270 --write_video /openpose/output/{motion_name}_270.mp4  --frame_rotate 270 --display 0 --part_candidates --number_people_max 1 --net_resolution -1x512')
    # os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose")
    # os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}.mp4 {output_path}/{motion_name}_png_openpose/%03d.png")
    # os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_90")
    # os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_90.mp4 {output_path}/{motion_name}_png_openpose_90/%03d.png")
    # os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_180")
    # os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_180.mp4 {output_path}/{motion_name}_png_openpose_180/%03d.png")
    # os.system(f"mkdir -p {output_path}/{motion_name}_png_openpose_270")
    # os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_270.mp4 {output_path}/{motion_name}_png_openpose_270/%03d.png")

    json_list = [f for f in os.listdir(json_path) if os.path.isfile(os.path.join(json_path, f)) and '.json' in f]
    json_list.sort()

    check_body_indices = lower_body_indices
    vectors_for_test = {}
    vectors_for_rendering = {}

    valid_json_input_data_list = []
    previous_fail_indices = copy.deepcopy(check_body_indices)
    previous_pelvis_pixel = np.array([960., 540.])
    for json_idx, json_file_name in enumerate(json_list):
        with open(json_path + '/' + json_file_name, 'r') as json_file:
            data = json.load(json_file, encoding='utf-8')
            isvalid, current_fail_indices, previous_pelvis_pixel, people_id = check_json_valid(data, previous_fail_indices, previous_pelvis_pixel, check_body_indices)
            if isvalid:
                valid_json_input_data_list.append(make_one_frame_vector(data, people_id))
                if len(valid_json_input_data_list) > 2:
                    # fix previous frame
                    for fail_index in previous_fail_indices:
                        valid_json_input_data_list[-2][fail_index] = \
                            .5 * (valid_json_input_data_list[-3][fail_index] + valid_json_input_data_list[-1][fail_index])
                    vectors_for_rendering[json_idx-1] = valid_json_input_data_list[-2]
                while len(valid_json_input_data_list) == 10 or (len(current_fail_indices) == 0 and len(valid_json_input_data_list) == 9):
                    # normalize pixel value
                    # for 9 frames, normalize x, y pixel to [-0.8, 0.8] x [-0.8, 0.8]
                    # center pelvis pixel would be (0, 0)
                    input_full_vectors = np.vstack(tuple([valid_json_input_data_list[_i][check_body_indices].copy() for _i in range(9)]))
                    center_frame_pelvis_pixel = valid_json_input_data_list[4][8, :2]
                    max_pixel = np.amax(input_full_vectors[:, :2], axis=0)
                    min_pixel = np.amin(input_full_vectors[:, :2], axis=0)
                    ratio = 1.25 * np.maximum(max_pixel - center_frame_pelvis_pixel, center_frame_pelvis_pixel - min_pixel)
                    for _j in range(2):
                        input_full_vectors[:, _j] -= center_frame_pelvis_pixel[_j]
                        input_full_vectors[:, _j] /= ratio[_j]
                    input_full_vectors = input_full_vectors.reshape(-1)

                    # append to vectors_for_learning
                    vectors_for_test[json_idx - len(valid_json_input_data_list) + 5] = input_full_vectors
                    valid_json_input_data_list.pop(0)
            else:
                del valid_json_input_data_list[:]
            previous_fail_indices = copy.deepcopy(current_fail_indices)

    device = torch.device("cuda")

    model = Net(input_size=3*7*9, output_size=10, is_cuda=torch.cuda.is_available())
    model_path = 'model_foot/10000.pt'

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    joint_index_map = {'rf': 0, 'lf': 1}
    contact_result = [[0 for _ in range(len(json_list))], [0 for _ in range(len(json_list))], [0 for _ in range(len(json_list))], [0 for _ in range(len(json_list))]]
    contact_result_count = [[0 for _ in range(len(json_list))], [0 for _ in range(len(json_list))], [0 for _ in range(len(json_list))], [0 for _ in range(len(json_list))]]

    for k, v in vectors_for_test.items():
        if v is not None:
            result = model(torch.from_numpy(v.astype(np.float32)).to(device).unsqueeze(0)).cpu().detach().numpy().reshape((-1, 2))
            for joint in joint_index_map.keys():
                joint_index = joint_index_map[joint]
                for frame in range(k-2, k+3):
                    contact_result_count[joint_index][frame] += 1
                    if result[frame-k+2, joint_index_map[joint]] > 0.5:
                        contact_result[joint_index][frame] += 1

    for joint in joint_index_map.values():
        for frame, _ in enumerate(contact_result[joint]):
            if contact_result_count[joint][frame] > 0:
                contact_result[joint][frame] /= contact_result_count[joint][frame]
            else:
                contact_result[joint][frame] = None

    original_image_path = f"{output_path}/{motion_name}_png_ori"
    os.system(f"mkdir -p {original_image_path}")
    os.system(f"ffmpeg -loglevel fatal -y -i {motion_path}/{motion_name}.mp4 {original_image_path}/%03d.png")
    contact_result_foot_output = np.asarray([[0, 0] for _ in range(len(json_list))])
    for json_idx, json_file_name in enumerate(json_list):
        with open(json_path + '/' + json_file_name, 'r') as json_file:
            data = json.load(json_file, encoding='utf-8')
            if json_idx in vectors_for_rendering.keys() and contact_result[0][json_idx] is not None and contact_result[1][json_idx] is not None:
                for joint_index in lower_body_indices:
                    if joint_index == 11 and contact_result[0][json_idx] is not None and contact_result[0][json_idx] > 0.49:
                        contact_result_foot_output[json_idx][0] = 1.
                    elif joint_index == 14 and contact_result[1][json_idx] is not None and contact_result[1][json_idx] > 0.49:
                        contact_result_foot_output[json_idx][1] = 1.

    np.save(f"{output_path}/{motion_name}_contact_info_foot.npy", contact_result_foot_output)

    os.system(f"mkdir -p {output_path}/{motion_name}_png_contact_foot")
    image_list = [f for f in os.listdir(original_image_path) if os.path.isfile(os.path.join(original_image_path, f)) and '.png' in f]
    image_list.sort()
    for json_idx, json_file_name in enumerate(json_list):
        with open(json_path + '/' + json_file_name, 'r') as json_file:
            img = cv2.imread(f"{output_path}/{motion_name}_png_ori/{image_list[json_idx]}")

            data = json.load(json_file, encoding='utf-8')
            if json_idx in vectors_for_rendering.keys() and contact_result[0][json_idx] is not None and contact_result[1][json_idx] is not None:
                frame_data = vectors_for_rendering[json_idx].reshape((-1, 3))[:, :2]
                line_size = 5
                cv2.line(img, tuple(map(int, frame_data[8])), tuple(map(int, frame_data[9])), (0, 0, 0), 5)
                cv2.line(img, tuple(map(int, frame_data[9])), tuple(map(int, frame_data[10])), (0, 0, 0), 5)
                cv2.line(img, tuple(map(int, frame_data[10])), tuple(map(int, frame_data[11])), (0, 0, 0), 5)
                cv2.line(img, tuple(map(int, frame_data[8])), tuple(map(int, frame_data[12])), (0, 0, 0), 5)
                cv2.line(img, tuple(map(int, frame_data[12])), tuple(map(int, frame_data[13])), (0, 0, 0), 5)
                cv2.line(img, tuple(map(int, frame_data[13])), tuple(map(int, frame_data[14])), (0, 0, 0), 5)
                for joint_index in lower_body_indices:
                    joint_size = 30 if joint_index in [11, 14] else 15
                    if joint_index == 11 and contact_result[0][json_idx] is not None and contact_result[0][json_idx] > 0.49:
                        cv2.circle(img, tuple(map(int, frame_data[joint_index])), joint_size, (0, 0, 255), -1)
                    elif joint_index == 14 and contact_result[1][json_idx] is not None and contact_result[1][json_idx] > 0.49:
                        cv2.circle(img, tuple(map(int, frame_data[joint_index])), joint_size, (0, 0, 255), -1)
                    elif joint_index < 12:
                        cv2.circle(img, tuple(map(int, frame_data[joint_index])), joint_size, (255, 255, 255), -1)
                    else:
                        cv2.circle(img, tuple(map(int, frame_data[joint_index])), joint_size, (128, 128, 128), -1)

            cv2.imwrite(f"{output_path}/{motion_name}_png_contact_foot/{json_idx+1:03d}.png", img)
    os.system(f"ffmpeg -loglevel fatal -y -i {output_path}/{motion_name}_png_contact_foot/%03d.png -c:v libx264 -pix_fmt yuv420p -an {output_path}/{motion_name}_foot_contact.mp4")
