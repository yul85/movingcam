import os
from os import listdir
from os.path import isfile, join, isdir
import time
import sys
import json
import numpy as np
import csv
import copy
import pickle


def check_json_valid(_data, _previous_fail_indices, _body_indices):
    _current_fail_indices = []
    if len(_data['people']) != 1:
        return False, copy.deepcopy(_body_indices)
    pose_keypoints_2d = _data['people'][0]['pose_keypoints_2d']
    pose_keypoints_2d = np.asarray(pose_keypoints_2d).reshape((-1, 3))
    for index in _body_indices:
        if pose_keypoints_2d[index][2] < 0.05:
            _current_fail_indices.append(index)

    # using part_candidates
    # part_candidates = _data['part_candidates'][0]
    # for index in _body_indices:
    #     candidate = part_candidates[str(index)]
    #     if len(candidate) == 0:
    #         _current_fail_indices.append(index)
    #         continue
    #     candidate = np.asarray(candidate).reshape((-1, 3))
    #     candidate = candidate[(-candidate[:, 2]).argsort()]
    #     if candidate[0, 2] < 0.05:
    #         _current_fail_indices.append(index)
    #         continue

    if len(set(_previous_fail_indices).intersection(set(_current_fail_indices))) > 0:
        return False, copy.deepcopy(_body_indices)

    return True, _current_fail_indices


def make_one_frame_vector(_data):
    one_frame_vector = np.zeros(0)
    for openpose_index in range(25):
        candidate = _data['part_candidates'][0][str(openpose_index)]
        if len(candidate) == 0:
            one_frame_vector = np.append(one_frame_vector, np.zeros(3))
        else:
            candidate = np.asarray(candidate).reshape((-1, 3))
            candidate = candidate[(-candidate[:, 2]).argsort()]
            one_frame_vector = np.append(one_frame_vector, candidate[0])
    return one_frame_vector.reshape((-1, 3))


if __name__ == '__main__':
    character_result_path = 'mixamo_render_result'
    characters = [f for f in listdir(character_result_path) if isdir(join(character_result_path, f))]
    print(characters)

    # lower_body_indices = np.asarray([8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24])
    full_body_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    lower_body_indices = np.asarray([8, 9, 10, 11, 12, 13, 14])
    upper_body_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8])
    body_indices = [lower_body_indices, upper_body_indices, full_body_indices]
    count_miss_total = [0 for _ in body_indices]
    count_frame_total = [0 for _ in body_indices]
    count_valid_input_vector_total = [0 for _ in body_indices]
    vectors_for_learning = [[] for _ in body_indices]
    for character_name in characters:
        print(character_name, 'Start')
        count_miss = [0 for _ in body_indices]
        count_frame = [0 for _ in body_indices]
        count_valid_input_vector = [0 for _ in body_indices]
        start_time = time.time()
        json_path = 'mixamo_openpose_result/' + character_name
        annotation_path = 'mixamo_annotation_result/' + character_name
        json_motion_dirs = [f for f in listdir(json_path) if isdir(join(json_path, f))]
        for d in json_motion_dirs:
            json_list = listdir(json_path + '/' + d)
            json_list.sort()
            if "_Fix" == d[-4:]:
                annotation_file_path = join(annotation_path, d[:-4] + '.txt')
            elif "_R_track" in d:
                annotation_file_path = join(annotation_path, d[:-8] + '.txt')
            elif "_T_track_" in d:
                annotation_file_path = join(annotation_path, d[:-15] + '.txt')
            annotation_lines = []
            with open(annotation_file_path, 'r') as annotation_file:
                csv_reader = csv.reader(annotation_file)
                for line in csv_reader:
                    if line[0] == 'rtoe':
                        continue
                    _line = ['1' if line[0] == '1' or line[2] == '1' else '0',
                             '1' if line[1] == '1' or line[3] == '1' else '0',
                             line[4], line[5]]
                    annotation_lines.append(list(map(float, _line)))
            for body_indicies_index, body_part_indices in enumerate(body_indices):
                valid_json_input_data_list = []
                valid_json_output_data_list = []
                previous_fail_indices = copy.deepcopy(body_part_indices)
                for json_idx in range(len(json_list)):
                    json_file_name = json_list[json_idx]
                    with open(json_path + '/' + d + '/' + json_file_name, 'r') as json_file:
                        data = json.load(json_file, encoding='utf-8')
                        count_frame[body_indicies_index] += 1
                        isvalid, current_fail_indices = check_json_valid(data, previous_fail_indices, body_part_indices)
                        if isvalid:
                            valid_json_input_data_list.append(np.asarray(data['people'][0]['pose_keypoints_2d']).reshape((-1, 3)))
                            if body_indicies_index == 0:
                                valid_json_output_data_list.append(annotation_lines[json_idx][:2])
                            elif body_indicies_index == 1:
                                valid_json_output_data_list.append(annotation_lines[json_idx][2:4])
                            elif body_indicies_index == 2:
                                valid_json_output_data_list.append(annotation_lines[json_idx])
                            if len(valid_json_input_data_list) > 2:
                                # fix previous frame
                                for fail_index in previous_fail_indices:
                                    valid_json_input_data_list[-2][fail_index] = \
                                        .5 * (valid_json_input_data_list[-3][fail_index] + valid_json_input_data_list[-1][fail_index])
                            while len(valid_json_input_data_list) == 10 or (len(current_fail_indices) == 0 and len(valid_json_input_data_list) == 9):
                                count_valid_input_vector[body_indicies_index] += 1
                                # normalize pixel value
                                # for 9 frames, normalize x, y pixel to [-0.8, 0.8] x [-0.8, 0.8]
                                # center pelvis pixel would be (0, 0)
                                input_full_vectors = np.vstack(tuple([valid_json_input_data_list[_i][body_part_indices] for _i in range(9)]))
                                center_frame_pelvis_pixel = valid_json_input_data_list[4][8, :2]
                                max_pixel = np.amax(input_full_vectors[:, :2], axis=0)
                                min_pixel = np.amin(input_full_vectors[:, :2], axis=0)
                                ratio = 1.25 * np.maximum(max_pixel - center_frame_pelvis_pixel, center_frame_pelvis_pixel - min_pixel)
                                # print(ratio)
                                for _j in range(2):
                                    input_full_vectors[:, _j] -= center_frame_pelvis_pixel[_j]
                                    input_full_vectors[:, _j] /= ratio[_j]
                                input_full_vectors = input_full_vectors.reshape(-1)

                                # output_full_vectors = np.vstack(tuple([valid_json_output_data_list[_i][:4] for _i in range(2, 7)])).reshape(-1)
                                output_full_vectors = np.vstack(tuple([valid_json_output_data_list[_i] for _i in range(2, 7)])).reshape(-1)
                                # print(input_full_vectors, output_full_vectors)

                                # append to vectors_for_learning
                                vectors_for_learning[body_indicies_index].append((input_full_vectors, output_full_vectors))
                                valid_json_input_data_list.pop(0)
                                valid_json_output_data_list.pop(0)
                        else:
                            del valid_json_input_data_list[:]
                            del valid_json_output_data_list[:]
                            count_miss[body_indicies_index] += 1
                        previous_fail_indices = current_fail_indices

        print(character_name, 'End', 'Total frame:', count_frame, 'valid input vector num:', count_valid_input_vector, 'miss skeleton frame:', count_miss)
        for i in range(len(body_indices)):
            count_miss_total[i] += count_miss[i]
            count_frame_total[i] += count_frame[i]
            count_valid_input_vector_total[i] += count_valid_input_vector[i]
    print('Total frame:', count_frame_total, 'valid input vector num:', count_valid_input_vector_total, 'miss skeleton frame:', count_miss_total)

    with open('mixamo_data_foot.pkl', 'wb') as f:
        pickle.dump(vectors_for_learning[0], f)

    with open('mixamo_data_hand.pkl', 'wb') as f:
        pickle.dump(vectors_for_learning[1], f)

    with open('mixamo_data_full.pkl', 'wb') as f:
        pickle.dump(vectors_for_learning[2], f)
