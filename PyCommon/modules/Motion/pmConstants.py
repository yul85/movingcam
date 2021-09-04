from enum import IntEnum

PM_HUMAN_NUM_LINKS = 58


def MaskBit(j):
    assert isinstance(j, int)
    return 0x01 << j


class PmHumanEnum(IntEnum):
    UNDEFINED = -1
    PELVIS = 0
    SPINE_1 = 1
    SPINE_2 = 2
    SPINE_3 = 3
    SPINE_4 = 4
    CHEST = 5
    NECK = 6
    HEAD = 7
    RIGHT_SHOULDER = 8
    LEFT_SHOULDER = 9
    RIGHT_COLLAR = 10
    LEFT_COLLAR = 11
    UPPER_RIGHT_ARM = 12
    UPPER_LEFT_ARM = 13
    LOWER_RIGHT_ARM = 14
    LOWER_LEFT_ARM = 15
    UPPER_RIGHT_LEG = 16
    UPPER_LEFT_LEG = 17
    LOWER_RIGHT_LEG = 18
    LOWER_LEFT_LEG = 19
    RIGHT_FOOT = 20
    LEFT_FOOT = 21
    RIGHT_TOE = 22
    LEFT_TOE = 23
    RIGHT_PALM = 24
    LEFT_PALM = 25
    RIGHT_HEEL = 26
    LEFT_HEEL = 27
    RIGHT_FINGER_11 = 28
    RIGHT_FINGER_12 = 29
    RIGHT_FINGER_13 = 30
    RIGHT_FINGER_21 = 31
    RIGHT_FINGER_22 = 32
    RIGHT_FINGER_23 = 33
    RIGHT_FINGER_31 = 34
    RIGHT_FINGER_32 = 35
    RIGHT_FINGER_33 = 36
    RIGHT_FINGER_41 = 37
    RIGHT_FINGER_42 = 38
    RIGHT_FINGER_43 = 39
    RIGHT_FINGER_51 = 40
    RIGHT_FINGER_52 = 41
    RIGHT_FINGER_53 = 42
    LEFT_FINGER_11 = 43
    LEFT_FINGER_12 = 44
    LEFT_FINGER_13 = 45
    LEFT_FINGER_21 = 46
    LEFT_FINGER_22 = 47
    LEFT_FINGER_23 = 48
    LEFT_FINGER_31 = 49
    LEFT_FINGER_32 = 50
    LEFT_FINGER_33 = 51
    LEFT_FINGER_41 = 52
    LEFT_FINGER_42 = 53
    LEFT_FINGER_43 = 54
    LEFT_FINGER_51 = 55
    LEFT_FINGER_52 = 56
    LEFT_FINGER_53 = 57


human_part_name = [
          "Hips",
          "_Spine1", "_Spine2", "_Spine3", "_Spine4",
    "Spine",  # chest
      "Spine1",  # neck
    "Spine1_Effector",  # head
          "right_shoulder", "left_shoulder",
          "right_collar", "left_collar",
      "RightArm", "LeftArm", "RightForeArm", "LeftForeArm",
      "RightUpLeg", "LeftUpLeg", "RightLeg", "LeftLeg",
      "RightFoot", "LeftFoot",
    "RightFoot_Effector",  # right toe
    "LeftFoot_Effector",   # left toe
      "right_palm", "left_palm",
          "right_heel", "left_heel",
      "right_finger_11", "right_finger_12", "right_finger_13",
      "right_finger_21", "right_finger_22", "right_finger_23",
      "right_finger_31", "right_finger_32", "right_finger_33",
      "right_finger_41", "right_finger_42", "right_finger_43",
      "right_finger_51", "right_finger_52", "right_finger_53",
      "left_finger_11", "left_finger_12", "left_finger_13",
      "left_finger_21", "left_finger_22", "left_finger_23",
      "left_finger_31", "left_finger_32", "left_finger_33",
      "left_finger_41", "left_finger_42", "left_finger_43",
      "left_finger_51", "left_finger_52", "left_finger_53"
]