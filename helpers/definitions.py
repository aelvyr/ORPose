cam_names = []

for i in range(1, 12):
    cam_names.append('gopro' + str(i))

num_cameras = len(cam_names)
num_keypoints = 26
num_keypoints_hands = 21
num_wholebody_keypoints = 17  # Number of keypoints in the COCO wholebody body model
hidden_keypoints = []

# Constants
# Original 26-keypoint body model pairs
KEYPOINT_PAIRS = [
    [
        15,
        13
    ],
    [
        13,
        11
    ],
    [
        11,
        19
    ],
    [
        16,
        14
    ],
    [
        14,
        12
    ],
    [
        12,
        19
    ],
    [
        17,
        18
    ],
    [
        18,
        19
    ],
    [
        18,
        5
    ],
    [
        5,
        7
    ],
    [
        7,
        9
    ],
    [
        18,
        6
    ],
    [
        6,
        8
    ],
    [
        8,
        10
    ],
    [
        1,
        2
    ],
    [
        0,
        1
    ],
    [
        0,
        2
    ],
    [
        1,
        3
    ],
    [
        2,
        4
    ],
    [
        3,
        5
    ],
    [
        4,
        6
    ],
    [
        15,
        20
    ],
    [
        15,
        22
    ],
    [
        15,
        24
    ],
    [
        16,
        21
    ],
    [
        16,
        23
    ],
    [
        16,
        25
    ]
]

# Wholebody 17-keypoint body model pairs based on COCO format
# The keypoints are:
# 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear, 5:left_shoulder, 6:right_shoulder, 
# 7:left_elbow, 8:right_elbow, 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip, 
# 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
WHOLEBODY_KEYPOINT_PAIRS = [
    [0, 1],  # nose to left eye
    [0, 2],  # nose to right eye
    [1, 3],  # left eye to left ear
    [2, 4],  # right eye to right ear
    [0, 5],  # nose to left shoulder
    [0, 6],  # nose to right shoulder
    [5, 6],  # left to right shoulder
    [5, 7],  # left shoulder to left elbow
    [7, 9],  # left elbow to left wrist
    [6, 8],  # right shoulder to right elbow
    [8, 10], # right elbow to right wrist
    [5, 11],  # left shoulder to left hip
    [6, 12],  # right shoulder to right hip
    [11, 12], # left hip to right hip
    [11, 13], # left hip to left knee
    [13, 15], # left knee to left ankle
    [12, 14], # right hip to right knee
    [14, 16]  # right knee to right ankle
]

KEYPOINT_PAIRS_HANDS = [
    [
        0,
        1
    ],
    [
        1,
        2
    ],
    [
        2,
        3
    ],
    [
        3,
        4
    ],
    [
        0,
        5
    ],
    [
        5,
        6
    ],
    [
        6,
        7
    ],
    [
        7,
        8
    ],
    [
        0,
        9
    ],
    [
        9,
        10
    ],
    [
        10,
        11
    ],
    [
        11,
        12
    ],
    [
        0,
        13
    ],
    [
        13,
        14
    ],
    [
        14,
        15
    ],
    [
        15,
        16
    ],
    [
        0,
        17
    ],
    [
        17,
        18
    ],
    [
        18,
        19
    ],
    [
        19,
        20
    ]
]
# KEYPOINT_PAIRS = [
#     (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),        # Upper body
#     (1, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),          # Left leg
#     (22, 23),                                             
#     (8, 12), (12, 13), (13, 14), (14, 20), (19, 20), (14, 21),      # Right leg
#     (0, 15), (0, 16), (15, 17), (16, 18),                            # Head/neck
#     (7, 26), (26, 27), (27, 28), (28, 29),                         # Left Hand
#     (7, 30), (30, 31), (31, 32), (32, 33),
#     (7, 34), (34, 35), (35, 36), (36, 37),
#     (7, 38), (38, 39), (39, 40), (40, 41),
#     (7, 42), (42, 43), (43, 44), (44, 45),
#     (4, 47), (47, 48), (48, 49), (49, 50),                         # Right Hand
#     (4, 51), (51, 52), (52, 53), (53, 54),
#     (4, 55), (55, 56), (56, 57), (57, 58),
#     (4, 59), (59, 60), (60, 61), (61, 62),
#     (4, 63), (63, 64), (64, 65), (65, 66)
# ]


# Fixed lengths between keypoints (based on known anatomy or average values)
FIXED_LENGTHS = {
    (3,4):0.3,
    (2,3):0.32,
    (1,2):0.21,
    (10,11):0.48,
    (9,10):0.5,
    (6,7):0.3,
    (5,6):0.32,
    (1,5):0.21,
    (13,14):0.48,
    (12,13):0.5,
    (0, 1):0.22
}

CANONICAL_HAND_POSE_3D = [
    [0.0, 0.0, 0.0],       # palm center (0)
    [0.031, 0.022, 0.0],       # thumb base (1)
    [0.051, 0.041, 0.01],     # thumb mid (2)
    [0.074, 0.077, 0.015],      # thumb tip (3)
    [0.088, 0.101, 0.02],     # thumb end (4)
    [0.025, 0.098, 0.0],     # index base (5)
    [0.0375, 0.14, 0.0],    # index mid (6)
    [0.043, 0.165, 0.0],    # index tip (7)
    [0.047, 0.185, 0.0],    # index end (8)
    [0.0, 0.098, 0.0],     # middle base (9)
    [0.006, 0.149, 0.0],    # middle mid (10)
    [0.008, 0.178, 0.0],    # middle tip (11)
    [0.009, 0.203, 0.0],    # middle end (12)
    [-0.021, 0.093, 0.0],      # ring base (13)
    [-0.029, 0.146, 0.0],     # ring mid (14)
    [-0.034, 0.175, 0.0],     # ring tip (15)
    [-0.036, 0.198, 0.0],     # ring end (16)
    [-0.041, 0.083, 0.0],      # pinky base (17)
    [-0.06, 0.122, 0.0],     # pinky mid (18)
    [-0.0685, 0.141, 0.0],     # pinky tip (19)
    [-0.075, 0.159, 0.0]      # pinky end (20)
]