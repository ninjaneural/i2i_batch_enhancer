import sys
from scenedetect import detect, ContentDetector

print(sys.argv[1])
scene_list = detect(sys.argv[1], ContentDetector())
for i, scene in enumerate(scene_list):
    print(
        "    Scene %2d: Start %s / Frame %d, End %s / Frame %d"
        % (
            i + 1,
            scene[0].get_timecode(),
            scene[0].get_frames(),
            scene[1].get_timecode(),
            scene[1].get_frames(),
        )
    )
