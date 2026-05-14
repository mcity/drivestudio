"""Probe an mcap bag's /tf and /tf_static topics to find sensor frame relationships.

Usage:
    python tools/probe_mcity_tf.py <bag.mcap>
"""
import sys
from collections import defaultdict
from mcap_ros2.reader import read_ros2_messages


def main(bag_path: str):
    edges = defaultdict(list)
    seen_messages = defaultdict(int)
    seen_frame_ids = set()
    seen_child_frame_ids = set()

    for msg in read_ros2_messages(
        bag_path,
        topics=["/tf", "/tf_static"],
    ):
        seen_messages[msg.channel.topic] += 1
        for tr in msg.ros_msg.transforms:
            edges[(tr.header.frame_id, tr.child_frame_id)].append(msg.channel.topic)
            seen_frame_ids.add(tr.header.frame_id)
            seen_child_frame_ids.add(tr.child_frame_id)
        if sum(seen_messages.values()) > 5000:
            break

    print("=== topic counts (first 5000 msgs) ===")
    for t, c in seen_messages.items():
        print(f"  {t}: {c}")

    print("\n=== unique edges (parent -> child : topic) ===")
    for (p, c), topics in sorted(edges.items()):
        topic_set = ",".join(sorted(set(topics)))
        print(f"  {p} -> {c}  [{topic_set}]")

    print("\n=== frames mentioned as parent ===")
    print(sorted(seen_frame_ids))
    print("\n=== frames mentioned as child ===")
    print(sorted(seen_child_frame_ids))


if __name__ == "__main__":
    main(sys.argv[1])
