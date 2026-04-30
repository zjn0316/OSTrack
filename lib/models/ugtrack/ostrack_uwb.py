from lib.models.ugtrack.ugtrack import UGTrack, build_ugtrack


class OSTrackUWB(UGTrack):
    """Compatibility wrapper for the old OSTrack-UWB model import path."""


def build_ostrack_uwb(cfg, training=True):
    return build_ugtrack(cfg, training=training)


OSTrack_UWB = OSTrackUWB
