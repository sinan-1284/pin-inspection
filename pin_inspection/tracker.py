"""Turn a stream of per-frame results into one inspection per pin.

A pin is counted once it has sat still for a few frames, and the view must
clear (no pin) before the next one counts, so a pin left under the camera
isn't counted over and over. Blobs far from pin-sized (a hand, the paper
edge) are reported as OBSTRUCTED and never counted.
"""


class PinTracker:
    def __init__(self, cfg):
        st = cfg.get("station", {})
        self.stable_frames = st.get("stable_frames", 3)
        self.stable_tol = st.get("stable_tol_mm", 0.3)
        self.clear_frames = st.get("clear_frames", 2)
        lo, hi = cfg["inspect"]["tolerances"]["length"]
        self.nominal = (lo + hi) / 2
        self.lengths = []
        self.clear = self.clear_frames  # start ready to count
        self.counted = False

    def plausible(self, res):
        """Roughly pin-shaped: within half / double the nominal length, thin."""
        mm = res.get("mm") or {}
        length = mm.get("length")
        if length is None:
            return True  # uncalibrated: can't judge, assume pin
        return 0.5 * self.nominal <= length <= 2 * self.nominal and mm.get("shaft_width", 0) < 3

    def update(self, res):
        """Feed one frame's measurements (or None). Returns (state, count_now).

        state is NO_PIN, OBSTRUCTED, SETTLING or PIN; count_now is True on the
        one frame where this pin should be recorded.
        """
        if res is None or not self.plausible(res):
            self.clear += 1
            self.lengths = []
            if self.clear >= self.clear_frames:
                self.counted = False
            return ("NO_PIN" if res is None else "OBSTRUCTED"), False
        self.clear = 0
        length = (res.get("mm") or res["px"])["length"]
        self.lengths = (self.lengths + [length])[-self.stable_frames:]
        steady = len(self.lengths) == self.stable_frames and max(self.lengths) - min(self.lengths) <= self.stable_tol
        if not steady:
            return "SETTLING", False
        if self.counted:
            return "PIN", False
        self.counted = True
        return "PIN", True
