import numpy as np

class PersistenceTracker:
    def __init__(self, min_persistence=3):
        self.min_persistence = min_persistence
        # specific to 3 frames: needs 2 consecutive matches (t-2 -> t-1, t-1 -> t)
        # We store the matches array: matches[i] = j means keypoint i in Source maps to j in Target.
        self.matches_history = []

    def update(self, matches_src_to_dst):
        """
        Updates the tracker with matches from Frame(t-1) to Frame(t).

        Args:
            matches_src_to_dst: Array of shape (N_src,), containing indices in dst, or -1.

        Returns:
            valid_indices_curr: Indices in Frame(t) that are part of a persistent chain.
            valid_indices_prev: Corresponding indices in Frame(t-1).
        """
        # Ensure numpy array
        matches = np.array(matches_src_to_dst)

        self.matches_history.append(matches)

        # We need at least (min_persistence - 1) sets of matches to link N frames.
        # For N=3, we need matches (t-2->t-1) and (t-1->t).
        if len(self.matches_history) < (self.min_persistence - 1):
            return np.array([]), np.array([])

        # If we have more history than needed, trim
        if len(self.matches_history) > (self.min_persistence - 1):
            self.matches_history.pop(0)

        # Now we check the chain.
        # Let's say min_persistence=3.
        # matches_history[0] is M_{t-2 -> t-1}
        # matches_history[1] is M_{t-1 -> t}

        # Start with indices in t-2 (simply range(len(matches_0)))
        # We want to find paths that survive.

        # M1: t-2 -> t-1
        m1 = self.matches_history[0]
        # M2: t-1 -> t
        m2 = self.matches_history[1]

        # Valid indices in t-1 coming from t-2
        # indices i in t-2 map to m1[i] in t-1.
        # We only care about m1[i] != -1.

        valid_t_minus_2 = np.where(m1 > -1)[0]
        indices_in_t_minus_1 = m1[valid_t_minus_2]

        # Now check if these indices in t-1 map to valid indices in t using m2.
        # Note: indices_in_t_minus_1 might be larger than len(m2) if keypoint counts change?
        # Usually LightGlue outputs matches for the *source* keypoints.
        # So m2 corresponds to keypoints in t-1.
        # We must ensure indices_in_t_minus_1 are within bounds of m2.

        valid_mask = indices_in_t_minus_1 < len(m2)
        indices_in_t_minus_1 = indices_in_t_minus_1[valid_mask]

        # Now look up in m2
        indices_in_t = m2[indices_in_t_minus_1]

        # Filter for valid matches in t
        final_valid_mask = indices_in_t > -1

        valid_indices_curr = indices_in_t[final_valid_mask]
        valid_indices_prev = indices_in_t_minus_1[final_valid_mask]

        return valid_indices_curr, valid_indices_prev

    def compute_inlier_ratio(self, matches):
        """
        Computes the ratio of valid matches to total keypoints.
        """
        matches = np.array(matches)
        num_matches = np.sum(matches > -1)
        total = len(matches)
        if total == 0:
            return 0.0
        return num_matches / total
