# apgcc/tracker.py
# Simple point tracker using Hungarian matching for temporal consistency.
# Associates detections across frames by spatial proximity.

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class Track:
    """Represents a single tracked point (person)."""
    _next_id = 1

    def __init__(self, position):
        self.id = Track._next_id
        Track._next_id += 1
        self.position = np.array(position, dtype=np.float32)  # Current (x, y)
        self.history = [self.position.copy()]                  # Trail of past positions
        self.age = 1                   # Total frames this track has existed
        self.hits = 1                  # Total number of successful associations
        self.time_since_update = 0     # Frames since last matched detection
        self.color = self._generate_color()

    def _generate_color(self):
        """Generate a stable, vivid color from the track ID."""
        # Use golden ratio hue spacing for visually distinct colors
        hue = (self.id * 0.618033988749895) % 1.0
        # Convert HSV to RGB (s=0.9, v=0.95 for vivid colors)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))

    def update(self, position):
        """Update track with a new matched detection."""
        self.position = np.array(position, dtype=np.float32)
        self.history.append(self.position.copy())
        # Keep only last N positions for trail drawing
        if len(self.history) > 30:
            self.history.pop(0)
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        """Called when no detection is matched to this track in a frame."""
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def reset_id_counter():
        Track._next_id = 1


class SimplePointTracker:
    """
    Associates point detections across video frames using Hungarian matching.

    Args:
        max_distance (float): Maximum L2 distance (in pixels) to associate
                              a detection with an existing track.
        max_age (int): Number of consecutive missed frames before a track
                       is deleted.
        min_hits (int): Minimum number of hits before a track is considered
                        confirmed (drawn on screen).
    """
    def __init__(self, max_distance=50.0, max_age=5, min_hits=2):
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        Track.reset_id_counter()

    def update(self, detections):
        """
        Update tracker with new frame detections.

        Args:
            detections: numpy array of shape (N, 2) with (x, y) coordinates,
                        or empty array if no detections.

        Returns:
            List of confirmed Track objects (with persistent IDs and colors).
        """
        if len(detections) == 0:
            # No detections: mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            self._remove_stale_tracks()
            return [t for t in self.tracks if t.hits >= self.min_hits]

        detections = np.array(detections, dtype=np.float32)

        if len(self.tracks) == 0:
            # No existing tracks: create new ones for all detections
            for det in detections:
                self.tracks.append(Track(det))
            return [t for t in self.tracks if t.hits >= self.min_hits]

        # ---- Hungarian Matching ----
        # Build cost matrix: L2 distance between each track and each detection
        track_positions = np.array([t.position for t in self.tracks])
        cost_matrix = np.linalg.norm(
            track_positions[:, np.newaxis, :] - detections[np.newaxis, :, :],
            axis=2
        )  # shape: (num_tracks, num_detections)

        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Determine matched, unmatched tracks, and unmatched detections
        matched_tracks = set()
        matched_detections = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.max_distance:
                self.tracks[row].update(detections[col])
                matched_tracks.add(row)
                matched_detections.add(col)

        # Mark unmatched tracks as missed
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for j in range(len(detections)):
            if j not in matched_detections:
                self.tracks.append(Track(detections[j]))

        # Remove tracks that have been missing for too long
        self._remove_stale_tracks()

        # Return only confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _remove_stale_tracks(self):
        """Remove tracks that haven't been updated for max_age frames."""
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def get_active_count(self):
        """Return the number of currently active confirmed tracks."""
        return len([t for t in self.tracks
                    if t.hits >= self.min_hits and t.time_since_update == 0])
