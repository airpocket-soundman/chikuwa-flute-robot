"""Input layout shared by the PyTorch performers and their NumPy runtime (no torch here, the UNO Q has none)."""

PITCH_CENTER, PITCH_SCALE = 1300.0, 600.0           # normalised pitch = (cents - centre) / scale
WINDOW = (0, 3, 6, 10, 15, 20, 30, 40, 55, 70)      # steps ahead of the target track the planner sees
SONG_LOOKAHEAD = (0, 5, 10, 20)                     # steps ahead of the song memory that are read
PREV_LOOK = (0, 3, 6, 10, 15, 20, 30)                # steps ahead read from the previous play's record (practice)
