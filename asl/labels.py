"""The letters the classifier recognizes, and conversions between letters and class indices."""

import string

# J and Z require motion, so they're excluded (see D1 in architecture.md)
EXCLUDED_LETTERS = {"J", "Z"}

LETTERS = [letter for letter in string.ascii_uppercase if letter not in EXCLUDED_LETTERS]

NUM_CLASSES = len(LETTERS)

LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTERS)}