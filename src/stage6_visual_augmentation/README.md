# Stage 6 - Visual augmentation

Draws a picture for each sound that is shown (`generate_augmentations()`; Qwen-Image-2512 in the final system) and
places the pictures beside the video while their sound plays (`composite_alongside()`), both in `__init__.py`.
`verify.py` checks each picture with the vision model and redraws it if it shows the wrong thing (`sense.py` is a
generic variant of that check, off in the final system);
`group.py` merges close repeats of one sound into one picture; `depict.py` drops a picture whose event is visibly
happening on screen.
