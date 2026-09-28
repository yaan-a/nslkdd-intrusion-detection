"""Small helpers used to demonstrate automated testing (Issue #2).

This module deliberately has no third-party dependencies so it can be tested
in CI without installing the ML stack or downloading the datasets.
"""


def to_binary_label(class_name):
    """Convert an NSL-KDD "class" value to a binary label.

    Returns 0 for normal traffic and 1 for any attack type. This is the same
    rule export_model.py applies to the "class" column:
    ``class.str.lower() != "normal"``.
    """
    return 0 if class_name.lower() == "normal" else 1
