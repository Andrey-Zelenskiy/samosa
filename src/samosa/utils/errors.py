#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
======================
samosa/utils/errors.py
======================

This script defines custom methods error handling.
"""


def custom_format_warning(msg, *args, **kwargs):
    """
    When throwing a warning, only throw the message
    """
    return str(msg) + "\n"
