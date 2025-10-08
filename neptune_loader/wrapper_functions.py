# Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
# Licensed under the Amazon Software License (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at https://aws.amazon.com/asl/ for more details.

"""Wrapper Functions.

Includes Wrapper functions for:
1. Removing white spaces
2. Exception Handling
"""

import logging

logger = logging.getLogger("log smart symptoms tree creation")


def on_failure(errReason: str, value=None):
    """Exception Handling in called Function.

    Args:
        value: Return Val if required.
        errReason: String value of Error description
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                logger.error(errReason)
                logger.error(f"Exception Occurred!!: {e}")
                raise Exception(f"Exception Occurred!!: {e}")

        return wrapper

    return decorator
