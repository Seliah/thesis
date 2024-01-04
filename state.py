import os

terminating = False
"""Flag to quit long running processes."""

service = os.getenv("RUN_AS_SERVICE", None) == "True"
"""Flag that state whether this program is run as a systemd service unit."""
