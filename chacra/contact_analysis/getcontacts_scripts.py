"""Thin shims that forward to the pip-installed getcontacts CLI entry points.

getcontacts is now an installable package (pip install git+https://github.com/Dan-Burns/getcontacts).
Each command (get-dynamic-contacts, get-static-contacts, …) is registered as a
console_scripts entry point by that package and will be on PATH when the
environment is active.  These shims exist solely so that ChACRA's own
[project.scripts] entries can delegate to them.
"""

import subprocess
import sys
from collections.abc import Callable


def _create_command_runner(command_name: str) -> Callable[[], None]:
    """Return a zero-argument callable that invokes *command_name* as a subprocess.

    The command must be available on PATH (i.e. getcontacts must be installed).
    """
    def runner() -> None:
        subprocess.run([command_name, *sys.argv[1:]], check=False)

    runner.__name__ = command_name.replace("-", "_")
    return runner


get_contact_bridges = _create_command_runner("get-contact-bridges")
get_contact_fingerprints = _create_command_runner("get-contact-fingerprints")
get_contact_flare = _create_command_runner("get-contact-flare")
get_contact_frequencies = _create_command_runner("get-contact-frequencies")
get_contact_singleframe = _create_command_runner("get-contact-singleframe")
get_contact_ticc = _create_command_runner("get-contact-ticc")
get_contact_trace = _create_command_runner("get-contact-trace")
get_dynamic_contacts = _create_command_runner("get-dynamic-contacts")
get_resilabels = _create_command_runner("get-resilabels")
get_static_contacts = _create_command_runner("get-static-contacts")
