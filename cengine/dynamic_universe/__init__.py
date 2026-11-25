"""Dynamic universe management & opportunity rotation (selection/rotation layer).

Pure decision core: config, permanent baseline registry, 4-layer model,
candidate lifecycle, and the rotation scheduler. No socket ownership, no second
data connection — the active universe computed here is fed to the existing
subscription owner (universe_sync / adaptive controller) by the manager layer.
"""
