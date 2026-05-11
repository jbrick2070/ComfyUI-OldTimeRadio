"""Config package for OTR.

Houses static lookup tables and pools (cast names, voice profiles,
trait pools) that callers in nodes/ import. Keeping these out of
nodes/ makes them easy to find without grepping prompt strings.
"""
