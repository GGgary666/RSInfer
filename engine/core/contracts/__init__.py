"""Minimal data and calls actually exchanged across unit or process boundaries.

Private object layouts, policies and helper algorithms belong to their owner, not here.
Shared definitions live outside unit directories so consumers can import them without
triggering manifest.cross_unit_import.
An agent may change an interface and all its consumers together. Team PR review is a
collaboration workflow, never a runtime human-approval requirement. No re-exports.
"""
