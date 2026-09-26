"""One background display-curve fit, with results owned by the UI thread.

The caller provides an executor's ``submit`` method and calls ``poll`` from its
own event loop. Workers receive independent diagram snapshots, never live model
objects; they publish results only through the completion queue.
"""
from queue import Empty, SimpleQueue

from .diagram import copy_diagram


class DisplayCurveCache:
    """Keep the latest prepared geometry without building a backlog of fits.

    All public methods must run on the caller's thread. ``sync`` notices model
    replacement; callers must use ``invalidate`` after an in-place geometry or
    crossing change. View zoom and colors do not require invalidation.
    """

    def __init__(self):
        self.diagram = None
        self.prepared = None
        self.error = None
        self.revision = 0
        self._inflight = None
        self._completed = SimpleQueue()

    @property
    def inflight(self):
        """Whether a submitted fit is still awaiting collection by ``poll``."""
        return self._inflight is not None

    def sync(self, diagram):
        """Discard cached results when the live diagram object changes."""
        if diagram is self.diagram:
            return False
        self.diagram = diagram
        self.invalidate()
        return True

    def invalidate(self):
        """Mark an in-place change, retaining any worker until it completes."""
        self.revision += 1
        self.prepared = None
        self.error = None

    def request(self, diagram, submit):
        """Snapshot and submit one fit if needed, returning whether submitted.

        An existing worker is allowed to finish even after an edit. Poll its
        completion, then request the newest diagram; intermediate states never
        enter the executor's queue. Failed fits wait for another model change
        instead of retrying on every redraw.
        """
        self.sync(diagram)
        if (diagram is None or self.prepared is not None or
                self.error is not None or self.inflight):
            return False

        # Copy on the model-owning thread before the worker can read anything.
        try:
            snapshot = copy_diagram(diagram)
        except Exception as error:
            self.error = error
            return False
        revision = self.revision
        token = object()
        completed = self._completed

        def fit():
            try:
                from .render import prepare_display_curves
                prepared = prepare_display_curves(snapshot)
            except Exception as error:
                completed.put((token, revision, None, error))
            else:
                completed.put((token, revision, prepared, None))

        self._inflight = token
        try:
            submit(fit)
        except Exception as error:
            self._inflight = None
            self.error = error
            return False
        return True

    def poll(self, diagram):
        """Install current results; return whether ready/error state changed."""
        self.sync(diagram)
        changed = False
        while True:
            try:
                token, revision, prepared, error = self._completed.get_nowait()
            except Empty:
                break
            # A submitter that raises after running synchronously may leave a
            # completion behind. It must not replace that submission's error.
            if token is not self._inflight:
                continue
            self._inflight = None
            if revision != self.revision:
                continue
            self.prepared, self.error = prepared, error
            changed = True
        return changed
