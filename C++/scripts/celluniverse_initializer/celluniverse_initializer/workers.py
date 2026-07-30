from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from qtpy import QtCore


class WorkerSignals(QtCore.QObject):
    returned = QtCore.Signal(str, int, object)
    failed = QtCore.Signal(str, int, str)


class ComputeWorker(QtCore.QRunnable):
    def __init__(
        self,
        kind: str,
        generation: int,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.generation = generation
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            self.signals.failed.emit(
                self.kind,
                self.generation,
                traceback.format_exc(),
            )
            return
        self.signals.returned.emit(self.kind, self.generation, result)
