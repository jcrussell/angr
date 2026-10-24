"""Copy-on-write handles naming the Rust state behind proxy-backed SimStates.

A SimState whose ``memory`` / ``registers`` / ``solver`` / ``callstack``
plugins are Rust proxies reads and writes one Rust state. ``SimState.copy()``
copies every plugin, and the copy must behave like an angr copy: it reads the
source as of the copy, and its writes never reach the source or a sibling
(angr-mx00p.112). The proxies therefore do not hold a raw ``state_id``; they
hold a :class:`RustStateBinding`, and every plugin of one ``SimState.copy()``
call shares the same new binding (through the copy memo).

A copied binding is *virtual*: it reads through its parent, so a copy that is
never written costs no Rust fork. The first write through a virtual binding
materializes it with ``RustSimState::fork`` of what it reads. A write through
a binding that still has virtual children first moves those children onto a
snapshot fork, so they keep reading the pre-write state. Reads after the
backing Rust state is released raise :class:`RustStateReleasedError`; they
never fall through to another state.

Who mints and releases the forks is the binding's *owner*:

* :class:`CallbackForkScope` owns the bindings of one SimProcedure, syscall or
  Python-VEX-fallback callback state. The pending Rust state is frozen while
  Python runs the callback, so the lazy read-through is exact. When the
  successors are known, :meth:`CallbackForkScope.settle` gives each successor
  its own Rust state, and :meth:`CallbackForkScope.close` drops every fork no
  successor claimed.
* :class:`CopyStashForkOwner` owns bindings outside a callback (a proxy
  constructed from a bare state id, or a successor binding after its callback
  closed). Its forks live in the ``_copies`` stash and are dropped when the
  binding that owns them is garbage collected.

The design and its extension to exported states are described in
``docs/advanced-topics/rust_proxy_writes_design.rst`` ("Copy semantics").
"""

from __future__ import annotations

import logging
import weakref

from angr.errors import SimStateError

l = logging.getLogger(name=__name__)

#: Holding stash for the forks a callback's copies materialize. A fork leaves
#: it when a successor claims it (adopted as the resumed state, or finished
#: by ``resume_forked_successor``) or when the scope closes and drops it.
CALLBACK_FORK_STASH = "_cb_forks"

_MEMO_KEY = "rust-state-binding"


class RustStateReleasedError(SimStateError):
    """The Rust state behind a proxy-backed SimState is gone.

    Raised by every proxy read or write through a binding whose state was
    released: an unclaimed callback copy after its callback finished, or a
    copy whose owner was closed. Never caught to fall back to another state.
    """


class RustStateBinding:
    """The Rust state one proxy-backed SimState reads and writes.

    ``state_id`` is the id reads go to. ``writable_id()`` is the id a write
    goes to, materializing a virtual binding first. See the module doc.
    """

    __slots__ = ("__weakref__", "_children", "_id", "_owner", "_parent", "_released")

    def __init__(self, owner, state_id: int | None, parent: RustStateBinding | None = None):
        self._owner = owner
        self._id = state_id
        self._parent = parent
        self._children: weakref.WeakSet | None = None
        self._released: str | None = None

    # ------------------------------------------------------------------
    # Reads and writes
    # ------------------------------------------------------------------

    @property
    def state_id(self) -> int:
        """The Rust state id reads through this binding go to."""
        b = self
        while True:
            if b._released is not None:
                raise RustStateReleasedError(b._released)
            if b._parent is None:
                return b._id
            b = b._parent

    @property
    def raw_id(self) -> int | None:
        """The id this binding owns, or ``None`` while it is virtual.

        For diagnostics and identity checks only: never raises, never resolves.
        """
        return self._id

    @property
    def is_virtual(self) -> bool:
        return self._parent is not None

    @property
    def owner(self):
        return self._owner

    def writable_id(self) -> int:
        """The Rust state id a write through this binding goes to.

        Materializes a virtual binding (a fork of what it reads) and moves
        virtual children onto a snapshot first, so neither the parent nor a
        sibling nor a child sees the write.
        """
        if self._released is not None:
            raise RustStateReleasedError(self._released)
        if self._parent is not None:
            self._materialize()
        if self._children:
            self._shelter_children()
        return self._id

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self, memo=None) -> RustStateBinding:
        """The binding a ``SimState.copy()`` of this state's plugins shares.

        Every proxy plugin of one ``SimState.copy()`` call receives the same
        new binding through ``memo``, so the copy's memory, registers and
        solver keep naming one Rust state.
        """
        if memo is not None:
            key = (_MEMO_KEY, id(self))
            child = memo.get(key)
            if child is None:
                child = memo[key] = self._child()
            return child
        return self._child()

    def _child(self) -> RustStateBinding:
        child = RustStateBinding(self._owner, None, parent=self)
        if self._released is not None:
            # A copy of a released state is released too; the copy itself is
            # allowed (SimState plumbing copies frames whose proxies it then
            # replaces) but any read or write through it raises.
            child._released = self._released
            return child
        if self._children is None:
            self._children = weakref.WeakSet()
        self._children.add(child)
        return child

    # ------------------------------------------------------------------
    # Internals shared with the owners
    # ------------------------------------------------------------------

    def _materialize(self) -> None:
        """Give a virtual binding a fork of the state it reads."""
        source = self.state_id
        new_id = self._owner.fork(source)
        self._detach_from_parent()
        self._id = new_id
        self._owner.track(self)

    def _detach_from_parent(self) -> None:
        parent = self._parent
        if parent is not None and parent._children is not None:
            parent._children.discard(self)
        self._parent = None

    def _shelter_children(self) -> None:
        """Move live virtual children onto a snapshot fork before a write."""
        children = [c for c in self._children if c._released is None]
        self._children = None
        if not children:
            return
        snapshot = RustStateBinding(self._owner, self._owner.fork(self._id))
        self._owner.track(snapshot)
        snapshot._children = weakref.WeakSet(children)
        for child in children:
            child._parent = snapshot

    def _take_over(self, holder: RustStateBinding) -> None:
        """Make this virtual binding the owner of ``holder``'s id.

        ``holder`` is the nearest materialized ancestor, whose id this binding
        reads. The edge is inverted rather than forked: ``holder`` becomes a
        virtual child of this binding, so every binding that read through it
        still reads the same unwritten state, now through this one.
        """
        state_id = holder._id
        self._detach_from_parent()
        self._id = state_id
        if holder._parent is not None:  # pragma: no cover - holder is materialized by construction
            raise AssertionError("take_over needs a materialized holder")
        holder._id = None
        holder._parent = self
        if self._children is None:
            self._children = weakref.WeakSet()
        self._children.add(holder)
        self._owner.retrack(holder, self)

    def _holder(self) -> RustStateBinding:
        """The nearest materialized binding on this binding's read path."""
        b = self
        while b._parent is not None:
            b = b._parent
        return b

    def _release(self, reason: str) -> None:
        self._released = reason

    def __repr__(self):
        state = "released" if self._released is not None else ("virtual" if self._parent is not None else "own")
        return f"<RustStateBinding {state} id={self._id}>"


class _NoForkOwner:
    """Owner for a binding constructed without a Python manager.

    Reads and writes through the root work; a write through a *copy* needs a
    fork and there is no manager to own one, so it raises.
    """

    def fork(self, source_id):
        raise NotImplementedError(
            f"cannot copy-on-write Rust state {source_id}: this proxy was built without a "
            "RustExplorationManager (python_mgr), so nothing can own the fork"
        )

    def track(self, binding):
        pass

    def retrack(self, old, new):
        pass


_NO_FORK_OWNER = _NoForkOwner()


def _drop_copy(mgr_ref, state_id):
    mgr = mgr_ref()
    if mgr is not None:
        mgr.drop_copy(state_id)


class CopyStashForkOwner:
    """Owner for bindings outside a callback scope.

    Forks go to the ``_copies`` stash through
    ``RustExplorationManager.fork_state_for_copy`` and are dropped when the
    binding owning them is garbage collected, the same lifetime as
    ``RustStateProxy.copy()``.
    """

    def __init__(self, python_mgr):
        self._mgr_ref = weakref.ref(python_mgr)

    def fork(self, source_id):
        mgr = self._mgr_ref()
        if mgr is None:
            raise RustStateReleasedError(f"the manager owning Rust state {source_id} is gone")
        return mgr.fork_state_for_copy(source_id)

    def track(self, binding):
        weakref.finalize(binding, _drop_copy, self._mgr_ref, binding._id)

    def retrack(self, old, new):  # pragma: no cover - only callback scopes invert edges
        raise AssertionError("CopyStashForkOwner bindings are never taken over")


def unscoped_binding(python_mgr, state_id: int) -> RustStateBinding:
    """A root binding for ``state_id`` outside any callback scope."""
    owner = getattr(python_mgr, "_copy_stash_fork_owner", None) if python_mgr is not None else None
    return RustStateBinding(owner if owner is not None else _NO_FORK_OWNER, state_id)


def binding_of(state) -> RustStateBinding | None:
    """The binding shared by a SimState's Rust proxy plugins.

    Returns ``None`` when the state has no Rust proxy plugin. Raises when two
    proxy plugins of the same state name different bindings: such a state's
    memory and registers would belong to different Rust states.
    """
    found = None
    for name in ("memory", "registers", "solver", "callstack"):
        plugin = state.plugins.get(name)
        binding = getattr(plugin, "_binding", None)
        if not isinstance(binding, RustStateBinding):
            continue
        if found is None:
            found = binding
        elif binding is not found:
            raise SimStateError(
                f"state {state!r}: its {name} proxy is bound to a different Rust state than its other proxies"
            )
    return found


class CallbackForkScope:
    """Owner of the bindings of one callback state (angr-mx00p.112).

    ``root`` names the pending callback state. Copies of the callback state
    are virtual bindings under it; writes materialize forks into
    :data:`CALLBACK_FORK_STASH`. :meth:`settle` hands each successor its own
    Rust state and releases every other binding; :meth:`close` drops what is
    left. The lazy read-through is exact only because Rust does not touch
    the pending state while the callback runs.
    """

    def __init__(self, python_mgr, pending_id: int):
        self._mgr = python_mgr
        self.pending_id = pending_id
        self.root = RustStateBinding(self, pending_id)
        # Bindings that own a fork in CALLBACK_FORK_STASH, by fork id.
        self._forks: dict[int, RustStateBinding] = {}
        self._claimed: dict[int, RustStateBinding] = {}
        # Claimed fork ids still parked in CALLBACK_FORK_STASH, waiting for
        # adoption or resume_forked_successor.
        self._unconsumed: set[int] = set()
        self._settled = False
        self._closed = False

    @property
    def fork_count(self) -> int:
        return len(self._forks)

    # Owner protocol -----------------------------------------------------

    def fork(self, source_id: int) -> int:
        if self._closed or self._settled:
            raise RustStateReleasedError(
                f"callback for Rust state {self.pending_id} has settled its successors; "
                "a copy of its state can no longer fork"
            )
        return self._mgr.fork_state_for_copy(source_id, stash=CALLBACK_FORK_STASH)

    def track(self, binding: RustStateBinding) -> None:
        self._forks[binding._id] = binding

    def retrack(self, old: RustStateBinding, new: RustStateBinding) -> None:
        state_id = new._id
        if self._forks.get(state_id) is old:
            self._forks[state_id] = new

    # Successors ---------------------------------------------------------

    def settle(self, successors) -> tuple[int, list[int]]:
        """Give every successor its own Rust state; release everything else.

        ``successors[0]`` resumes the pending callback; the rest become extra
        states. Returns ``(resumed_id, extra_ids)``. ``resumed_id`` is either
        the pending id (the first successor reads the pending state
        unchanged) or a fork the caller adopts with
        ``adopt_pending_callback_state``. Every extra id is a fork in
        :data:`CALLBACK_FORK_STASH` for ``resume_forked_successor``.

        A successor that still reads through a shared state takes it over
        (when nothing else claims it) or gets a private fork of it. Every
        binding no successor claimed is released and its fork dropped, so a
        stale copy raises instead of reading a state the resume is about to
        change. Must run before anything writes a Rust state directly.
        """
        if self._settled or self._closed:
            raise RuntimeError(f"callback for Rust state {self.pending_id} already settled its successors")
        bindings = []
        for succ in successors:
            binding = binding_of(succ)
            if binding is None or binding._owner is not self:
                raise SimStateError(
                    f"callback successor {succ!r} is not a copy of the callback state for Rust state "
                    f"{self.pending_id}; it has no Rust state to resume"
                )
            bindings.append(binding)
        resumed_id = self._claim(bindings[0], allow_pending=True)
        extra_ids = [self._claim(b, allow_pending=False) for b in bindings[1:]]
        self._settled = True
        self._unconsumed = {sid for sid in self._claimed if sid != self.pending_id}
        self._release_unclaimed()
        return resumed_id, extra_ids

    def _claim(self, binding: RustStateBinding, *, allow_pending: bool) -> int:
        target = binding.state_id
        if target in self._claimed or (target == self.pending_id and not allow_pending):
            # Another successor owns what this one reads (or it reads the
            # pending state, which only the first successor may become): give
            # it a private fork. Nothing has been written since the copy, so
            # the fork is exactly what it reads.
            new_id = self.fork(target)
            binding._detach_from_parent()
            binding._id = new_id
            self.track(binding)
        elif binding._parent is not None:
            binding._take_over(binding._holder())
        self._claimed[binding._id] = binding
        return binding._id

    def consumed(self, state_id: int) -> None:
        """A claimed fork left :data:`CALLBACK_FORK_STASH` (adopted or routed)."""
        self._unconsumed.discard(state_id)

    def _release_unclaimed(self) -> None:
        reason = (
            f"copy of the callback state for Rust state {self.pending_id} that is not one of its "
            "successors, released when the callback settled"
        )
        claimed = set(map(id, self._claimed.values()))
        for state_id, binding in list(self._forks.items()):
            if id(binding) in claimed or binding._id != state_id:
                continue
            binding._release(reason)
            del self._forks[state_id]
            self._drop(state_id)
        if id(self.root) not in claimed and self.root._id is not None:
            # The first successor owns a fork that replaces the pending state.
            self.root._release(reason)
        self._release_virtuals(reason)

    def _release_virtuals(self, reason: str) -> None:
        # A virtual binding left at this point reads a state the resume is
        # about to change (or drop): release it rather than let it read on.
        stack = [self.root, *self._forks.values(), *self._claimed.values()]
        seen = set()
        while stack:
            b = stack.pop()
            if id(b) in seen:
                continue
            seen.add(id(b))
            children = b._children
            b._children = None
            if not children:
                continue
            for child in list(children):
                if child._parent is not None:
                    child._release(reason)
                stack.append(child)

    def _drop(self, state_id: int) -> None:
        if not self._mgr.drop_copy(state_id, stash=CALLBACK_FORK_STASH):
            raise RuntimeError(f"callback fork {state_id} was not in {CALLBACK_FORK_STASH!r} when released")

    def close(self) -> None:
        """Drop every fork still parked, release the copies, hand off the rest.

        Without a :meth:`settle` (the callback deadended, errored, or resumed
        the pending state in place) every copy is released and its fork
        dropped; the root keeps naming the pending state. After a settle,
        claimed forks that never left the holding stash (the handler failed
        before finishing them) are dropped too. Claimed bindings, which now
        name live states, are handed to the manager's
        :class:`CopyStashForkOwner`. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        reason = f"copy of the callback state for Rust state {self.pending_id}, released when the callback finished"
        if not self._settled:
            for state_id, binding in list(self._forks.items()):
                if binding._id == state_id:
                    binding._release(reason)
                    self._drop(state_id)
            self._forks.clear()
            self._release_virtuals(reason)
            self._claimed[self.pending_id] = self.root
        for state_id in sorted(self._unconsumed):
            self._claimed.pop(state_id)._release(reason)
            self._drop(state_id)
        self._unconsumed.clear()
        owner = getattr(self._mgr, "_copy_stash_fork_owner", None) or _NO_FORK_OWNER
        for binding in self._claimed.values():
            binding._owner = owner
        self._forks.clear()
        self._claimed.clear()
