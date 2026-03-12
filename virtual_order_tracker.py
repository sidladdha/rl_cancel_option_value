"""
virtual_order_tracker.py
========================
Event-Driven Virtual Order Observer — Counterfactual Analysis Edition

Operates in two independent modes controlled by the `mode` argument:

    Standard mode  (mode="standard")
    ---------------------------------
    The live trading simulation.  Cancel rules are ACTIVE:
      • Lax strand-cancel (Kwan & Philip 2015 inspired):
        Replaces the old bare 5-tick BBO-distance rule.
        An order is cancelled only when ALL THREE gates fire simultaneously:
          1. Signal gate     — QueuePressureRatio >= QPR_THRESHOLD (2.0)
          2. Queue-depth gate — q_frac < QPR_QFRAC_GATE (0.20)
                               Orders deeper in the book have "queue insurance"
                               and are left to rest even under pressure.
          3. Persistence gate — QPR has been continuously >= threshold for
                               QPR_PERSIST_NS (10 ms), filtering MBO flicker
                               and spoofing bursts.
        The logic is fully symmetric: identical code runs for side B and side A.
        No `if side == 'A'` branch exists anywhere in the cancel path.

      • 10-tick reposition: unchanged.
      • Post-fill 100 ms grace period: unchanged.

    Shadow mode  (mode="shadow")
    ----------------------------
    The counterfactual — cancel rules are INACTIVE:
      • No strand-cancel fires.
      • No repositioning.
      • QP windows are not populated (zero compute cost).
    Orders rest until they fill naturally or the caller explicitly cancels
    them at session end via cancel_order().

Queue Pressure Signal  (per active virtual order, Standard mode only)
----------------------------------------------------------------------
A rolling QPR_WINDOW_NS (50 ms) window tracks two streams per order:

    Self-Side Depletion (stream A)
        C/M/T/F messages on the order's own side at exactly the order's
        limit_price, while shares_ahead > 0.  Only volume that is strictly
        ahead of this order counts — messages behind it are irrelevant.

    Opposite-Side BBO Growth (stream B)
        ADD messages on the opposite side arriving at the current opposite
        BBO.  A growing contra BBO signals balanced churn, not toxic flow.

    QueuePressureRatio:
        QPR = sum(stream_A) / (sum(stream_B) + QPR_EPSILON)

        QPR >> 1  queue draining fast, contra side quiet → potential toxicity
        QPR ~  1  balanced churn → hold
        QPR << 1  contra BBO growing strongly → very safe to hold

Output — flat list of event dicts, one per terminal event
---------------------------------------------------------
All existing fields are preserved, plus two new fields on every event:
    queue_pressure_ratio : float  QPR at event time (0.0 for Shadow events)
    qp_q_frac_gate       : bool   True if q_frac < QPR_QFRAC_GATE at event time

Design constraints
------------------
* Zero look-ahead bias.
* Column names match create_multiday_data.py exactly.
* OFI = Δbid_size_1 − Δask_size_1 (standard definition).
* Symmetry: no side-specific branching in cancel logic.
"""

from __future__ import annotations

import logging
import uuid
from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Trigger / mode labels ─────────────────────────────────────────────────────
TRIG_FILL         = "FILL"
TRIG_CANCEL       = "CANCEL"
TRIG_BBO_MOVE     = "BBO_MOVE"   # kept for downstream compat; no longer emitted by on_message
TRIG_QUEUE_CHANGE = "QUEUE_CHANGE"

MODE_STANDARD = "standard"
MODE_SHADOW   = "shadow"

# ── Unchanged thresholds ──────────────────────────────────────────────────────
QUEUE_CHANGE_THRESHOLD     = 0.10          # 10% shares_ahead change → emit event
STRAND_TICKS_THRESHOLD     = 5             # kept for reference; no longer drives cancels
REPOSITION_TICKS_THRESHOLD = 10            # mid drift ticks → reposition
POST_FILL_GRACE_NS         = 100_000_000   # 100 ms post-fill grace for mid capture

# ── Lax cancellation parameters ───────────────────────────────────────────────
# Gate 1 — signal threshold
# QPR must reach or exceed this value before the cancel can even be considered.
QPR_THRESHOLD  = 2.0

# Gate 2 — queue-depth gate
# q_frac must be BELOW this value (order is near the front of the queue).
# Orders with q_frac >= QPR_QFRAC_GATE have queue insurance and are never
# cancelled by QPR alone.
QPR_QFRAC_GATE = 0.20

# Gate 3 — persistence (temporal buffer)
# QPR must have been continuously >= QPR_THRESHOLD for this many nanoseconds.
# 10 ms filters single-tick spikes, spoofing flickers, and burst artefacts.
QPR_PERSIST_NS = 10_000_000   # 10 ms

# Rolling window width for QPR calculation.
QPR_WINDOW_NS  = 50_000_000   # 50 ms

# Epsilon in QPR denominator — prevents division by zero.
QPR_EPSILON    = 1.0   # shares

# ── Max order lifetime ─────────────────────────────────────────────────────────
# Force-cancel any standard order that has been live longer than this.
# Rationale: in quiet markets (lunch, pre-close), QPR never fires because the
# rolling window is empty (no messages → no queue changes). Orders accumulate
# stale birth metadata (birth_q_frac, birth_mid) that is no longer meaningful.
# A realistic market maker would cancel and re-place after ~5 minutes to stay
# competitive. This also prevents 200 orders sitting idle for 30+ minutes and
# then all filling at once when activity resumes, which would corrupt the
# fill-rate / sharpe statistics by mixing very-long-lifetime fills with normal ones.
#
# Applied only to standard tracker orders (shadow orders follow the standard order's
# lifecycle and are cleaned up by _retire_std / _close_session).
MAX_LIFETIME_NS = 300_000_000_000   # 300 s = 5 minutes

# Minimum price increment — used to convert float prices to integer tick indices
# for fast equality comparison (int == is much faster than abs(float-float)<eps).
# Override at runtime if your instrument uses a different tick size.
TICK_SIZE = 0.01
_TICK_INV = 1.0 / TICK_SIZE   # multiply instead of divide

def _to_ticks(price: float) -> int:
    """Convert float price to integer tick index."""
    return int(round(price * _TICK_INV))

# Feature names (preserved for downstream compatibility)
STATE_COLS = ["q_frac", "ln_shares_ahead", "OFI", "spread", "ln_order_size"]


# ─────────────────────────────────────────────────────────────────────────────
# Book side  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class _BookSide:
    """
    One side of the limit order book.

    - best() is O(1) — cached, updated incrementally on apply().
    - shares_strictly_ahead() is O(log n) via bisect on a sorted price list
      with a prefix-sum array.  Both are rebuilt lazily when the price set
      changes (_sorted_dirty flag), which costs O(n_levels) once and is then
      reused for every query until the next structural change.
    - top_n_levels() reuses the same sorted list.
    """
    __slots__ = ("_is_bid", "_levels", "_best", "_sorted_dirty", "_shares_dirty",
                 "_sorted_keys", "_prefix_sums", "_neg_keys")

    def __init__(self, is_bid: bool) -> None:
        self._is_bid         = is_bid
        self._levels: Dict[float, int] = {}
        self._best: Optional[float]    = None
        self._sorted_dirty: bool       = True   # key set changed → need re-sort + prefix
        self._shares_dirty: bool        = False  # shares changed only → need prefix only
        self._sorted_keys:  list       = []   # sorted best-first
        self._prefix_sums:  list       = []   # prefix_sums[i] = shares at levels[0..i-1]
        self._neg_keys:     list       = []   # negated keys for bisect (bid only)

    def _recompute_best(self) -> None:
        if not self._levels:
            self._best = None
        else:
            self._best = max(self._levels) if self._is_bid else min(self._levels)

    def _rebuild_sorted(self) -> None:
        """Rebuild sorted key list and/or prefix sums.  Called lazily when dirty."""
        if self._sorted_dirty:
            # Key set changed: full re-sort + prefix sum recomputation
            sk = sorted(self._levels.keys(), reverse=self._is_bid)
            self._sorted_keys = sk
            if self._is_bid:
                self._neg_keys = [-p for p in sk]
            self._sorted_dirty = False
        else:
            # Only shares changed: reuse existing sorted key order
            sk = self._sorted_keys
        # Always recompute prefix sums when called
        pf = [0] * (len(sk) + 1)
        s = 0
        for i, p in enumerate(sk):
            pf[i] = s
            s += self._levels[p]
        pf[len(sk)] = s
        self._prefix_sums = pf
        self._shares_dirty = False

    def _is_dirty(self) -> bool:
        return self._sorted_dirty or self._shares_dirty

    def apply(self, msg_type: str, price: float, shares: int) -> None:
        if msg_type == "A":
            prev = self._levels.get(price, 0)
            self._levels[price] = prev + shares
            if prev == 0:
                # New level: sorted order AND prefix sums change
                self._sorted_dirty = True
                b = self._best
                if b is None:
                    self._best = price
                elif self._is_bid and price > b:
                    self._best = price
                elif not self._is_bid and price < b:
                    self._best = price
            else:
                # Existing level: sorted order unchanged, only prefix sums stale.
                # Use lightweight flag so rebuild skips the sort step.
                self._shares_dirty = True
        elif msg_type in ("C", "M", "T", "F"):
            remaining = self._levels.get(price, 0) - shares
            if remaining <= 0:
                self._levels.pop(price, None)
                self._sorted_dirty = True   # key set changed
                if price == self._best:
                    self._recompute_best()
            else:
                self._levels[price] = remaining
                self._shares_dirty = True   # shares changed, order unchanged

    def best(self) -> Optional[float]:
        return self._best

    def size_at_best(self) -> int:
        b = self._best
        return self._levels[b] if b is not None else 0

    def shares_strictly_ahead(self, price: float) -> int:
        """Shares at all prices strictly better than price.  O(log n)."""
        if self._sorted_dirty or self._shares_dirty:
            self._rebuild_sorted()
        if self._is_bid:
            # neg_keys is ascending; bisect_left finds first pos where -key >= -price
            # i.e. first pos where key <= price → everything before = keys > price
            idx = _bisect_left(self._neg_keys, -price)
            return self._prefix_sums[idx]
        else:
            # sorted_keys is ascending for ask; find first pos >= price
            idx = _bisect_left(self._sorted_keys, price)
            return self._prefix_sums[idx]

    def shares_at(self, price: float) -> int:
        return self._levels.get(price, 0)

    def top_n_levels(self, n: int) -> List[tuple]:
        """Up to n price levels sorted best-first as (price, shares) tuples."""
        if not self._levels:
            return []
        if self._sorted_dirty:
            self._rebuild_sorted()
        # shares_dirty doesn't affect key order — just share counts — which
        # top_n_levels reads directly from _levels, so no rebuild needed.
        keys = self._sorted_keys[:n]
        return [(p, self._levels[p]) for p in keys]


# ─────────────────────────────────────────────────────────────────────────────
# Queue Pressure Window  (new)
# ─────────────────────────────────────────────────────────────────────────────

_QEntry = Tuple[int, float]   # (ts_ns, shares)


class _QueuePressureWindow:
    """
    Per-virtual-order rolling 50 ms window for the QPR signal.

    Two deques hold timestamped (ts_ns, shares) entries:
        _depletion  : self-side C/M/T/F at the order's own price while ahead > 0
        _opp_growth : opposite-side ADD at the current opposite BBO

    Both deques are trimmed lazily on every call to update() or ratio(),
    so amortised cost is O(1) per message.

    Classification is fully symmetric — the caller passes order_side and
    opp_bbo; no side-specific branching lives here.
    """

    __slots__ = ("_dep", "_grow", "_dep_sum", "_grow_sum")

    def __init__(self) -> None:
        self._dep:      Deque[_QEntry] = deque()
        self._grow:     Deque[_QEntry] = deque()
        self._dep_sum:  float = 0.0
        self._grow_sum: float = 0.0

    def update(
        self,
        ts_ns:        int,
        msg_type:     str,
        msg_side:     str,
        msg_price:    float,
        msg_shares:   float,
        order_side:   str,
        order_price:  float,
        shares_ahead: float,
        opp_bbo:      Optional[float],
        msg_ticks:    int = -1,
        order_ticks:  int = -1,
    ) -> None:
        """
        Classify message and append to the correct deque, then expire stale entries.

        Stream A — self-side depletion:
            msg_side == order_side
            AND msg_type in (C, M, T, F)   (removes volume)
            AND msg_price == order_price    (at this order's level)
            AND shares_ahead > 0            (there is queue ahead of this order)

        Stream B — opposite-side BBO growth:
            msg_side != order_side
            AND msg_type == "A"             (adds volume)
            AND opp_bbo is not None
            AND msg_price == opp_bbo        (at the current opposite best)
        """
        s = float(msg_shares)
        # Use integer tick comparison when available (passed as order_ticks/msg_ticks)
        # Falls back to float comparison for backward compat.
        _price_match = (order_ticks == msg_ticks) if (order_ticks >= 0) else (abs(msg_price - order_price) < 1e-9)
        if (msg_side == order_side
                and msg_type in ("C", "M", "T", "F")
                and _price_match
                and shares_ahead > 0):
            self._dep.append((ts_ns, s))
            self._dep_sum += s

        elif (msg_side != order_side
                and msg_type == "A"
                and opp_bbo is not None
                and abs(msg_price - opp_bbo) < 1e-9):
            self._grow.append((ts_ns, s))
            self._grow_sum += s

        # Expire entries older than the window — subtract from running sums
        cutoff = ts_ns - QPR_WINDOW_NS
        while self._dep  and self._dep[0][0]  < cutoff:
            self._dep_sum  -= self._dep.popleft()[1]
        while self._grow and self._grow[0][0] < cutoff:
            self._grow_sum -= self._grow.popleft()[1]

    def ratio(self, ts_ns: int) -> float:
        """Return current QPR after expiring stale entries. O(expired) amortised."""
        cutoff = ts_ns - QPR_WINDOW_NS
        while self._dep  and self._dep[0][0]  < cutoff:
            self._dep_sum  -= self._dep.popleft()[1]
        while self._grow and self._grow[0][0] < cutoff:
            self._grow_sum -= self._grow.popleft()[1]
        return self._dep_sum / (self._grow_sum + QPR_EPSILON)

    def clear(self) -> None:
        self._dep.clear()
        self._grow.clear()
        self._dep_sum  = 0.0
        self._grow_sum = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Virtual order  (three new QP fields added at the end)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _VirtualOrder:
    virtual_id:                str
    birth_ts_ns:               int
    birth_mid:                 float
    birth_bid:                 float
    birth_ask:                 float
    side:                      str
    price:                     float   # current limit price (may be repositioned)
    price_ticks:               int     # price as integer ticks (price/tick_size, rounded)
    shadow_price:              float   # original birth price — NEVER changes
    size:                      int
    shares_ahead:              float
    last_emitted_shares_ahead: float
    repositioned:              bool    # True after first 10-tick reposition
    reposition_ts_ns:          int     # 0 = never repositioned; set once at reposition time
    fill_ts_ns:                int     # 0 = not yet filled
    fill_price:                float
    mid_post_100ms:            float
    # ── Queue Pressure state (Standard mode only) ─────────────────────────
    qp_window:         _QueuePressureWindow = field(default_factory=_QueuePressureWindow)
    qp_above_since_ns: int   = 0    # ts_ns when QPR first crossed threshold; 0 = below
    qp_ratio_last:     float = 0.0  # most recently computed QPR (for event output)


# ─────────────────────────────────────────────────────────────────────────────
# OFI accumulator  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class _OFIAccumulator:
    def __init__(self) -> None:
        self._delta_bid: int = 0
        self._delta_ask: int = 0

    def update(self, side: str, msg_type: str, shares: int) -> None:
        if side == "B":
            if msg_type == "A":
                self._delta_bid += shares
            elif msg_type in ("C", "M", "T", "F"):
                self._delta_bid -= shares
        elif side == "A":
            if msg_type == "A":
                self._delta_ask += shares
            elif msg_type in ("C", "M", "T", "F"):
                self._delta_ask -= shares
        # Any other side value (e.g. 'N' for non-displayed) is silently ignored

    def value(self) -> float:
        return float(self._delta_bid - self._delta_ask)

    def reset(self) -> None:
        self._delta_bid = 0
        self._delta_ask = 0


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker
# ─────────────────────────────────────────────────────────────────────────────

class VirtualOrderTracker:
    """
    Tracks virtual limit orders through the raw MBO message stream.

    Parameters
    ----------
    mode : "standard" | "shadow"
    queue_change_threshold : float  (default 0.10 = 10%)
    """

    def __init__(
        self,
        mode: str = MODE_STANDARD,
        queue_change_threshold: float = QUEUE_CHANGE_THRESHOLD,
    ) -> None:
        if mode not in (MODE_STANDARD, MODE_SHADOW):
            raise ValueError(f"mode must be 'standard' or 'shadow'; got '{mode}'")
        self.mode         = mode
        self._is_standard = (mode == MODE_STANDARD)   # bool, faster than string cmp in hot loop
        self._threshold   = queue_change_threshold
        self._bid_book    = _BookSide(is_bid=True)
        self._ask_book    = _BookSide(is_bid=False)
        self._exchange_orders: Dict[str, tuple]         = {}
        self._virtual_orders:  Dict[str, _VirtualOrder] = {}
        self._ofi              = _OFIAccumulator()
        self._prev_best_bid: Optional[float] = None
        self._prev_best_ask: Optional[float] = None
        # Price-level index for O(affected) per-message iteration
        self._price_idx: Dict[str, Dict[int, set]] = {"B": {}, "A": {}}
        # Orders in post-fill grace period (only need mid update + terminal check)
        self._grace_vids: set = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_book(self) -> None:
        """
        Clear book and exchange registry.  Call at each session boundary.
        Also resets QP windows on all live orders — stale session pressure
        data must not carry across session boundaries.
        """
        self._bid_book  = _BookSide(is_bid=True)
        self._ask_book  = _BookSide(is_bid=False)
        self._exchange_orders.clear()
        self._prev_best_bid = None
        self._prev_best_ask = None
        self._ofi.reset()
        for vo in self._virtual_orders.values():
            vo.qp_window.clear()
            vo.qp_above_since_ns = 0
            vo.qp_ratio_last     = 0.0
        # Grace set is cleared at session boundary (all fills abandoned)
        self._grace_vids.clear()
        logger.debug("VirtualOrderTracker[%s]: book reset.", self.mode)

    def place_order(self, side: str, price: float, size: int, ts_ns: int,
                    shares_ahead: Optional[float] = None) -> str:
        """
        Register a new virtual limit order.  Returns virtual_id.
        Call AFTER feeding all messages up to the desired placement time.

        shares_ahead: pre-computed by the caller to avoid a redundant book scan
                      when placing std and shadow orders at the same price.
                      If None, computed here.
        """
        if side not in ("B", "A"):
            raise ValueError(f"place_order: side must be 'B' or 'A', got {side!r}")
        bid  = self._bid_book.best()
        ask  = self._ask_book.best()
        book = self._bid_book if side == "B" else self._ask_book
        sa   = shares_ahead if shares_ahead is not None else float(book.shares_strictly_ahead(price))

        vo = _VirtualOrder(
            virtual_id                = uuid.uuid4().hex,
            birth_ts_ns               = ts_ns,
            birth_mid                 = _safe_mid(bid, ask),
            birth_bid                 = bid if bid is not None else np.nan,
            birth_ask                 = ask if ask is not None else np.nan,
            side                      = side,
            price                     = price,
            price_ticks               = _to_ticks(price),
            shadow_price              = price,
            size                      = size,
            shares_ahead              = sa,
            last_emitted_shares_ahead = sa,
            repositioned              = False,
            reposition_ts_ns          = 0,
            fill_ts_ns                = 0,
            fill_price                = np.nan,
            mid_post_100ms            = np.nan,
            # qp_window / qp_above_since_ns / qp_ratio_last use dataclass defaults
        )
        self._virtual_orders[vo.virtual_id] = vo
        # Register in price-level index
        _idx = self._price_idx[side]
        if vo.price_ticks not in _idx:
            _idx[vo.price_ticks] = set()
        _idx[vo.price_ticks].add(vo.virtual_id)
        logger.debug("[%s] Placed %s side=%s price=%.4f sa=%.0f",
                     self.mode, vo.virtual_id[:8], side, price, sa)
        return vo.virtual_id

    def cancel_order(self, virtual_id: str, ts_ns: int) -> Optional[dict]:
        """
        Caller-initiated cancel (e.g. forced at session end for shadow orders).
        Returns a terminal CANCEL event dict, or None if already resolved.
        """
        vo = self._virtual_orders.pop(virtual_id, None)
        if vo is None:
            logger.debug("[%s] cancel_order: %s not found (already resolved).",
                         self.mode, virtual_id[:8])
            return None
        # Remove from price index and grace set
        _sidx = self._price_idx[vo.side]
        if vo.price_ticks in _sidx:
            _sidx[vo.price_ticks].discard(virtual_id)
            if not _sidx[vo.price_ticks]:
                del _sidx[vo.price_ticks]
        self._grace_vids.discard(virtual_id)
        bid  = self._bid_book.best()
        ask  = self._ask_book.best()
        book = self._bid_book if vo.side == "B" else self._ask_book
        sa   = float(book.shares_strictly_ahead(vo.price))
        sv   = _build_state_vec(vo, sa, book, self._ofi.value(),
                                _safe_spread(bid, ask))
        qfrac = _queue_fraction(vo, sa, book)
        # NOTE: do NOT reset OFI here. cancel_order() is called externally
        # (e.g. from _close_session which cancels many orders in sequence).
        # Resetting OFI inside cancel_order would zero it out for every
        # subsequent cancel in the same call-site, corrupting the OFI value
        # in each cancel event after the first.  The caller is responsible
        # for OFI management.  on_message() resets OFI once after all orders
        # are processed; _close_session resets the entire book via reset_book().
        return _make_event(
            vo=vo, trigger=TRIG_CANCEL, ts_ns=ts_ns,
            terminal=True, exec_price=np.nan, mid_post=np.nan,
            state_vec=sv, new_bid=bid, new_ask=ask, new_sa=sa,
            mode=self.mode,
            qp_ratio=vo.qp_ratio_last,
            qp_q_frac_gate=(qfrac < QPR_QFRAC_GATE),
        )

    def on_message(
        self,
        row:      dict,
        ts_ns:    int   = 0,
        msg_type: str   = "",
        side:     str   = "",
        price:    float = 0.0,
        shares:   int   = 0,
        oid:      str   = "",
    ) -> List[dict]:
        """
        Process one MBO message row.
        Returns a list of 0-or-more event dicts.

        ts_ns, msg_type, side, price, shares, oid:
            Pre-extracted by the pipeline to avoid redundant dict lookups.
            If ts_ns == 0 or msg_type == "" the values are read from row
            (backward-compatible fallback).
        """
        if not msg_type:
            msg_type = row["message_type"]
            side     = row["side"]
            price    = float(row["price_float"])
            shares   = int(row["shares"])
            oid      = str(row["order_id"])
            if ts_ns == 0:
                ts    = row["timestamp"]
                ts_ns = int(ts.timestamp() * 1e9) if hasattr(ts, "timestamp") else int(ts)

        # Guard: only B and A sides are valid for book/order tracking.
        # ITCH data contains other side codes (e.g. 'N' for non-displayed,
        # auction imbalance indicators, etc.) that must be ignored silently.
        if side not in ("B", "A"):
            return []

        emitted: List[dict] = []
        msg_ticks = _to_ticks(price)   # integer ticks for fast price equality

        # 1. OFI (before book changes)
        self._ofi.update(side, msg_type, shares)

        # 2. Exchange order registry  (tuple: side, price, remaining_shares)
        if msg_type == "A":
            self._exchange_orders[oid] = (side, price, shares)
        elif msg_type in ("C", "M"):
            self._exchange_orders.pop(oid, None)
        elif msg_type in ("T", "F"):
            if oid in self._exchange_orders:
                _es, _ep, _eq = self._exchange_orders[oid]
                _eq -= shares
                if _eq <= 0:
                    self._exchange_orders.pop(oid, None)
                else:
                    self._exchange_orders[oid] = (_es, _ep, _eq)

        # 3. Book update
        bk = self._bid_book if side == "B" else self._ask_book
        bk.apply(msg_type, price, shares)

        # 4. New BBO
        new_bid   = self._bid_book.best()
        new_ask   = self._ask_book.best()
        # tick is only needed for the 10-tick reposition check (standard, not-yet-repositioned).
        # Defer computation to avoid the call on every message when no order needs it.
        _tick: Optional[float] = None   # computed lazily below

        # 5. Per-virtual-order processing using price-level index.
        _cur_mid    = _safe_mid(new_bid, new_ask)
        _cur_spread = _safe_spread(new_bid, new_ask)
        _ofi_val    = self._ofi.value()
        terminal_ids: List[str] = []

        # ── Pass A: grace-period orders (tiny separate set) ───────────────
        if self._grace_vids:
            for _gvid in list(self._grace_vids):
                _gvo = self._virtual_orders.get(_gvid)
                if _gvo is None:
                    self._grace_vids.discard(_gvid)
                    continue
                _gbook = self._bid_book if _gvo.side == "B" else self._ask_book
                if not np.isnan(_cur_mid):
                    _gvo.mid_post_100ms = _cur_mid
                if ts_ns >= _gvo.fill_ts_ns + POST_FILL_GRACE_NS:
                    terminal_ids.append(_gvid)
                    self._grace_vids.discard(_gvid)
                    _gsv = _build_state_vec(_gvo, 0.0, _gbook, _ofi_val, _cur_spread)
                    emitted.append(_make_event(
                        vo=_gvo, trigger=TRIG_FILL, ts_ns=_gvo.fill_ts_ns,
                        terminal=True, exec_price=_gvo.fill_price,
                        mid_post=_gvo.mid_post_100ms, state_vec=_gsv,
                        new_bid=new_bid, new_ask=new_ask, new_sa=0.0,
                        mode=self.mode, qp_ratio=_gvo.qp_ratio_last,
                        qp_q_frac_gate=False,
                    ))

        # ── Pass B: build working sets from price-level index ─────────────
        _same_idx    = self._price_idx[side]
        _opp_side    = "A" if side == "B" else "B"
        _opp_idx     = self._price_idx[_opp_side]
        _opp_bbo_msg = new_ask if side == "B" else new_bid

        _visit_same: List[str] = []
        for _ticks, _vset in _same_idx.items():
            if (side == "B" and _ticks >= msg_ticks) or (side == "A" and _ticks <= msg_ticks):
                _visit_same.extend(_vset)

        _bbo_changed = (
            (side == "B" and new_bid != self._prev_best_bid) or
            (side == "A" and new_ask != self._prev_best_ask)
        )
        _visit_opp: List[str] = []
        if _bbo_changed or msg_type == "A":
            for _vset in _opp_idx.values():
                _visit_opp.extend(_vset)

        if self._grace_vids:
            _visit_same = [v for v in _visit_same if v not in self._grace_vids]
            _visit_opp  = [v for v in _visit_opp  if v not in self._grace_vids]

        # ── Pass C: same-side active orders ───────────────────────────────
        for vid in _visit_same:
            vo = self._virtual_orders.get(vid)
            if vo is None:
                continue

            # ── Max-lifetime force-cancel (standard orders only) ──────────
            # In quiet markets QPR never fires (empty window → qpr=0).
            # Without this, orders can sit idle for 30+ minutes with stale
            # birth metadata, then all fill at once when activity resumes.
            if self._is_standard and (ts_ns - vo.birth_ts_ns) > MAX_LIFETIME_NS:
                sv = _build_state_vec(vo, vo.shares_ahead,
                                      self._bid_book if vo.side == "B" else self._ask_book,
                                      _ofi_val, _cur_spread)
                emitted.append(_make_event(
                    vo=vo, trigger=TRIG_CANCEL, ts_ns=ts_ns,
                    terminal=True, exec_price=np.nan, mid_post=np.nan,
                    state_vec=sv, new_bid=new_bid, new_ask=new_ask,
                    new_sa=vo.shares_ahead, mode=self.mode,
                    qp_ratio=vo.qp_ratio_last, qp_q_frac_gate=False,
                ))
                terminal_ids.append(vid)
                logger.debug(
                    "[standard] max-lifetime cancel %s side=%s age_s=%.1f",
                    vid[:8], vo.side, (ts_ns - vo.birth_ts_ns) / 1e9,
                )
                continue   # skip all further processing for this order this message

            vo_book = self._bid_book if vo.side == "B" else self._ask_book
            opp_bbo = _opp_bbo_msg
            old_sa  = vo.shares_ahead

            _msg_affects_sa = (msg_ticks > vo.price_ticks) if side == "B" else (msg_ticks < vo.price_ticks)
            new_sa = float(vo_book.shares_strictly_ahead(vo.price)) if _msg_affects_sa else old_sa

            if self._is_standard:
                _stream_a = (msg_type in ("C", "M", "T", "F")
                             and msg_ticks == vo.price_ticks and old_sa > 0)
                if _stream_a:
                    vo.qp_window.update(
                        ts_ns=ts_ns, msg_type=msg_type, msg_side=side,
                        msg_price=price, msg_shares=float(shares),
                        order_side=vo.side, order_price=vo.price,
                        shares_ahead=old_sa, opp_bbo=opp_bbo,
                        msg_ticks=msg_ticks, order_ticks=vo.price_ticks,
                    )

            if self._is_standard and not vo.repositioned:
                if not np.isnan(vo.birth_mid) and not np.isnan(_cur_mid):
                    if _tick is None:
                        _tick = _tick_estimate(new_bid, new_ask)
                    drift_ticks = abs(_cur_mid - vo.birth_mid) / _tick
                    if drift_ticks > REPOSITION_TICKS_THRESHOLD:
                        new_price = new_bid if vo.side == "B" else new_ask
                        if new_price is not None and new_price > 0:
                            _old_ticks = vo.price_ticks
                            vo.price       = new_price
                            vo.price_ticks = _to_ticks(new_price)
                            vo.repositioned      = True
                            vo.reposition_ts_ns  = ts_ns   # timestamp of 10-tick drift event
                            new_sa = float(vo_book.shares_strictly_ahead(vo.price))
                            vo.shares_ahead              = new_sa
                            vo.last_emitted_shares_ahead = new_sa
                            vo.qp_window.clear()
                            vo.qp_above_since_ns = 0
                            _pidx = self._price_idx[vo.side]
                            if _old_ticks in _pidx:
                                _pidx[_old_ticks].discard(vid)
                                if not _pidx[_old_ticks]:
                                    del _pidx[_old_ticks]
                            if vo.price_ticks not in _pidx:
                                _pidx[vo.price_ticks] = set()
                            _pidx[vo.price_ticks].add(vid)

            if msg_ticks == vo.price_ticks and msg_type in ("T", "F"):
                fill_detected = (new_sa == 0)
            else:
                fill_detected = False
            if vo.side == "B":
                price_cross = (new_ask is not None and new_ask <= vo.price)
            else:
                price_cross = (new_bid is not None and new_bid >= vo.price)
            is_fill = fill_detected or price_cross

            strand_cancel = False
            if self._is_standard and not is_fill:
                if not vo.qp_window._dep and not vo.qp_window._grow:
                    qpr = 0.0
                    if vo.qp_above_since_ns != 0:
                        vo.qp_above_since_ns = 0
                    vo.qp_ratio_last = 0.0
                    qfrac = 0.0
                else:
                    qpr = vo.qp_window.ratio(ts_ns)
                    vo.qp_ratio_last = qpr
                    if qpr >= QPR_THRESHOLD:
                        if vo.qp_above_since_ns == 0:
                            vo.qp_above_since_ns = ts_ns
                    else:
                        vo.qp_above_since_ns = 0
                    if qpr >= QPR_THRESHOLD:
                        qfrac      = _queue_fraction(vo, new_sa, vo_book)
                        gate_depth = qfrac < QPR_QFRAC_GATE
                        gate_persist = (
                            vo.qp_above_since_ns > 0
                            and (ts_ns - vo.qp_above_since_ns) >= QPR_PERSIST_NS
                        )
                        strand_cancel = gate_depth and gate_persist
                        if strand_cancel:
                            logger.debug(
                                "[standard] lax-cancel %s side=%s qpr=%.2f "
                                "qfrac=%.2f persist_ms=%.1f",
                                vid[:8], vo.side, qpr, qfrac,
                                (ts_ns - vo.qp_above_since_ns) / 1e6,
                            )
                    else:
                        qfrac = 0.0

            if _msg_affects_sa and not is_fill and not strand_cancel:
                q_change    = _pct_change(vo.last_emitted_shares_ahead, new_sa)
                should_emit = q_change > self._threshold
            else:
                should_emit = is_fill or strand_cancel
            if not should_emit:
                vo.shares_ahead = new_sa
                continue

            if is_fill:
                vo.fill_ts_ns     = ts_ns
                vo.fill_price     = float(price)
                vo.mid_post_100ms = _cur_mid
                vo.shares_ahead              = new_sa
                vo.last_emitted_shares_ahead = new_sa
                self._grace_vids.add(vid)
                _pidx = self._price_idx[vo.side]
                if vo.price_ticks in _pidx:
                    _pidx[vo.price_ticks].discard(vid)
                    if not _pidx[vo.price_ticks]:
                        del _pidx[vo.price_ticks]
                continue

            sv     = _build_state_vec(vo, new_sa, vo_book, _ofi_val, _cur_spread)
            qfrac  = _queue_fraction(vo, new_sa, vo_book)
            trigger = TRIG_CANCEL if strand_cancel else TRIG_QUEUE_CHANGE
            emitted.append(_make_event(
                vo=vo, trigger=trigger, ts_ns=ts_ns,
                terminal=strand_cancel, exec_price=np.nan, mid_post=np.nan,
                state_vec=sv, new_bid=new_bid, new_ask=new_ask, new_sa=new_sa,
                mode=self.mode, qp_ratio=vo.qp_ratio_last,
                qp_q_frac_gate=(qfrac < QPR_QFRAC_GATE),
            ))
            if strand_cancel:
                terminal_ids.append(vid)
            vo.shares_ahead              = new_sa
            vo.last_emitted_shares_ahead = new_sa

        # ── Pass D: opposite-side active orders (price_cross + stream B) ──
        for vid in _visit_opp:
            vo = self._virtual_orders.get(vid)
            if vo is None:
                continue
            vo_book = self._bid_book if vo.side == "B" else self._ask_book
            opp_bbo = new_bid if vo.side == "B" else new_ask
            old_sa  = vo.shares_ahead
            new_sa  = old_sa

            if self._is_standard:
                _stream_b = (msg_type == "A" and opp_bbo is not None
                             and abs(price - opp_bbo) < 1e-9)
                if _stream_b:
                    vo.qp_window.update(
                        ts_ns=ts_ns, msg_type=msg_type, msg_side=side,
                        msg_price=price, msg_shares=float(shares),
                        order_side=vo.side, order_price=vo.price,
                        shares_ahead=old_sa, opp_bbo=opp_bbo,
                        msg_ticks=msg_ticks, order_ticks=vo.price_ticks,
                    )

            if vo.side == "B":
                price_cross = (new_ask is not None and new_ask <= vo.price)
            else:
                price_cross = (new_bid is not None and new_bid >= vo.price)
            if not price_cross:
                continue

            # Guard: skip if already in grace (fill set by Pass C or Pass A this
            # same message, order still in _virtual_orders pending section-6 removal).
            # Without this, Pass D would overwrite fill_ts_ns and re-add to grace_vids,
            # causing a duplicate terminal event in the pipeline and a lost slot.
            if vo.fill_ts_ns > 0:
                continue

            vo.fill_ts_ns     = ts_ns
            vo.fill_price     = float(price)
            vo.mid_post_100ms = _cur_mid
            vo.shares_ahead              = new_sa
            vo.last_emitted_shares_ahead = new_sa
            self._grace_vids.add(vid)
            _pidx = self._price_idx[vo.side]
            if vo.price_ticks in _pidx:
                _pidx[vo.price_ticks].discard(vid)
                if not _pidx[vo.price_ticks]:
                    del _pidx[vo.price_ticks]
        # 6. Remove finished orders
        for vid in terminal_ids:
            vo = self._virtual_orders.pop(vid, None)
            if vo is not None:
                _pidx = self._price_idx[vo.side]
                if vo.price_ticks in _pidx:
                    _pidx[vo.price_ticks].discard(vid)
                    if not _pidx[vo.price_ticks]:
                        del _pidx[vo.price_ticks]
                self._grace_vids.discard(vid)

        # 7. Reset OFI accumulator — once per message, after all orders processed
        self._ofi.reset()

        # 8. Advance BBO
        self._prev_best_bid = new_bid
        self._prev_best_ask = new_ask

        return emitted


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers  (all unchanged except _make_event gains two new params)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mid(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return bid if bid is not None else (ask if ask is not None else np.nan)


def _safe_spread(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is not None and ask is not None:
        return ask - bid
    return np.nan


def _tick_estimate(bid: Optional[float], ask: Optional[float]) -> float:
    s = _safe_spread(bid, ask)
    return s if (not np.isnan(s) and s > 0) else 0.01


def _pct_change(old_val: float, new_val: float) -> float:
    if old_val == 0:
        return 0.0 if new_val == 0 else 1.0
    return abs(new_val - old_val) / abs(old_val)


def _queue_fraction(vo: _VirtualOrder, shares_ahead: float, book: _BookSide) -> float:
    at_price = float(book.shares_at(vo.price))
    total    = shares_ahead + at_price
    return shares_ahead / total if total > 0 else 0.0


def _build_state_vec(
    vo: _VirtualOrder, sa: float, book: _BookSide, ofi: float, spread: float,
) -> np.ndarray:
    return np.array(
        [_queue_fraction(vo, sa, book), float(np.log1p(sa)),
         ofi, spread, float(np.log1p(vo.size))],
        dtype=np.float32,
    )


def _make_event(
    vo: _VirtualOrder, trigger: str, ts_ns: int, terminal: bool,
    exec_price: float, mid_post: float, state_vec: np.ndarray,
    new_bid: Optional[float], new_ask: Optional[float],
    new_sa: float, mode: str,
    qp_ratio: float = 0.0,
    qp_q_frac_gate: bool = False,
) -> dict:
    return {
        # ── existing fields (order preserved) ────────────────────────────
        "virtual_id":        vo.virtual_id,
        "mode":              mode,
        "trigger":           trigger,
        "ts_ns":             ts_ns,
        "birth_ts_ns":       vo.birth_ts_ns,
        "birth_mid":         vo.birth_mid,
        "birth_bid":         vo.birth_bid,
        "birth_ask":         vo.birth_ask,
        "side":              vo.side,
        "limit_price":       vo.price,
        "shadow_price":      vo.shadow_price,
        "repositioned":      vo.repositioned,
        "reposition_ts_ns":  vo.reposition_ts_ns,   # 0 when never repositioned
        "exec_price":        exec_price,
        "mid_post_100ms":    mid_post,
        "state":             state_vec,
        "shares_ahead":      new_sa,
        "best_bid":          new_bid,
        "best_ask":          new_ask,
        "mid":               _safe_mid(new_bid, new_ask),
        "terminal":          terminal,
        "mid_after_50ms":    np.nan,
        "mid_100ms_ago":     np.nan,
        "reward":            np.nan,
        # ── new QPR fields ────────────────────────────────────────────────
        # queue_pressure_ratio: QPR value in the 50 ms window at event time.
        #   For Shadow events this is always 0.0 (window not populated).
        #   ContributionEngine can group cancel outcomes by signal strength.
        # qp_q_frac_gate: True when the order was in the vulnerable front
        #   QPR_QFRAC_GATE (20%) of the queue at event time.  Useful for
        #   auditing whether the depth gate is calibrated correctly.
        "queue_pressure_ratio": qp_ratio,
        "qp_q_frac_gate":       qp_q_frac_gate,
    }