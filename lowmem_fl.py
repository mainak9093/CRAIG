"""[REPLICATION PATCH 2026-06-12] Exact low-memory facility-location ordering.

Why this file exists
--------------------
The original pipeline (util.faciliy_location_order) materializes a full per-class
similarity matrix S = max(dist) - dist via sklearn.pairwise_distances. For the
covtype experiment the training split is 290,506 points (~145k per class), so S
would need ~84 GB (float32) -- plus a same-size temporary -- which exceeds the RAM
of both the laptop (15.7 GB) and most single GPU servers. The paper's authors ran
selection through the external SMTK toolkit on large-memory hardware.

This module computes the *same* greedy ordering without ever materializing S:
similarity columns S[:, j] = m - dist(X, x_j) are computed on demand (O(N d) per
column), and the global statistics (m = max pairwise distance, per-column sums for
the initial heap) are obtained in one chunked pass.

Faithfulness
------------
The objective and update rules mirror lazy_greedy.FacilityLocation line by line:
  - log-transformed facility location: norm * log(1 + f_norm * sum(curMax))
  - f_norm = alpha / sum_i max_j S[i, j]  (= alpha / (N*m), diagonal similarity is m)
  - norm   = 1 / log(1 + alpha)
  - the original's `if not ndx` quirk (candidate index 0 after the first selection
    returns log(1 + alpha) as its gain) is reproduced intentionally.
The greedy pick at each step maximizes a monotone transform of the plain facility
location gain, so the selected order matches the dense implementation up to
floating-point ties. Cluster sizes/weights are computed by chunked nearest-selected
assignment, identical in meaning to np.argmax(S[i, order]).

Memory: O(N * chunk) instead of O(N^2). Time: one O(N^2 d / chunk) pass for the
statistics + O(N d) per lazy-greedy gain evaluation.
"""
import heapq
import math
import time

import numpy as np

# Classes larger than this use the low-memory path (45k^2 float32 = ~8 GB; the
# dense path also needs a same-size temporary, so 25k is a safe cutoff for <=64GB
# hosts). Raise this on large-memory machines to use the original dense path.
LOWMEM_THRESHOLD = 25_000

_CHUNK = 2048  # rows per block in the chunked passes


def _pairwise_stats(X):
    """One chunked pass over the distance matrix: returns (m, colsums, secs).

    m       : max pairwise euclidean distance (the original's np.max(dists))
    colsums : sum_i dist(i, j) for every column j (for the initial heap gains)
    """
    start = time.time()
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    sq = np.einsum('ij,ij->i', X, X)
    m = 0.0
    colsums = np.zeros(n, dtype=np.float64)
    for s in range(0, n, _CHUNK):
        e = min(s + _CHUNK, n)
        # dist^2 block = ||x_i||^2 + ||x_j||^2 - 2 x_i.x_j  (rows s:e vs all)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (X[s:e] @ X.T)
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2, out=d2)
        m = max(m, float(d.max()))
        colsums += d.sum(axis=0, dtype=np.float64)
    return m, colsums, time.time() - start


class _LowMemFacilityLocation:
    """Mirror of lazy_greedy.FacilityLocation with on-demand similarity columns."""

    def __init__(self, X, m, colsums, alpha=1.0):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.sq = np.einsum('ij,ij->i', self.X, self.X)
        self.n = self.X.shape[0]
        self.m = m
        # sum_j S[i, j] per column j = n*m - sum_i dist(i, j)
        self.col_S_sums = self.n * m - colsums
        self.curVal = 0.0
        self.curMax = np.zeros(self.n, dtype=np.float64)
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / (self.n * m)        # original: alpha / D[:, V].max(1).sum()
        self.norm = 1.0 / math.log(1 + self.alpha)     # original: 1 / inc(V, [])
        self.started = False                           # True once the first element is added

    def column(self, j):
        """S[:, j] = m - dist(X, x_j), computed on demand."""
        d2 = self.sq + self.sq[j] - 2.0 * (self.X @ self.X[j])
        np.maximum(d2, 0.0, out=d2)
        return self.m - np.sqrt(d2)

    def inc_initial(self, j):
        """Gain of j for the empty set -- vectorizable via the precomputed column sums."""
        return self.norm * math.log(1 + self.f_norm * self.col_S_sums[j]) - self.curVal

    def inc(self, j):
        if self.started:
            if not j:  # original quirk: `if not ndx` is True for index 0
                return math.log(1 + self.alpha * 1)
            return (self.norm * math.log(1 + self.f_norm *
                                         np.maximum(self.curMax, self.column(j)).sum())
                    - self.curVal)
        return self.norm * math.log(1 + self.f_norm * float(self.col_S_sums[j])) - self.curVal

    def add(self, j):
        cur_old = self.curVal
        col = self.column(j)
        if self.started:
            np.maximum(self.curMax, col, out=self.curMax)
        else:
            self.curMax = col
            self.started = True
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.append(self.curVal - cur_old)
        return self.curVal


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)


def _heappop_max(heap):
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def _lazy_greedy_heap(F, B):
    """lazy_greedy.lazy_greedy_heap with the initial gains taken from column sums.

    The heap is built by sequential max-heap pushes in index order, exactly like
    the original ([_heappush_max(order, (F.inc(sset, i), i)) for i in V]), so the
    internal heap layout -- and therefore tie-breaking on equal gains -- matches.
    """
    sset = []
    # initial gains for the empty set, all candidates at once
    init_gains = F.norm * np.log1p(F.f_norm * F.col_S_sums)
    order = []
    heapq._heapify_max(order)
    for j in range(F.n):
        _heappush_max(order, (float(init_gains[j]), j))

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(el[1])
        if improv >= 0:
            if not order:
                F.add(el[1])
                sset.append(el[1])
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    F.add(el[1])
                    sset.append(el[1])
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)
    return sset


def _cluster_sizes(X, order, weights=None):
    """Chunked equivalent of: sz[argmax(S[i, order])] += 1 (or += weights[i])."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    sel = X[order]
    sq_sel = np.einsum('ij,ij->i', sel, sel)
    sz = np.zeros(len(order), dtype=np.float64)
    for s in range(0, X.shape[0], _CHUNK):
        e = min(s + _CHUNK, X.shape[0])
        blk = X[s:e]
        d2 = (np.einsum('ij,ij->i', blk, blk)[:, None] + sq_sel[None, :]
              - 2.0 * (blk @ sel.T))
        nearest = np.argmin(d2, axis=1)  # max similarity == min distance
        if weights is None:
            np.add.at(sz, nearest, 1.0)
        else:
            np.add.at(sz, nearest, weights[s:e])
    return sz


def facility_location_order_lowmem(X_class, B, weights=None):
    """Drop-in low-memory replacement for one class of util.faciliy_location_order.

    Args
    - X_class: np.array [N, d], the feature rows of ONE class
    - B: int, number of points to select
    - weights: optional per-point weights for the cluster sizes

    Returns (order, cluster_sizes, greedy_time, similarity_time) with `order`
    indexing into X_class (the caller maps back to global indices).
    """
    n = X_class.shape[0]
    B = int(min(B, n))
    m, colsums, sim_time = _pairwise_stats(X_class)
    start = time.time()
    F = _LowMemFacilityLocation(X_class, m, colsums)
    order = _lazy_greedy_heap(F, B)
    greedy_time = time.time() - start
    order = np.asarray(order, dtype=np.int64)
    sz = _cluster_sizes(X_class, order, weights)
    return order, sz, greedy_time, sim_time
