# split.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit


@dataclass
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class Split3Result:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _mode_int(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    vals, cnt = np.unique(a.astype(np.int64), return_counts=True)
    return int(vals[int(np.argmax(cnt))])


def _group_label_int(a: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.int64)
    pos = a[a > 0]
    if pos.size:
        return _mode_int(pos)
    return _mode_int(a)


def _validate_class_distribution(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
) -> bool:
    """
    Validate bahwa setiap class punya minimal 1 vessel di train AND test.
    Ini penting untuk stratified split agar model bisa belajar semua class.
    """
    y = np.asarray(y).astype(np.int64)
    groups = np.asarray(groups)
    
    # Get unique group labels per class in train & test
    classes_in_train = set()
    classes_in_test = set()
    
    for idx_set, class_set in [(train_idx, classes_in_train), (test_idx, classes_in_test)]:
        unique_groups = np.unique(groups[idx_set])
        for group in unique_groups:
            group_mask = groups[idx_set] == group
            group_labels = y[idx_set][group_mask]
            group_class = _group_label_int(group_labels)
            class_set.add(int(group_class))
    
    # Check all classes that are actually present in this dataset. Some
    # event-level tasks keep a stable label map even when a small run has no
    # samples for one class, e.g. transshipment loitering without encounter.
    all_classes = set(np.unique(y).astype(int).tolist())
    if not (classes_in_train == all_classes and classes_in_test == all_classes):
        return False
    
    return True


def _validate_window_distribution(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    y: np.ndarray,
    num_classes: int,
) -> bool:
    y = np.asarray(y).astype(np.int64)
    all_classes = set(np.unique(y).astype(int).tolist())
    return set(np.unique(y[left_idx]).astype(int).tolist()) == all_classes and set(
        np.unique(y[right_idx]).astype(int).tolist()
    ) == all_classes


def _split_once_strat_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> SplitResult | None:
    groups = np.asarray(groups)
    y = np.asarray(y)

    uniq = np.unique(groups)
    if uniq.size < 3:
        return None

    # label per vessel = mode label windows dalam vessel tsb
    group_labels = np.zeros(uniq.shape[0], dtype=np.int64)
    for i, g in enumerate(uniq):
        idx = np.where(groups == g)[0]
        group_labels[i] = _group_label_int(y[idx])

    vals, cnt = np.unique(group_labels, return_counts=True)
    # kalau ada kelas yg cuma 1 vessel → stratify bisa error
    if (vals.size < 2) or np.any(cnt < 2):
        return None

    g_train, g_test = train_test_split(
        uniq,
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels,
        shuffle=True,
    )

    g_train = set(map(str, g_train.tolist()))
    is_train = np.array([str(g) in g_train for g in groups], dtype=bool)

    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]
    return SplitResult(train_idx=train_idx, test_idx=test_idx)


def _group_labels(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(np.int64)
    groups_str = np.asarray(groups).astype(str)
    uniq = np.unique(groups_str)

    labels = np.zeros(uniq.shape[0], dtype=np.int64)
    for i, g in enumerate(uniq):
        idx = np.where(groups_str == str(g))[0]
        labels[i] = _group_label_int(y[idx])

    return uniq.astype(str), labels, groups_str


def _indices_from_group_sets(
    groups_str: np.ndarray,
    train_groups: list[str],
    val_groups: list[str],
    test_groups: list[str],
) -> Split3Result:
    train_set = set(map(str, train_groups))
    val_set = set(map(str, val_groups))
    test_set = set(map(str, test_groups))

    return Split3Result(
        train_idx=np.where(np.isin(groups_str, list(train_set)))[0],
        val_idx=np.where(np.isin(groups_str, list(val_set)))[0],
        test_idx=np.where(np.isin(groups_str, list(test_set)))[0],
    )


def _classwise_group_split_threeway(
    y: np.ndarray,
    groups: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
    num_classes: int,
) -> Split3Result | None:
    """
    Deterministic class-aware split at vessel level.

    For gear classification the minority classes only have a handful of
    vessels, so a global stratified split can still produce a validation set
    missing a class. This allocates each class independently while preserving
    disjoint MMSI groups.
    """
    uniq, labels, groups_str = _group_labels(y, groups)
    rng = np.random.RandomState(random_state)

    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []

    for c in range(num_classes):
        cls_groups = uniq[labels == c].copy()
        if cls_groups.size < 3:
            return None

        n = int(cls_groups.size)
        n_test = max(1, int(round(n * float(test_size))))
        n_val = max(1, int(round(n * float(val_size))))

        while n_test + n_val >= n and (n_test > 1 or n_val > 1):
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        if n_test + n_val >= n:
            return None

        group_counts = {
            str(g): int(np.sum(groups_str == str(g)))
            for g in cls_groups.tolist()
        }
        total_windows = float(sum(group_counts.values()))
        target_test = max(1.0, total_windows * float(test_size))
        target_val = max(1.0, total_windows * float(val_size))

        best_perm: np.ndarray | None = None
        best_score = float("inf")
        tries = min(2048, max(128, n * 128))

        for _ in range(tries):
            perm = cls_groups.copy()
            rng.shuffle(perm)

            cand_test = perm[:n_test].tolist()
            cand_val = perm[n_test:n_test + n_val].tolist()
            cand_train = perm[n_test + n_val:].tolist()

            test_windows = float(sum(group_counts[str(g)] for g in cand_test))
            val_windows = float(sum(group_counts[str(g)] for g in cand_val))
            train_windows = float(sum(group_counts[str(g)] for g in cand_train))

            score = (
                abs(test_windows - target_test) / target_test
                + abs(val_windows - target_val) / target_val
                + 0.25 * abs(train_windows - (total_windows - target_test - target_val))
                  / max(total_windows - target_test - target_val, 1.0)
            )

            if val_windows < target_val * 0.35:
                score += (target_val * 0.35 - val_windows) / target_val * 3.0
            if test_windows < target_test * 0.35:
                score += (target_test * 0.35 - test_windows) / target_test * 2.0

            if score < best_score:
                best_score = score
                best_perm = perm

        if best_perm is None:
            return None

        test_groups.extend(best_perm[:n_test].tolist())
        val_groups.extend(best_perm[n_test:n_test + n_val].tolist())
        train_groups.extend(best_perm[n_test + n_val:].tolist())

    split = _indices_from_group_sets(groups_str, train_groups, val_groups, test_groups)
    if split.train_idx.size == 0 or split.val_idx.size == 0 or split.test_idx.size == 0:
        return None

    ok_train_val = _validate_class_distribution(
        split.train_idx, split.val_idx, y, groups, num_classes
    )
    ok_train_test = _validate_class_distribution(
        split.train_idx, split.test_idx, y, groups, num_classes
    )
    if not (ok_train_val and ok_train_test):
        return None

    return split


def group_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_groups: bool = True,
    max_tries: int = 200,
) -> SplitResult:
    """
    Group split (no vessel leakage). Kalau stratify_groups=True:
    - coba stratified split di level vessel
    - retry beberapa kali untuk dapat coverage kelas yg lebih bagus
    - VALIDATE: setiap class minimal ada 1 vessel di train dan test
    """
    if not stratify_groups:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return SplitResult(train_idx=train_idx, test_idx=test_idx)

    y = np.asarray(y).astype(np.int64)
    groups = np.asarray(groups)

    num_classes = int(np.max(y)) + 1 if y.size else 1

    best = None
    best_score = -1

    for k in range(max_tries):
        rs = int(random_state + k)
        cand = _split_once_strat_groups(X, y, groups, test_size, rs)
        if cand is None:
            continue

        # VALIDATE: setiap class harus ada di train dan test
        if not _validate_class_distribution(cand.train_idx, cand.test_idx, y, groups, num_classes):
            continue

        y_tr = y[cand.train_idx]
        y_te = y[cand.test_idx]

        # coverage score: berapa kelas yg muncul di train & test
        cov_tr = np.unique(y_tr).size
        cov_te = np.unique(y_te).size
        # penalti kalau ada kelas hilang
        score = (cov_tr + cov_te) - 2 * (num_classes - cov_tr) - 2 * (num_classes - cov_te)

        if score > best_score:
            best_score = score
            best = cand

        # kalau semua kelas muncul di dua sisi, berhenti
        if cov_tr == num_classes and cov_te == num_classes:
            best = cand
            break

    if best is not None:
        return best

    # fallback biasa
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return SplitResult(train_idx=train_idx, test_idx=test_idx)


def group_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_groups: bool = True,
    max_tries: int = 400,
) -> Split3Result:
    """
    Three-way group split (no vessel leakage).

    Train is used for fitting the model and scaler, validation is used for
    checkpoint/tau/aggregation selection, and test is reserved for final eval.
    """
    y = np.asarray(y).astype(np.int64)
    groups = np.asarray(groups)

    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be > 0 and sum to < 1.")

    num_classes = int(np.max(y)) + 1 if y.size else 1
    uniq = np.unique(groups)
    if uniq.size < 3:
        raise ValueError("Need at least 3 unique groups for a non-overlapping train/val/test split.")

    if not stratify_groups:
        trainval_size = 1.0 - float(test_size)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        trainval_idx, test_idx = next(gss_test.split(X, y, groups=groups))

        rel_val_size = float(val_size) / trainval_size
        gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state + 1)
        tr_rel, val_rel = next(gss_val.split(X[trainval_idx], y[trainval_idx], groups=groups[trainval_idx]))
        return Split3Result(
            train_idx=trainval_idx[tr_rel],
            val_idx=trainval_idx[val_rel],
            test_idx=test_idx,
        )

    classwise = _classwise_group_split_threeway(
        y=y,
        groups=groups,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        num_classes=num_classes,
    )
    if classwise is not None:
        return classwise

    group_labels = np.zeros(uniq.shape[0], dtype=np.int64)
    for i, g in enumerate(uniq):
        idx = np.where(groups == g)[0]
        group_labels[i] = _group_label_int(y[idx])

    vals, cnt = np.unique(group_labels, return_counts=True)
    can_stratify = vals.size >= 2 and not np.any(cnt < 3)
    require_group_class_coverage = vals.size == num_classes and not np.any(cnt < 3)

    best: Split3Result | None = None
    best_score = -1
    trainval_size = 1.0 - float(test_size)
    rel_val_size = float(val_size) / trainval_size

    for k in range(max_tries):
        rs = int(random_state + k)

        try:
            if can_stratify:
                g_trainval, g_test, y_trainval, _ = train_test_split(
                    uniq,
                    group_labels,
                    test_size=test_size,
                    random_state=rs,
                    stratify=group_labels,
                    shuffle=True,
                )
                tv_vals, tv_cnt = np.unique(y_trainval, return_counts=True)
                if tv_vals.size < 2 or np.any(tv_cnt < 2):
                    continue
                g_train, g_val = train_test_split(
                    g_trainval,
                    test_size=rel_val_size,
                    random_state=rs + 10_000,
                    stratify=y_trainval,
                    shuffle=True,
                )
            else:
                g_trainval, g_test = train_test_split(
                    uniq,
                    test_size=test_size,
                    random_state=rs,
                    shuffle=True,
                )
                g_train, g_val = train_test_split(
                    g_trainval,
                    test_size=rel_val_size,
                    random_state=rs + 10_000,
                    shuffle=True,
                )
        except ValueError:
            continue

        g_train = set(map(str, np.asarray(g_train).tolist()))
        g_val = set(map(str, np.asarray(g_val).tolist()))
        g_test = set(map(str, np.asarray(g_test).tolist()))

        g_as_str = np.asarray(groups).astype(str)
        train_idx = np.where(np.isin(g_as_str, list(g_train)))[0]
        val_idx = np.where(np.isin(g_as_str, list(g_val)))[0]
        test_idx = np.where(np.isin(g_as_str, list(g_test)))[0]

        if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
            continue

        if require_group_class_coverage:
            ok_train_val = _validate_class_distribution(
                train_idx, val_idx, y, groups, num_classes
            )
            ok_train_test = _validate_class_distribution(
                train_idx, test_idx, y, groups, num_classes
            )
        else:
            ok_train_val = _validate_window_distribution(
                train_idx, val_idx, y, num_classes
            )
            ok_train_test = _validate_window_distribution(
                train_idx, test_idx, y, num_classes
            )
        if not (ok_train_val and ok_train_test):
            continue

        cov_train = np.unique(y[train_idx]).size
        cov_val = np.unique(y[val_idx]).size
        cov_test = np.unique(y[test_idx]).size
        score = (
            cov_train + cov_val + cov_test
            - 2 * (num_classes - cov_train)
            - 2 * (num_classes - cov_val)
            - 2 * (num_classes - cov_test)
        )

        cand = Split3Result(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        if score > best_score:
            best_score = score
            best = cand

        if cov_train == num_classes and cov_val == num_classes and cov_test == num_classes:
            return cand

    if best is not None:
        return best

    # Last-resort fallback still keeps vessels disjoint. If it drops a class
    # from validation or test, fail loudly instead of silently selecting a
    # biased checkpoint/tau.
    trainval_size = 1.0 - float(test_size)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups=groups))
    rel_val_size = float(val_size) / trainval_size
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state + 1)
    tr_rel, val_rel = next(gss_val.split(X[trainval_idx], y[trainval_idx], groups=groups[trainval_idx]))
    fallback = Split3Result(
        train_idx=trainval_idx[tr_rel],
        val_idx=trainval_idx[val_rel],
        test_idx=test_idx,
    )
    fallback_has_coverage = (
        _validate_class_distribution(fallback.train_idx, fallback.val_idx, y, groups, num_classes)
        and _validate_class_distribution(fallback.train_idx, fallback.test_idx, y, groups, num_classes)
        if require_group_class_coverage
        else _validate_window_distribution(fallback.train_idx, fallback.val_idx, y, num_classes)
        and _validate_window_distribution(fallback.train_idx, fallback.test_idx, y, num_classes)
    )
    if not fallback_has_coverage:
        raise RuntimeError(
            "Could not create a train/val/test split with every class present "
            "in train, validation, and test. Increase val_size/test_size or add more vessels."
        )

    return fallback
