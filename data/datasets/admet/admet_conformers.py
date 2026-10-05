"""
RDKit 3D conformer generation + on-disk cache for the TDC ADMET datasets.

Conformer generation follows the original Uni-Mol recipe (Zhou et al., ICLR 2023)
so the fine-tuning input distribution matches the Uni-Mol pretraining distribution
our 3D encoder was trained on: per molecule we generate up to N_CONF ETKDG
conformers, each MMFF94-optimized. Generation is a SINGLE bulk
`EmbedMultipleConfs` call (not a loop of separate EmbedMolecule calls -- see
deviation 3 below), seeded by the caller's `conformer_seed`.

DIFFERENCES FROM VANILLA UNI-MOL (all deliberate; none of Uni-Mol / Uni-Mol2 /
Uni-Mol+ do any of these -- checked against their published code):

  0. LARGEST-FRAGMENT SELECTION. ~3.6% of TDC ADMET "Drug" SMILES are not a single
     molecule -- salt counterions (.[Cl-], .[Na+]), hydrates (.O), occasional raw-data
     duplicates (see check_admet_fragmentation.py for the per-dataset breakdown, up
     to 11% for solubility_aqsoldb). RDKit's distance-geometry embedder has no
     constraint tying disconnected fragments together, and MMFF's default
     ignoreInterfragInteractions=True means optimization doesn't relate them either --
     so embedding the raw multi-component SMILES gives the counterion/water an
     arbitrary, physically meaningless position relative to the parent molecule. We
     keep only the largest connected fragment (by heavy-atom count) before embedding.
     This drops the counterion/solvate atoms entirely rather than feed the encoder a
     fabricated relative pose. A no-op for the ~96% of molecules that are already a
     single fragment.

  1. NO 2D FALLBACK. Uni-Mol replaces an un-embeddable molecule (and any conformer
     whose embedding/MMFF fails) with a FLAT 2D conformation (z = 0) via
     Compute2DCoords. Our encoder is a 3D model and cannot use a flat conformer, so
     we never emit one. A molecule that yields no 3D conformer at all returns None
     and is dropped by the dataset (with a logged count) -- exactly like the old
     single-conformer code. With the chirality fix below this should essentially
     never happen: every parseable drug-like molecule embeds in 3D.

  2. CHIRALITY-HARDENED 3D EMBED. Plain ETKDG (chirality enforced) cannot place
     bridged bicyclics -- tropanes, norbornanes, quaternary-N alkaloids -- whose ring
     stereocentres over-constrain the distance-geometry bounds matrix; vanilla
     Uni-Mol therefore drops exactly these to a flat 2D blob. We probe once with
     chirality enforced (faithful, correct stereo for the easy majority); if that
     fails we embed with random-coordinate init and RELAXED chirality
     (enforceChirality=False), which recovers them as real 3D geometry (verified on
     all of hia_hou's failures, incl. a 100-atom calabash-curare alkaloid).

     enforceChirality=False can flip an *acyclic* stereocentre (ring/bridgehead
     centres are topologically locked and unaffected). It is only ever used on the
     handful of molecules that would otherwise be a flat 2D blob, so on net it is
     strictly better geometry than vanilla Uni-Mol on those molecules.

  3. RMSD-PRUNED BULK EMBEDDING (not a Uni-Mol-vs-us deviation -- Uni-Mol's own
     paper doesn't specify one either way, but this codebase's PREVIOUS version
     did neither: it called single-conformer EmbedMolecule N_CONF times onto the
     SAME mol object, overwriting the one stored conformer each time, with
     NOTHING checking that conformer k actually differed from conformer k-1 --
     i.e. no diversity control existed at all, and it was slower (N_CONF
     separate embed calls instead of one bulk one that reuses candidate-pool
     information). Rewritten to a single `AllChem.EmbedMultipleConfs` call with
     `pruneRmsThresh` (`data.conformer_prune_rms` in the finetune configs,
     default -1 i.e. pruning OFF pending a decision on the threshold; set e.g.
     0.5 Angstrom to enable): a candidate embed within that RMSD
     of one already accepted is discarded and re-tried. Rigid molecules (e.g.
     aspirin, caffeine) therefore legitimately return FEWER conformers than
     N_CONF even when embedding never fails -- there simply aren't N_CONF
     RMSD-distinct 3D layouts to find, and returning near-duplicates under the
     old code was never buying anything. `conformer_seed` is now the ACTUAL
     RDKit embedding seed (previously a cache-filename tag only; the real
     per-slot seeds were hardcoded 0..N_CONF-1), so it genuinely controls the
     generated geometry -- verified: same seed -> identical output, different
     seed -> different output.

Other Uni-Mol-faithful behavior kept:
  * Per-conformer MMFF94 optimization. If MMFF raises (missing params), we KEEP the
    raw embedded 3D coordinates for that conformer (still 3D) instead of failing.
  * Atoms carry explicit Hs (AddHs), aligned with QM9-style pretraining inputs.

A molecule may return FEWER than N_CONF conformers (>= 1): pruning removes
near-duplicates (see deviation 3) and a stubborn flexible bicyclic may only
embed on some random starts. Variable count is fine for sample-one-per-epoch
training and average-at-test inference; it avoids padding with duplicate
geometries.

Cache: one pickle per (dataset, seed, n_conf, prune_rms[, nofragstrip] tag),
keyed by raw SMILES. Each value is (atomic_numbers [N] int64, conformers
list[ [N,3] f32 ]) with len >= 1, or None on a genuine failure (unparseable
SMILES, or no 3D embedding within the time budget). NO BACKWARDS
COMPATIBILITY with caches built by the pre-rewrite code -- see
build_or_load_conformers' docstring.
"""

import os
import pickle
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")          # silence per-molecule embed warnings
except ImportError as e:                    # pragma: no cover
    raise ImportError(
        "ADMET fine-tuning needs RDKit. Install with `pip install rdkit`."
    ) from e


# (atomic_numbers [N] int64, list of >=1 positions [N,3] f32), or None on failure.
Conformer = Optional[Tuple[np.ndarray, List[np.ndarray]]]

N_CONF = 10             # conformers per molecule (Uni-Mol uses 10 3D; their +1 2D is dropped here)

# Upper bound on wall-clock spent in the relaxed (hard-molecule) embedding loop.
# Each relaxed embed is capped (~2s), so total time per hard molecule is ~this + one
# embed. Easy molecules never enter this path. This is a one-time cached build, so the
# budget is generous; lower it if your dataset has many slow flexible-bicyclic molecules.
HARD_BUDGET_S = 40.0


# -----------------------------------------------------------------------------
# parsing
# -----------------------------------------------------------------------------
def _largest_fragment(mol):
    """Keep only the largest connected component (by heavy-atom count). A no-op
    for single-component (or atomless) molecules. See module docstring, deviation 0."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:                # 0 atoms (e.g. "") or already single-fragment
        return mol
    return max(frags, key=lambda f: f.GetNumAtoms())


def _parse(smiles: str, strip_fragments: bool = True):
    """SMILES -> RDKit mol, with a sanitize-retry for borderline strings and
    (optionally) largest-fragment selection for multi-component input (salts,
    hydrates, mixtures) -- see module docstring, deviation 0. None if the string
    cannot be parsed/sanitized at all."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return _largest_fragment(mol) if strip_fragments else mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return _largest_fragment(mol) if strip_fragments else mol


# -----------------------------------------------------------------------------
# bulk multi-conformer embedding onto mol_h (Hs already added)
# -----------------------------------------------------------------------------
# Default RDKit pruneRmsThresh (Angstrom): a candidate conformer within this
# RMSD of one already accepted is discarded and re-tried under a fresh random
# start, so the returned set is geometrically DIVERSE rather than (as the old
# per-seed EmbedMolecule loop produced) an unchecked sequence of overwrite-
# embeds where nothing verified conformer k actually differed from k-1.
# -1.0 disables pruning (RDKit's own default / "off" sentinel) -- the default
# HERE, pending a decision on what pruning threshold (if any) is worth the
# extra embed attempts it costs; -1 is the closest match to the old
# (unpruned, but also unchecked) behavior as a neutral starting point.
DEFAULT_PRUNE_RMS = -1.0


def _embed_multi_strict(mol_h, seed: int, n_conf: int, prune_rms: float) -> List[int]:
    """Faithful Uni-Mol embed: ETKDG with chirality ENFORCED (correct stereo),
    bulk multi-conformer. One RDKit call embeds up to n_conf conformers
    (RDKit internally tries different random starts derived from `seed`),
    rather than the old code's n_conf separate EmbedMolecule calls onto the
    same overwritten mol -- strictly faster, and gives EmbedMultipleConfs'
    own pruneRmsThresh diversity check a full n_conf-wide candidate pool to
    prune duplicates from, which per-call embedding could never do. Returns
    the list of accepted conformer ids (len may be < n_conf, or empty)."""
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    p.maxIterations = 200
    p.pruneRmsThresh = float(prune_rms)
    return list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conf, params=p))


def _embed_multi_relaxed(mol_h, seed: int, n_conf: int, prune_rms: float) -> List[int]:
    """The fix: random-coord init + RELAXED chirality, bulk multi-conformer.
    Recovers bridged bicyclics (tropanes / norbornanes / quaternary-N
    alkaloids) that strict ETKDG can't place at all. Only reached when the
    strict bulk embed above returned ZERO conformers -- see _gen_3d_conformers."""
    p = AllChem.ETKDGv3()
    p.useRandomCoords = True
    p.enforceChirality = False
    p.maxIterations = 200
    p.pruneRmsThresh = float(prune_rms)
    p.randomSeed = int(seed)
    return list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conf, params=p))


def _gen_3d_conformers(mol_noH, n_conf: int, seed: int,
                       prune_rms: float = DEFAULT_PRUNE_RMS,
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Up to n_conf MMFF94-optimized, RMSD-pruned 3D conformers.

    ONE bulk strict-chirality embed call; if that yields nothing at all (the
    bridged-stereocentre case), retry with the relaxed-chirality path,
    reseeded each attempt, under a wall-clock budget. Preserves the old
    policy exactly: "prefer faithful chirality, only fall back to relaxed if
    strict recovers NOTHING" -- a strict embed that succeeds on some but not
    all n_conf attempts still keeps its (possibly partial) result rather than
    also trying relaxed. `seed` now genuinely determines the geometry (RDKit
    derives each internal embed attempt's random start from it), unlike the
    old code where the actual embedding seeds were hardcoded 0..n_conf-1 and
    the caller's seed was only ever a cache-filename tag. A conformer whose
    MMFF optimization raises keeps its raw embedded 3D coords. Returns
    (z [N], confs); confs may be shorter than n_conf, and is empty if nothing
    embedded. Hs included; no 2D fallback."""
    mol_h = Chem.AddHs(mol_noH)
    z = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()], dtype=np.int64)

    cids = _embed_multi_strict(mol_h, seed, n_conf, prune_rms)
    if not cids:
        t0 = time.time()
        attempt = 0
        while not cids and (time.time() - t0) < HARD_BUDGET_S:
            cids = _embed_multi_relaxed(mol_h, seed * 100 + attempt, n_conf, prune_rms)
            attempt += 1

    confs: List[np.ndarray] = []
    for cid in cids:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
        except Exception:
            pass                               # keep the raw embedded 3D coords (still 3D)
        pos = mol_h.GetConformer(cid).GetPositions().astype(np.float32)
        assert pos.shape[0] == z.shape[0], "3D coord/atom mismatch"
        confs.append(pos)
    return z, confs


# -----------------------------------------------------------------------------
# SMILES -> conformers (None on genuine failure; never a flat 2D conformer)
# -----------------------------------------------------------------------------
def smi2coords(smiles: str, n_conf: int = N_CONF, strip_fragments: bool = True,
              seed: int = 42, prune_rms: float = DEFAULT_PRUNE_RMS) -> Conformer:
    """SMILES -> (atomic_numbers [N], list of >=1 3D conformers [N,3]), or None.

    None means a genuine failure: the SMILES is unparseable, or no 3D conformer
    could be embedded even with relaxed chirality inside the time budget. With the
    chirality fix this should be vanishingly rare for drug-like molecules.

    strip_fragments: keep only the largest connected component of multi-component
    SMILES (salts/hydrates/mixtures) before embedding -- see module docstring,
    deviation 0. Default True; set False to reproduce vanilla-Uni-Mol-style
    whole-SMILES embedding (not recommended -- see check_admet_fragmentation.py
    for how badly disconnected fragments overlap under that embedding).

    seed: the actual RDKit embedding seed (unlike the pre-rewrite code, where
    this was a cache-filename tag ONLY and the real embedding seeds were
    hardcoded 0..n_conf-1) -- see _gen_3d_conformers.
    prune_rms: RDKit pruneRmsThresh (Angstrom); -1 disables pruning."""
    mol_noH = _parse(smiles, strip_fragments=strip_fragments)
    if mol_noH is None or mol_noH.GetNumAtoms() == 0:
        return None
    z, confs = _gen_3d_conformers(mol_noH, n_conf, seed, prune_rms=prune_rms)
    if not confs:
        return None
    return z, confs


def build_or_load_conformers(
    name: str,
    smiles: Iterable[str],
    cache_dir: str,
    seed: int = 42,                     # now the REAL embedding seed too -- see smi2coords
    n_conf: int = N_CONF,
    strip_fragments: bool = True,
    n_workers: int = 1,
    prune_rms: float = DEFAULT_PRUNE_RMS,
) -> Dict[str, Conformer]:
    """Return {smiles -> (z, conformer_list) | None}, building+persisting misses.

    One pickle per (dataset, seed-tag, n_conf[, nofragstrip]). Only SMILES not
    already cached are embedded, so adding the test split after train (or
    re-running) is free for done molecules.

    n_conf is baked into the cache filename (not just `seed`): `todo` is computed
    from smiles NOT ALREADY A KEY in the cache dict, regardless of how many
    conformers that entry holds, so a cache built with a smaller n_conf would
    otherwise be silently treated as "done" and never topped up when n_conf is
    later increased -- exactly the failure mode a conformer-count sweep would hit.

    strip_fragments: forwarded to smi2coords -- see its docstring. Changing this
    changes the embedded geometry, so it's baked into the cache filename (a
    strip_fragments=False cache lives in a separate file) rather than silently
    reusing a cache built under the other setting.

    n_workers: n_workers > 1 embeds molecules in a joblib process pool (the default
    "loky" backend): each molecule's conformer generation is independent, CPU-bound
    RDKit work, so this parallelizes cleanly across processes (RDKit isn't
    thread-safe, hence processes rather than threads). 1 runs sequentially
    in-process, exactly like before. Negative values follow joblib's own n_jobs
    convention: -1 = all available CPUs, -2 = all minus one, etc. -- resolved by
    joblib itself (loky.cpu_count(), which is affinity/cgroup-aware, e.g. under a
    SLURM --cpus-per-task allocation) rather than hand-rolled here. Either way the
    worker count is capped to the number of molecules actually being embedded, so a
    small incremental batch doesn't spawn an oversized pool.

    prune_rms: RDKit pruneRmsThresh (Angstrom) -- see _embed_multi_strict's
    docstring. Baked into the cache filename (below) for the same reason
    strip_fragments is: it changes the embedded geometry, so a cache built
    under a different value must never be silently reused as "done".

    NO BACKWARDS COMPATIBILITY: this rewrite switched from a sequence of
    single-conformer EmbedMolecule calls onto the same overwritten mol (no
    diversity check -- nothing verified conformer k differed from k-1) to a
    single bulk EmbedMultipleConfs call with RMSD pruning, and `seed` changed
    from a cache-filename tag only to the ACTUAL embedding seed. Both mean a
    cache built by the pre-rewrite code is neither format- nor content-
    compatible; there is no migration path and none is needed -- just delete
    the old *.pkl files (or, since prune_rms is now part of the filename, an
    old cache simply won't be matched and will be rebuilt under the new name).
    A molecule cached as None here is a genuine failure and is NOT retried on re-run."""
    os.makedirs(cache_dir, exist_ok=True)
    prune_tag = f"prune{prune_rms:g}"
    tag = (f"seed{seed}_nconf{n_conf}_{prune_tag}" if strip_fragments
          else f"seed{seed}_nconf{n_conf}_{prune_tag}_nofragstrip")
    path = os.path.join(cache_dir, f"{name}_{tag}.pkl")

    cache: Dict[str, Conformer] = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)

    todo = [s for s in dict.fromkeys(smiles) if s not in cache]  # de-dup, keep order
    if todo:
        n_jobs = min(Parallel(n_jobs=n_workers)._effective_n_jobs(), len(todo))
        print(f"[conformers] {name}: embedding {len(todo)} new molecule(s) "
              f"(up to {n_conf} 3D conformers each; cache has {len(cache)}; "
              f"workers={n_jobs}) -> {path}", flush=True)
        n_failed = n_partial = 0
        worker = partial(smi2coords, n_conf=n_conf, strip_fragments=strip_fragments,
                         seed=seed, prune_rms=prune_rms)
        results_iter = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(worker)(smi) for smi in todo
        )
        pbar = tqdm(zip(todo, results_iter), total=len(todo), desc=f"[conformers] {name}")
        for smi, conf in pbar:
            cache[smi] = conf
            if conf is None:
                n_failed += 1
            elif len(conf[1]) < n_conf:
                n_partial += 1
            pbar.set_postfix(failed=n_failed, partial=n_partial)
        # PID-unique tmp name, NOT a shared `path + ".tmp"`: SLURM array tasks
        # (see scripts/submit_finetune_biogen_adme.sh and the biogen_*/
        # submit_test_biogen_jepa_finetune.sh arrays) run concurrently against
        # this same cache, and on a COLD cache every task builds it at once.
        # os.replace is atomic, which protects a reader from ever seeing a
        # half-written file -- but it does nothing about multiple WRITERS
        # sharing one tmp path. Measured with 5 concurrent writers: the shared
        # name failed 8/8 trials, every time as FileNotFoundError, because the
        # first process's os.replace renames the shared tmp away while the
        # others are still mid-write and then try to rename a path that no
        # longer exists. (Interleaved pickle streams silently corrupting the
        # file are possible in principle too, but the replace race fires first
        # and much more reliably.) One tmp per process makes each write
        # self-contained -- 0/8 failures -- and the last replace wins; since
        # every task embeds the same molecules under the same seed, whichever
        # writer wins is equally correct.
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)  # atomic publish
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)     # don't leave a stray tmp on failure/interrupt
            raise
        print(f"[conformers] {name}: done "
              f"({n_failed}/{len(todo)} failed -> None, "
              f"{n_partial} got < {n_conf} conformers)", flush=True)

    return cache


# -----------------------------------------------------------------------------
# conformer selection helpers (consumed by the dataset / predict path)
# -----------------------------------------------------------------------------
def sample_conformer(conf: Conformer, rng: np.random.Generator) -> Optional[np.ndarray]:
    """Uni-Mol training: randomly sample ONE of the molecule's conformers."""
    if conf is None:
        return None
    _, confs = conf
    return confs[int(rng.integers(len(confs)))]


def all_conformers(conf: Conformer) -> List[np.ndarray]:
    """Uni-Mol inference: every conformer (predictions are averaged over them)."""
    if conf is None:
        return []
    return list(conf[1])