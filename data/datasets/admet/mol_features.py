"""
Featurizer registry + caching for the feature-vector molecular baselines
(finetuning/admet/baseline.py's 6-model battery: {Morgan, RDKit descriptors, both} x
{MLP, LightGBM}).
Mirrors two existing conventions rather than inventing new ones:

  * data/datasets/admet/admet_conformers.py's "build once over the whole molecule pool,
    cache to disk, reuse across folds/seeds" pattern (build_or_load_features
    below), plus its RDKit parsing conventions (RDLogger silencing).
  * The Contrastive KERMT paper's own baseline code (NVIDIA-BioNeMo/KERMT,
    kermt/data/molfeaturegenerator.py), itself adapted from chemprop's
    features-generator registry: `register_featurizer`/`get_featurizer`
    below is a direct analogue of their `register_features_generator`/
    `get_features_generator`, and `morgan_fingerprint`'s binary mode
    reproduces their `morgan_binary_features_generator` bit-for-bit.

COMPOSITION: a model does NOT get a fused "morgan+rdkit" featurizer -- it
lists several (`baseline.featurizers: [morgan, rdkit_desc]`) and
finetuning.admet.baseline._build_features concatenates them in listed order. Each is
cached separately, so the Morgan cache is shared by every Morgan-using
variant and the (slower) RDKit descriptor block is computed once per dataset
and reused by all 3 variants that include it.

PICKLABILITY: build_or_load_features dispatches through joblib with
`partial(fn, **kwargs)`, so every featurizer must stay a module-level
function taking `smiles` as its first positional argument.
"""

import hashlib
import os
import pickle
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdReducedGraphs
    RDLogger.DisableLog("rdApp.*")          # silence per-molecule parse/deprecation warnings
except ImportError as e:                    # pragma: no cover
    raise ImportError(
        "ADMET baselines need RDKit. Install with `pip install rdkit`."
    ) from e


FeaturesGenerator = Callable[..., Optional[np.ndarray]]
FEATURIZERS: Dict[str, FeaturesGenerator] = {}


def register_featurizer(name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    def decorator(fn: FeaturesGenerator) -> FeaturesGenerator:
        FEATURIZERS[name] = fn
        return fn
    return decorator


def get_featurizer(name: str) -> FeaturesGenerator:
    if name not in FEATURIZERS:
        raise ValueError(f"Unknown featurizer {name!r}; available: {list(FEATURIZERS)}")
    return FEATURIZERS[name]


@register_featurizer("morgan")
def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024,
                       use_counts: bool = False) -> Optional[np.ndarray]:
    """Morgan (ECFP-like) fingerprint [n_bits] float32.

    use_counts=False -> binary 0.0/1.0 presence bits: the exact "Morgan
    fingerprints (RDKit) at radius 2, 1024 bits" baseline feature, and
    bit-identical to the legacy AllChem.GetMorganFingerprintAsBitVect this
    used to call (asserted in the test suite, since already-published
    morgan_mlp numbers depend on it).
    use_counts=True -> substructure OCCURRENCE COUNTS in the same hashed
    bins. Strictly more information than the binary form (a fragment
    appearing 3x is distinguishable from once), which can help tree models
    in particular; searched as an HPO categorical rather than assumed better.

    Built on rdFingerprintGenerator (the modern, non-deprecated RDKit API)
    so binary and count modes come from one generator instead of two
    different legacy entry points. None on an unparsable SMILES -- mirrors
    data/datasets/admet/admet_conformers.py's parse-failure convention (reported by the
    caller, never silently substituted with zeros, which would be a
    fabricated molecule).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    arr = gen.GetCountFingerprintAsNumPy(mol) if use_counts else gen.GetFingerprintAsNumPy(mol)
    return arr.astype(np.float32)


@register_featurizer("avalon")
def avalon_fingerprint(smiles: str, n_bits: int = 1024,
                       use_counts: bool = True) -> Optional[np.ndarray]:
    """Avalon fingerprint [n_bits] float32 -- the second of the three
    fingerprint types Notwell & Wood (arXiv:2310.00174, MapLight) found
    complementary to Morgan for a strong ADMET fixed-descriptor baseline
    ("ECFP and Avalon fingerprints performed better than other molecular
    fingerprints available through RDKit"). Their published code
    (maplightrx/MapLight-TDC, maplight.py) calls GetHashedMorganFingerprint
    and GetAvalonCountFP at n_bits=1024 -- use_counts=True reproduces that.

    n_bits is intentionally NOT exposed as an HPO axis by the search-space
    fragments that use this featurizer: benchmarking shows folding size has
    little effect on Avalon specifically (r^2=0.993 between 1024-bit and
    16384-bit versions), unlike Morgan where it matters more.

    use_counts=False -> binary presence bits via GetAvalonFP. Unlike
    morgan_fingerprint's two modes (one hashed generator, counts vs. binary
    read off the same bit assignment), GetAvalonCountFP and GetAvalonFP are
    SEPARATE RDKit algorithms with different bit-setting logic -- verified
    the two are NOT a strict counts>=binary superset at matching indices
    (a handful of bits set in the binary form are 0 in the count form).
    Treat them as two distinct featurizations, not one with an on/off count
    toggle. None on an unparsable SMILES, matching morgan_fingerprint's
    convention.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_bits = int(n_bits)
    if use_counts:
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


@register_featurizer("erg")
def erg_fingerprint(smiles: str, fuzz_increment: float = 0.3, min_path: int = 1,
                    max_path: int = 15) -> Optional[np.ndarray]:
    """Extended Reduced Graph (ErG) fingerprint, a 2D pharmacophore
    description [Stiefl et al. 2006] -- the third fingerprint type in the
    MapLight recipe above ("a third type of fingerprint that performed well
    when used in conjunction with ECFP and Avalon fingerprints"). Encodes
    pharmacophore-triplet counts rather than substructures, so it is
    complementary to Morgan/Avalon rather than redundant with them.

    WIDTH DEPENDS ON max_path: verified output width == 21 * max_path
    (210 at max_path=10, 315 at the RDKit/MapLight default 15, 420 at 20).
    Search-space fragments using this featurizer must therefore fix
    max_path rather than sweep it, since a per-trial width change would
    otherwise ripple into in_dim/checkpoint shapes. fuzz_increment (default
    0.3) trades off pharmacophore-similarity fuzziness for scaffold-hopping
    use; property-prediction use may prefer a smaller, "crisper" value, so
    it IS a reasonable HPO axis.

    Always dense float64 (no binary/count distinction -- ErG is inherently a
    hologram of triplet occurrence counts). None on an unparsable SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.asarray(rdReducedGraphs.GetErGFingerprint(
        mol, 0, float(fuzz_increment), int(min_path), int(max_path)))
    return arr.astype(np.float32)


# Fixed, sorted descriptor ordering captured ONCE at import: Descriptors'
# own registry order is not contractually stable across RDKit versions, and
# these vectors get pickled into a cache and reused across processes/runs, so
# the column meaning must not depend on iteration order. Sorting by name
# makes it reproducible and inspectable.
RDKIT_DESCRIPTOR_NAMES: List[str] = sorted(name for name, _ in Descriptors._descList)


@register_featurizer("rdkit_desc")
def rdkit_descriptors(smiles: str, ipc_log1p: bool = True) -> Optional[np.ndarray]:
    """All RDKit 2D descriptors [len(RDKIT_DESCRIPTOR_NAMES)] float32, in the
    fixed sorted order above (217 on RDKit 2026.03: physicochemical,
    topological, charge/EState, plus 85 `fr_*` substructure counts).

    SANITIZING (order matters):
      1. Ipc -> log1p(Ipc). Ipc is the information content of the
         characteristic polynomial and grows ~exponentially with molecule
         size: measured 4.4e2 -> 1.1e11 across the real Biogen ADME pool, and
         it overflows to inf outright on larger molecules. Left raw it
         single-handedly dominates any StandardScaler fit (and hence the MLP
         input); log1p brings it into the same order as its neighbours while
         staying monotonic, so no ranking information is lost.
      2. non-finite -> 0.0 (np.nan_to_num). A handful of descriptors are
         undefined for exotic valences/disconnected fragments; 0.0 keeps the
         vector usable and, after train-fit standardization, sits at that
         column's mean rather than at an arbitrary extreme.
    Note descriptors are NOT scaled here -- scaling must be fit on the TRAIN
    split only (leakage), which this per-molecule, split-agnostic, cached
    function cannot know about. See finetuning.admet.baseline.fit_apply_feature_scaler.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = Descriptors.CalcMolDescriptors(mol)
    if ipc_log1p and "Ipc" in d:
        ipc = d["Ipc"]
        d = dict(d)
        d["Ipc"] = float(np.log1p(ipc)) if np.isfinite(ipc) and ipc >= 0 else ipc
    arr = np.array([d.get(name, np.nan) for name in RDKIT_DESCRIPTOR_NAMES], dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


# descriptastorus' generator is expensive to construct (it loads the reference
# CDF parameters), so build it once, lazily, per process -- joblib workers each
# get their own. Module-level function + module-level cache keeps
# build_or_load_features' `partial(fn, **kwargs)` picklable.
_RDKIT2D_NORM_GEN = None


def _rdkit2d_normalized_generator():
    global _RDKIT2D_NORM_GEN
    if _RDKIT2D_NORM_GEN is None:
        try:
            from descriptastorus.descriptors import rdNormalizedDescriptors
        except ImportError as e:   # pragma: no cover
            raise ImportError(
                "The rdkit_2d_normalized featurizer needs descriptastorus. "
                "Install with `pip install descriptastorus==2.8.0`."
            ) from e
        _RDKIT2D_NORM_GEN = rdNormalizedDescriptors.RDKit2DNormalized()
    return _RDKIT2D_NORM_GEN


@register_featurizer("rdkit_2d_normalized")
def rdkit_2d_normalized(smiles: str) -> Optional[np.ndarray]:
    """The 200 CDF-normalized RDKit 2D descriptors from descriptastorus
    [Kelley 2018] -- byte-for-byte the featurization the Contrastive KERMT
    paper uses for its "RDKit Descriptor + MLP" baseline ("200 RDKit 2D
    descriptors normalized with descriptastorus"), and the same one chemprop
    exposes as `rdkit_2d_normalized`.

    WHY THIS RATHER THAN `rdkit_desc` (raw 217 + a fitted StandardScaler):
    descriptastorus applies each descriptor's empirical CDF, estimated once on
    a large external reference chemical space, mapping every column into
    [0, 1]. That is categorically better behaved than standardizing raw
    descriptors on the training split:
      * BOUNDED. Nothing can blow up. Standardizing raw descriptors on Biogen
        ADME produces inputs up to 53 sigma, driven by near-constant `fr_*`
        columns (e.g. fr_azide is 0 for almost every molecule, so its tiny std
        turns the few positives into enormous z-scores).
      * SPLIT-INDEPENDENT. The transform does not depend on which molecules
        landed in train, so it cannot differ between train and test, and there
        is no fit step to leak (contrast fit_apply_feature_scaler, which must
        be fitted on train only).
    Models using this featurizer therefore set `feature_scaling: none` -- the
    CDF *is* the normalization, and that is what KERMT feeds its MLP.

    `rdkit_desc` above is kept registered so existing runs stay reproducible.
    Returns None on an unparsable SMILES (descriptastorus itself returns None,
    or a leading validity flag of False, for those)."""
    out = _rdkit2d_normalized_generator().process(smiles)
    if out is None or not out[0]:      # out[0] is descriptastorus' validity flag
        return None
    arr = np.asarray(out[1:], dtype=np.float64)
    # Defensive: the reference CDFs are well-behaved on drug-like input, but a
    # pathological molecule can still yield nan; keep the same never-fabricate
    # convention as everything else here.
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr.astype(np.float32)


def build_or_load_features(name: str, smiles: Iterable[str], cache_dir: str,
                           featurizer_name: str, featurizer_kwargs: Optional[dict] = None,
                           n_workers: int = 1) -> Dict[str, Optional[np.ndarray]]:
    """Return {smiles -> feature_vector | None}, building+persisting misses.

    One pickle per (dataset `name`, `featurizer_name`, `featurizer_kwargs`) --
    the cache filename encodes ALL of these (kwargs via a short hash), not
    just `name`, so sweeping e.g. radius: 2 -> 3 during later HPO can never
    silently reuse a stale cache built under different featurizer params.
    Unparsable SMILES are cached as None (a genuine, reported failure) and
    are NOT retried on re-run -- same convention as build_or_load_conformers."""
    featurizer_kwargs = dict(featurizer_kwargs or {})
    featurizer = get_featurizer(featurizer_name)
    kwargs_tag = ",".join(f"{k}={v}" for k, v in sorted(featurizer_kwargs.items()))
    tag_hash = hashlib.sha1(f"{featurizer_name}:{kwargs_tag}".encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_{featurizer_name}_{tag_hash}.pkl")

    cache: Dict[str, Optional[np.ndarray]] = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)

    todo = [s for s in dict.fromkeys(smiles) if s not in cache]   # de-dup, keep order
    if todo:
        n_jobs = min(Parallel(n_jobs=n_workers)._effective_n_jobs(), len(todo))
        print(f"[features] {name}: computing {featurizer_name} features "
              f"(kwargs={featurizer_kwargs}) for {len(todo)} new molecule(s) "
              f"(cache has {len(cache)}; workers={n_jobs}) -> {path}", flush=True)
        worker = partial(featurizer, **featurizer_kwargs)
        results_iter = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(worker)(smi) for smi in todo
        )
        n_failed = 0
        pbar = tqdm(zip(todo, results_iter), total=len(todo), desc=f"[features] {name}")
        for smi, feat in pbar:
            cache[smi] = feat
            if feat is None:
                n_failed += 1
            pbar.set_postfix(failed=n_failed)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)   # atomic write
        print(f"[features] {name}: done ({n_failed}/{len(todo)} failed -> None)", flush=True)

    return cache


def build_concat_features(name: str, all_smiles: Iterable[str], cache_dir: str,
                          featurizer_names: List[str],
                          featurizer_kwargs: Optional[Dict[str, dict]] = None,
                          n_workers: int = 1) -> Dict[str, Optional[np.ndarray]]:
    """{smiles -> concatenated feature vector | None} over every featurizer in
    `featurizer_names`, concatenated IN LISTED ORDER. Extracted from
    finetuning/admet/baseline.py's original inline `_build_features` (which is now a
    thin wrapper around this -- see there) so the same composition logic
    serves both the feature-vector baselines and finetuning.admet.training.train's encoder-fusion
    path (finetune.fusion, see MultiTaskFinetuneDataset's `features` param)
    without duplicating it or making either path import the other's module.

    Each featurizer is built and cached SEPARATELY (one pickle per
    featurizer+kwargs, via build_or_load_features above) rather than as one
    fused "morgan+rdkit" featurizer, so the Morgan cache is shared across
    every Morgan-using caller and the slower RDKit descriptor block is
    computed once per dataset and reused by everyone who wants it.

    A molecule is None (and so dropped+reported downstream, e.g. by
    MolecularFeatureDataset) if ANY featurizer failed on it, so the
    concatenated vector is never a partially-fabricated mix of real and
    substituted values.

    featurizer_kwargs: {featurizer_name -> kwargs dict}. A featurizer absent
    from this dict (or the dict itself being None) gets {} -- most
    featurizers here (rdkit_2d_normalized, and morgan at its defaults) need
    no kwargs at all."""
    kwargs_by_name = featurizer_kwargs or {}
    per_featurizer = [
        build_or_load_features(name, all_smiles, cache_dir, fname,
                               kwargs_by_name.get(fname, {}), n_workers=n_workers)
        for fname in featurizer_names
    ]

    if len(per_featurizer) == 1:
        return per_featurizer[0]

    out: Dict[str, Optional[np.ndarray]] = {}
    for smi in dict.fromkeys(all_smiles):
        parts = [d.get(smi) for d in per_featurizer]
        out[smi] = None if any(p is None for p in parts) else np.concatenate(parts).astype(np.float32)
    return out
