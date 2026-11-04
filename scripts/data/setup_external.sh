#!/bin/bash
# Clone the external repositories the Atom-JEPA-Bio data and evaluation read, at the
# commits used for the release, into $MVC_ROOT (default data/mvc):
#   expansion-ml-comparison    Biogen ADME / ExpansionRx tables with the cluster split
#   model-validation-central   the atom-jepa study: protocol, published fold metrics,
#                              model_comparison.py (finetuning/admet/eval_mvc.py, report_mvc.py)
#   monroe                     PCBA preprocessing (scripts/data/pcba_monroe_prep.py) and
#                              the TabPFN 3 head (finetuning/admet/eval_mvc_tabpfn.py)
#
#   bash scripts/data/setup_external.sh
set -euo pipefail
ROOT="${MVC_ROOT:-data/mvc}"
mkdir -p "$ROOT"

clone() {   # url dir commit
    if [[ ! -d "$ROOT/$2/.git" ]]; then
        git clone --quiet "$1" "$ROOT/$2"
    fi
    git -C "$ROOT/$2" fetch --quiet origin
    git -C "$ROOT/$2" checkout --quiet "$3"
    echo "$ROOT/$2 @ $(git -C "$ROOT/$2" rev-parse --short HEAD)"
}

clone https://github.com/PatWalters/expansion-ml-comparison.git expansion-ml-comparison \
    06582771d48af8096f6b05a5a247b177189a8fa4
clone https://github.com/patwalters/model-validation-central.git model-validation-central \
    4deac088a2cfc3876d3bf9897440d66f08cbcebd
clone https://github.com/blazejba/monroe.git monroe \
    de075eabe98e4adbbedbc86ca9e877f4bfd9b629
