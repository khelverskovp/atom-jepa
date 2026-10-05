"""Print what this machine supports: python -m atom_jepa.info"""

import importlib.metadata as md

import torch

import atom_jepa


def main():
    cuda = torch.cuda.is_available()
    print(f"atom-jepa {atom_jepa.__version__}, torch {torch.__version__} (CUDA build: {torch.version.cuda})")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0) if cuda else 'none'}")
    print(f"bf16: {'yes' if cuda and torch.cuda.is_bf16_supported() else 'no'}")
    versions = {p: md.version(p) for p in ("cuequivariance", "cuequivariance-torch")
                if _installed(p)}
    print(f"cuEquivariance: {', '.join(f'{p} {v}' for p, v in versions.items()) or 'not installed'}")
    if cuda and not versions:
        major = (torch.version.cuda or "12").split(".")[0]
        print(f'  install the kernels with: pip install "atom-jepa[cu{major}]"')


def _installed(package):
    try:
        md.version(package)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
