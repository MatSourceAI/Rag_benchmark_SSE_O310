"""Illustrative text processing for this release, NOT the historical experimental code.

Only self-contained chemistry-like inline math is normalized. Unrecognized
notation is retained. No bibliographic section or table is silently removed.
"""
import re
from dataclasses import dataclass, asdict

WRAPPER = re.compile(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}")
SUBSCRIPT = re.compile(r"_\s*\{\s*([0-9.]+)\s*\}")
ELEMENTS = set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split())

def _formula(match):
    original = match.group(0)
    candidate = match.group(1)
    candidate = WRAPPER.sub(lambda m: re.sub(r'\s+', '', m.group(1)), candidate)
    candidate = SUBSCRIPT.sub(lambda m: re.sub(r'\s+', '', m.group(1)), candidate)
    candidate = re.sub(r'\s+', '', candidate)
    # Do not drop charges, powers, fractions, units, relations or parentheses.
    pieces = re.findall(r'([A-Z][a-z]?)(\d*(?:\.\d+)?)', candidate)
    reconstructed = ''.join(e+n for e,n in pieces)
    if len(pieces) < 2 or reconstructed != candidate:
        return original
    if not all(e in ELEMENTS for e,n in pieces):
        return original
    return candidate

def normalize_inline_formulae(text):
    return re.sub(r'(?<!\$)\$(?!\$)([^$\n]+)\$(?!\$)', _formula, text)

@dataclass
class Chunk:
    start: int
    end: int
    content: str

def chunk_text(text, max_chars=3000, overlap=400):
    if not isinstance(max_chars, int) or not isinstance(overlap, int):
        raise ValueError('Chunk settings must be integers')
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError('Require 0 <= overlap < max_chars')
    start, result = 0, []
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            # Prefer a boundary in the latter half, while guaranteeing progress.
            lower = start + max(overlap + 1, max_chars//2)
            candidates = list(re.finditer(r'\n\s*\n|(?<=[.!?])\s+', text[lower:end]))
            if candidates:
                end = lower + candidates[-1].end()
        result.append(Chunk(start, end, text[start:end]))
        if end == len(text):
            break
        start = end - overlap
    return result

def process(text):
    normalized = normalize_inline_formulae(text)
    return {'implementation_status':'illustrative_revision_not_historical',
            'raw_chars':len(text), 'normalized_chars':len(normalized),
            'removed_chars':len(text)-len(normalized),
            'chunks':[asdict(c) for c in chunk_text(normalized)]}
