import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from pathlib import Path
import pickle
import arc
from arc import Rubidium85, Rubidium87, ShirleyMethod


class CachedShiftsResult:
    def __init__(self, freqs, targetShifts):
        self.freqs = np.asarray(freqs)
        self.targetShifts = np.asarray(targetShifts)


_SHIFTS_CACHE_FILE = Path(__file__).with_name("_shifts_cache.pkl")
_SHIFTS_CACHE = {}


def _make_shifts_cache_key(n, l, j, mj, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs):
    return (
        int(n),
        int(l),
        float(j),
        float(mj),
        int(q),
        int(nmin),
        int(nmax),
        int(lmax),
        float(Efield),
        float(freqMin),
        float(freqMax),
        int(numFreqs),
    )


def _is_valid_cache_key(key):
    return isinstance(key, tuple) and len(key) == 12


def _to_cache_payload(calc):
    return {
        "freqs": np.asarray(calc.freqs),
        "targetShifts": np.asarray(calc.targetShifts),
    }


def _load_shifts_cache(cache_file=None):
    global _SHIFTS_CACHE
    cache_path = Path(cache_file) if cache_file is not None else _SHIFTS_CACHE_FILE
    if not cache_path.exists():
        _SHIFTS_CACHE = {}
        return _SHIFTS_CACHE

    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError):
        _SHIFTS_CACHE = {}
        return _SHIFTS_CACHE

    if not isinstance(payload, dict):
        _SHIFTS_CACHE = {}
        return _SHIFTS_CACHE

    loaded_cache = {}
    for key, value in payload.items():
        if not _is_valid_cache_key(key) or not isinstance(value, dict):
            continue
        freqs = value.get("freqs")
        target_shifts = value.get("targetShifts")
        if freqs is None or target_shifts is None:
            continue
        loaded_cache[key] = CachedShiftsResult(freqs, target_shifts)

    _SHIFTS_CACHE = loaded_cache
    return _SHIFTS_CACHE


def _save_shifts_cache(cache_file=None):
    cache_path = Path(cache_file) if cache_file is not None else _SHIFTS_CACHE_FILE
    payload = {}
    for key, value in _SHIFTS_CACHE.items():
        if not _is_valid_cache_key(key):
            continue
        if not hasattr(value, "freqs") or not hasattr(value, "targetShifts"):
            continue
        payload[key] = _to_cache_payload(value)

    try:
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        return False
    return True


_load_shifts_cache()


def get_state_string(n, l, j, mj, show_plus=True):
    l_name = {
        0: "S", 1: "P", 2: "D", 3: "F",
        4: "G", 5: "H", 6: "I"
    }

    def frac_str(x, show_plus=show_plus):
        f = Fraction(x).limit_denominator()
        s = f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        if show_plus and x > 0:
            s = "+" + s
        return s

    j_str = frac_str(j, show_plus=False)
    mj_str = frac_str(mj)

    return rf"$|{n}{l_name.get(l, l)}_{{{j_str}}},\, m_j={mj_str}\rangle$"

def calculateShifts(n, l, j, mj, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs):
    """Calculate Shirley shifts for a given set of parameters.
    Parameters:
    n, l, j, mj: Quantum numbers for the target state.
    q: polarization of the electric field.
    Efield: Electric field strength.
    freqMin, freqMax: Minimum and maximum frequencies.
    numFreqs: Number of frequencies.
    Returns:
    freqs: Array of frequencies.
    targetShifts: Array of energy shifts corresponding to the frequencies.
    """
    cache_key = _make_shifts_cache_key(n, l, j, mj, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs)
    cached_calc = _SHIFTS_CACHE.get(cache_key)
    if cached_calc is not None:
        return cached_calc

    calc = ShirleyMethod(Rubidium87())
    calc.defineBasis(n, l, j, mj, q, nmin, nmax, lmax)
    calc.defineShirleyHamiltonian(fn=1)
    freqs = np.linspace(freqMin, freqMax, numFreqs)
    calc.diagonalise(Efield, freqs, progressOutput=True)
    _SHIFTS_CACHE[cache_key] = calc
    _save_shifts_cache()
    return calc

def plotShifts(n, l, j, mj, q, nrange, lmax, Efield, freqMin, freqMax, numFreqs, title=None, twopiunits=False):
    calc = calculateShifts(n, l, j, mj, q, n-nrange, n+nrange, lmax, Efield, freqMin, freqMax, numFreqs)
    state_string = get_state_string(n, l, j, mj)
    if twopiunits:
        plt.plot(calc.freqs/(2*np.pi)/1e9, calc.targetShifts/1e6, label=state_string)
        plt.xlabel('Frequency (2π GHz)')
    else:
        plt.plot(calc.freqs/1e9, calc.targetShifts/1e6, label=state_string)
        plt.xlabel('Frequency (GHz)')
    plt.ylabel('Energy Shift (MHz)')
    if title is not None:
        plt.title(title)
    else:
        plt.title(f'Energy Shift vs Frequency for {state_string}')
    plt.legend()
    return calc

def compare_shifts(states, q, nrange, lmax, Efield, freqMin, freqMax, numFreqs, title=None, twopiunits=False):
    calcs = []
    if title is None:
        title = "Energy Shift vs Frequency for Multiple States"
    for state in states:
        n, l, j, mj = state
        calc = plotShifts(n, l, j, mj, q, nrange, lmax, Efield, freqMin, freqMax, numFreqs, title=title, twopiunits=False)
        calcs.append(calc)
    return calcs
