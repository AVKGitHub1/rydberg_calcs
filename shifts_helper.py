import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import arc
from arc import Rubidium85, ShirleyMethod

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
    calc = ShirleyMethod(Rubidium85())
    calc.defineBasis(n, l, j, mj, q, nmin, nmax, lmax)
    calc.defineShirleyHamiltonian(fn=1)
    freqs = np.linspace(freqMin, freqMax, numFreqs)
    calc.diagonalise(Efield, freqs, progressOutput=True)
    return calc

def plotShifts(n, l, j, mj, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs, title=None):
    calc = calculateShifts(n, l, j, mj, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs)
    state_string = get_state_string(n, l, j, mj)
    plt.plot(calc.freqs, calc.targetShifts, label=state_string)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy Shift (Hz)')
    if title is not None:
        plt.title(title)
    else:
        plt.title(f'Energy Shift vs Frequency for {state_string}')
    plt.legend()
    return calc

def compare_shifts(states, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs, title=None):
    calcs = []
    if title is None:
        title = "Energy Shift vs Frequency for Multiple States"
    for state in states:
        n, l, j, mj = state
        calc = plotShifts(n, l, j, mj, q, nmin, nmax, lmax, Efield, freqMin, freqMax, numFreqs, title=title)
        calcs.append(calc)
    return calcs