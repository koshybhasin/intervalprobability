# ShipsDestroyer
import sys
import math

def prob_all_destroyed(n: int, k: int) -> float:
    """
    Exact probability that an n×n grid is fully destroyed after k bombs.
    Bombs land uniformly on n^2 cells with replacement.
    A bomb destroys its entire row and column.
    """
    n2 = n * n
    fail = 0.0

    # Inclusion-exclusion over choosing r unhit rows and c unhit cols (both >= 1)
    for r in range(1, n + 1):
        choose_r = math.comb(n, r)
        nr = n - r
        for c in range(1, n + 1):
            choose_c = math.comb(n, c)
            nc = n - c

            p = (nr * nc) / n2  # probability a single bomb lands outside those rows+cols
            term = choose_r * choose_c * (p ** k)

            # (-1)^(r+c)
            if (r + c) % 2 == 0:
                fail += term
            else:
                fail -= term

    return 1.0 - fail


def main():
    if len(sys.argv) < 2:
        print("Usage: python ship_destroyer.py n [k]")
        print("  n: grid size (n×n)")
        print("  k: number of bombs (optional; default k=n)")
        sys.exit(1)

    n = int(sys.argv[1])
    if n <= 0:
        raise ValueError("n must be positive")

    k = int(sys.argv[2]) if len(sys.argv) >= 3 else n
    if k < 0:
        raise ValueError("k must be nonnegative")

    ans = prob_all_destroyed(n, k)
    print(f"{ans:.5f}")


if __name__ == "__main__":
    main()