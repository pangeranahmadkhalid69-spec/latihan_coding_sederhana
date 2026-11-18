"""
Simulasi Hukum Newton I, II, dan III dengan Python
Library: numpy, matplotlib
Jalankan: python simulasi_newton.py
"""
import numpy as np
import matplotlib.pyplot as plt

DT = 0.01      # langkah waktu (s)
T_MAX = 10.0   # durasi simulasi (s)
t = np.arange(0, T_MAX, DT)


def simulate_1d(m, force_func, x0=0.0, v0=0.0):
    """Integrasi numerik (semi-implicit Euler) untuk a = F/m."""
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    x[0], v[0] = x0, v0
    for i in range(1, len(t)):
        a[i - 1] = force_func(t[i - 1], x[i - 1], v[i - 1]) / m
        v[i] = v[i - 1] + a[i - 1] * DT
        x[i] = x[i - 1] + v[i] * DT
    a[-1] = a[-2]
    return x, v, a


# ---------------------------------------------------------------
# HUKUM NEWTON I (Kelembaman / Inersia)
# Jika resultan gaya = 0, benda diam tetap diam, benda bergerak
# tetap bergerak dengan kecepatan konstan.
# ---------------------------------------------------------------
def hukum_1():
    m, v0 = 2.0, 3.0
    # Tanpa gaya (permukaan licin): kecepatan konstan
    x_a, v_a, _ = simulate_1d(m, lambda t, x, v: 0.0, v0=v0)
    # Ada gaya gesek (gaya luar): kecepatan berubah -> kontras
    x_b, v_b, _ = simulate_1d(m, lambda t, x, v: -1.5 * np.sign(v) if abs(v) > 1e-3 else 0.0, v0=v0)
    return x_a, v_a, x_b, v_b


# ---------------------------------------------------------------
# HUKUM NEWTON II (F = m.a)
# Gaya sama, massa berbeda -> percepatan berbeda.
# ---------------------------------------------------------------
def hukum_2():
    F = 10.0
    masses = [1.0, 2.0, 5.0]
    results = {}
    for m in masses:
        x, v, a = simulate_1d(m, lambda t, x, v: F)
        results[m] = (x, v, a)
    return F, results


# ---------------------------------------------------------------
# HUKUM NEWTON III (Aksi-Reaksi)
# Dua benda saling mendorong: F_12 = -F_21, momentum total kekal.
# ---------------------------------------------------------------
def hukum_3():
    m1, m2 = 60.0, 40.0          # massa dua skater (kg)
    x1, x2 = np.zeros_like(t), np.zeros_like(t)
    v1, v2 = np.zeros_like(t), np.zeros_like(t)
    f1, f2 = np.zeros_like(t), np.zeros_like(t)
    x1[0], x2[0] = -0.5, 0.5
    for i in range(1, len(t)):
        # Saling dorong 0 - 1 detik dengan gaya 50 N
        F = 50.0 if t[i - 1] < 1.0 else 0.0
        f2[i - 1] = F      # gaya pada benda 2 (aksi)
        f1[i - 1] = -F     # gaya pada benda 1 (reaksi)
        v1[i] = v1[i - 1] + (f1[i - 1] / m1) * DT
        v2[i] = v2[i - 1] + (f2[i - 1] / m2) * DT
        x1[i] = x1[i - 1] + v1[i] * DT
        x2[i] = x2[i - 1] + v2[i] * DT
    p_total = m1 * v1 + m2 * v2
    return m1, m2, x1, x2, v1, v2, f1, f2, p_total


def main():
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle("Simulasi Hukum Newton I, II, dan III", fontsize=16, fontweight="bold")

    # ---- Hukum I ----
    x_a, v_a, x_b, v_b = hukum_1()
    axes[0, 0].plot(t, x_a, label="F = 0 (licin)")
    axes[0, 0].plot(t, x_b, label="Ada gesekan")
    axes[0, 0].set(title="Hukum I: Posisi vs Waktu", xlabel="t (s)", ylabel="x (m)")
    axes[0, 0].legend()
    axes[0, 1].plot(t, v_a, label="F = 0 -> v konstan")
    axes[0, 1].plot(t, v_b, label="Ada gaya luar -> v berubah")
    axes[0, 1].set(title="Hukum I: Kecepatan vs Waktu", xlabel="t (s)", ylabel="v (m/s)")
    axes[0, 1].legend()
    axes[0, 2].axis("off")
    axes[0, 2].text(0.05, 0.5,
                    "Hukum I (Inersia)\n\nJika \u03a3F = 0 maka\nbenda mempertahankan\nkeadaan geraknya\n(v konstan, a = 0).",
                    fontsize=12, va="center")

    # ---- Hukum II ----
    F, res = hukum_2()
    for m, (x, v, a) in res.items():
        axes[1, 0].plot(t, a, label=f"m = {m} kg")
        axes[1, 1].plot(t, v, label=f"m = {m} kg")
        axes[1, 2].plot(t, x, label=f"m = {m} kg")
    axes[1, 0].set(title=f"Hukum II: Percepatan (F = {F} N)", xlabel="t (s)", ylabel="a (m/s\u00b2)")
    axes[1, 1].set(title="Hukum II: Kecepatan", xlabel="t (s)", ylabel="v (m/s)")
    axes[1, 2].set(title="Hukum II: Posisi", xlabel="t (s)", ylabel="x (m)")
    for ax in axes[1]:
        ax.legend()

    # ---- Hukum III ----
    m1, m2, x1, x2, v1, v2, f1, f2, p = hukum_3()
    axes[2, 0].plot(t, f1, label="Gaya pada benda 1")
    axes[2, 0].plot(t, f2, "--", label="Gaya pada benda 2")
    axes[2, 0].set(title="Hukum III: Aksi = -Reaksi", xlabel="t (s)", ylabel="F (N)")
    axes[2, 0].legend()
    axes[2, 1].plot(t, v1, label=f"v1 (m = {m1:.0f} kg)")
    axes[2, 1].plot(t, v2, label=f"v2 (m = {m2:.0f} kg)")
    axes[2, 1].set(title="Hukum III: Kecepatan", xlabel="t (s)", ylabel="v (m/s)")
    axes[2, 1].legend()
    axes[2, 2].plot(t, p, label="p total")
    axes[2, 2].set(title="Momentum Total Kekal", xlabel="t (s)", ylabel="p (kg m/s)", ylim=(-5, 5))
    axes[2, 2].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("simulasi_newton.png", dpi=150)
    print("Grafik disimpan ke simulasi_newton.png")

    # Verifikasi numerik singkat
    print(f"[Hukum I]   v akhir tanpa gaya      : {v_a[-1]:.3f} m/s (awal 3.000)")
    for m, (_, _, a) in res.items():
        print(f"[Hukum II]  m = {m} kg -> a = {a[10]:.3f} m/s\u00b2 (F/m = {F/m:.3f})")
    print(f"[Hukum III] selisih gaya aksi+reaksi: {np.max(np.abs(f1 + f2)):.2e} N")
    print(f"[Hukum III] momentum total maks     : {np.max(np.abs(p)):.2e} kg m/s")

    plt.show()


if __name__ == "__main__":
    main()
