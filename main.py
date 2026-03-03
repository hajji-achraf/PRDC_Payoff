import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

st.set_page_config(page_title="PRDC Swap", layout="wide")
st.title("PRDC Swap — Simulation du Payoff")
st.divider()

# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.header("Parametres du produit")
    S0    = st.slider("S0 — taux de change initial",  50,    200,  110)
    cf    = st.slider("cf — taux etranger (rf)",       0.01,  0.15, 0.05, step=0.005)
    cd    = st.slider("cd — taux domestique (rd)",     0.005, 0.10, 0.02, step=0.005)
    er    = st.slider("er — taux actualisation",       0.00,  0.10, 0.03, step=0.005)
    T     = st.slider("T  — duree (annees)",           1,     20,   10)

    st.divider()
    st.subheader("Simulation FX")
    sigma = st.slider("sigma — volatilite",  0.05, 0.50, 0.15, step=0.01)
    mu    = st.slider("mu — drift",         -0.10, 0.15, 0.03, step=0.005)

    st.divider()
    st.subheader("Barriere KO")
    ko_B  = st.slider("Niveau barriere B", S0, S0 * 3, int(S0 * 1.5), step=1)


# ═══════════════════════════════════════════════
#  FONCTIONS
# ═══════════════════════════════════════════════
def payoff(S, S0, cf, cd):
    return np.maximum(cf * S / S0 - cd, 0.0)


def simulate(S0, mu, sigma, T):
    np.random.seed()
    N    = T * 1000
    dt   = T / N
    t    = np.linspace(0, T, N + 1)
    S    = np.empty(N + 1)
    S[0] = S0
    for i in range(N):
        Z      = np.random.randn()
        S[i+1] = S[i] * (1 + mu * dt + sigma * np.sqrt(dt) * Z)
    return t, S


def cashflows(t, S, S0, cf, cd, er, ko_B):
    rows    = []
    N_steps = len(t) - 1
    T_int   = int(round(t[-1]))

    for i in range(1, T_int + 1):
        idx = int(round(i * N_steps / T_int))
        idx = min(idx, N_steps)
        ti  = t[idx]
        s   = S[idx]

        if s >= ko_B:
            rows.append({
                "Periode":   f"T{i}",
                "t (ans)":   round(ti, 2),
                "S(t)":      round(s, 2),
                "Coupon":    0.0,
                "PV Coupon": 0.0,
                "Etat":      "KO"
            })
            break

        c  = payoff(s, S0, cf, cd)
        pv = c * np.exp(-er * ti)
        rows.append({
            "Periode":   f"T{i}",
            "t (ans)":   round(ti, 2),
            "S(t)":      round(s, 2),
            "Coupon":    round(c, 6),
            "PV Coupon": round(pv, 6),
            "Etat":      "Actif" if c > 0 else "Floor"
        })

    return pd.DataFrame(rows).set_index("Periode")


# ══════════════════��════════════════════════════
#  CALCULS
# ═══════════════════════════════════════════════
S_star       = (cd / cf) * S0
t, S_path    = simulate(S0, mu, sigma, T)
df           = cashflows(t, S_path, S0, cf, cd, er, ko_B)
ko_row       = df[df["Etat"] == "KO"]
ko_date      = ko_row.iloc[0]["t (ans)"] if not ko_row.empty else None
total_coupon = df["Coupon"].sum()
total_pv     = df["PV Coupon"].sum()
flux_dates   = np.arange(1, T + 1)


# ═══════════════════════════════════════════════
#  METRIQUES
# ═══════════════════════════════════════════════





# ═══════════════════════════════════════════════
#  SECTION 1 — PAYOFF
# ═══════════════════════════════════════════════
st.subheader("Section 1 — Structure du Payoff")

st.latex(r"C_i = \max\!\left( c_f \cdot \frac{S(t_i)}{S_0} - c_d \;,\; 0 \right)")

st.info(
    "- cf = taux etranger rf — pente du payoff\n\n"
    "- cd = taux domestique rd — plancher\n\n"
    "- S(ti) = taux de change a la date ti\n\n"
    "- S0 = taux de change initial\n\n"
    "- S* = (cd/cf) x S0 = seuil de declenchement\n\n"
    "- B = barriere KO — si S(ti) >= B, produit desactive"
)

st.success(f"S* = {S_star:.2f}  |  Barriere B = {ko_B}")

S_range   = np.linspace(S0 * 0.3, S0 * 2.5, 400)
C_avec_ko = np.where(S_range >= ko_B, 0.0, payoff(S_range, S0, cf, cd))

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.fill_between(S_range, C_avec_ko,
                 where=((S_range >= S_star) & (S_range < ko_B)),
                 color="teal", alpha=0.25, label="Zone de profit")
ax1.plot(S_range, C_avec_ko, color="teal", linewidth=2.5, label="Payoff PRDC")
ax1.axvline(S_star, color="gold",      linestyle="--", label=f"S* = {S_star:.2f}")
ax1.axvline(S0,     color="steelblue", linestyle=":",  label=f"S0 = {S0}")
ax1.axvline(ko_B,   color="red",       linestyle="-.", linewidth=2, label=f"KO = {ko_B}")
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.set_xlabel("S")
ax1.set_ylabel("Coupon Ci")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
st.pyplot(fig1)

st.divider()


# ═══════════════════════════════════════════════
#  SECTION 2 — SIMULATION S(t)
# ═══════════════════════════════════════════════
st.subheader("Section 2 — Simulation du taux de change S(t)")

st.latex(r"S_{i+1} = S_i \left(1 + \mu\,\Delta t + \sigma\sqrt{\Delta t}\,Z_i\right), \quad Z_i \sim \mathcal{N}(0,1)")

st.info(
    "- mu = drift annuel\n\n"
    "- sigma = volatilite annuelle\n\n"
    "- N = T x 1000 pas (discretisation fine)"
)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(t, S_path, color="teal", linewidth=1.5, label="S(t)")

for i, ti in enumerate(flux_dates):
    if ti <= t[-1]:
        idx  = int(round(ti * (len(t) - 1) / T))
        idx  = min(idx, len(S_path) - 1)
        s_ti = S_path[idx]
        ax2.axvline(ti, color="purple", linestyle=":", linewidth=1.2, alpha=0.6)
        ax2.scatter([ti], [s_ti], color="white", s=80, zorder=6,
                    edgecolors="purple", linewidth=2)
        ax2.annotate(f"T{i+1}", xy=(ti, s_ti),
                     xytext=(ti + 0.05, s_ti + S0 * 0.03),
                     fontsize=9, color="purple", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color="purple", lw=0.8))

ax2.axhline(S_star, color="gold",      linestyle="--", label=f"S* = {S_star:.1f}")
ax2.axhline(S0,     color="steelblue", linestyle=":",  label=f"S0 = {S0}")
ax2.axhline(ko_B,   color="red",       linestyle="-.", linewidth=2, label=f"KO = {ko_B}")
if ko_date:
    ax2.axvline(ko_date, color="red", linewidth=2,
                label=f"KO a t = {ko_date} ans")
ax2.set_xlabel("Temps (annees)")
ax2.set_ylabel("S(t)")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3)
st.pyplot(fig2)

st.divider()


# ═══════════════════════════════════════════════
#  SECTION 3 — COUPONS ANNUELS + TABLEAU
# ═════════════════════��═════════════════════════
st.subheader("Section 3 — Coupons annuels et Valeur du Swap")

st.latex(r"\text{PV}_i = C_i \cdot e^{-er \cdot t_i}")



st.markdown("#### Tableau des coupons annuels")
st.dataframe(
    df.style
      .format({"t (ans)": "{:.2f}", "S(t)": "{:.2f}",
               "Coupon": "{:.6f}", "PV Coupon": "{:.6f}"})
      .applymap(
          lambda v: "color: teal" if isinstance(v, float) and v > 0
               else "color: red"  if isinstance(v, float) and v == 0
               else "",
          subset=["Coupon"]),
    use_container_width=True,
)

st.markdown("---")

if ko_date:
    st.error(f"Produit KO a t = {ko_date} ans")
else:
    st.success("Produit actif jusqu'a maturite")
