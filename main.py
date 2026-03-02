
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="PRDC Swap", layout="wide")
st.title("📊 PRDC Swap — Simulation du Payoff")
st.divider()



# ═══════════════════════════════════════════════
with st.sidebar:

    st.header("🔧 Paramètres du produit")
    S0    = st.slider("S₀ — taux de change initial",  50,    200,  110)
    cf    = st.slider("cf — coupon étranger",          0.01,  0.15, 0.05, step=0.005)
    cd    = st.slider("cd — coupon domestique",        0.005, 0.10, 0.02, step=0.005)
    er    = st.slider("er — taux d'actualisation",     0.00,  0.10, 0.03, step=0.005)
    T     = st.slider("T  — durée (années)",           1,     20,   10)

    st.divider()
    st.subheader("📐 Simulation FX")
    sigma = st.slider("σ — volatilité",  0.05, 0.50, 0.15, step=0.01)
    mu    = st.slider("μ — drift",      -0.10, 0.15, 0.03, step=0.005)
    seed  = 42

    st.divider()
    st.subheader("🚧 Barrière KO")
    ko_B  = st.slider("Niveau barrière B", S0, S0 * 3, int(S0 * 1.5), step=1)
    st.caption(f"Si S(t) ≥ **{ko_B}** → produit arrêté, plus de coupons")


# ═══════════════════════════════════════════════
#  FONCTIONS
# ═══════════════════════════════════════════════
def payoff(S, S0, cf, cd):
    """Ci = max( cf * S/S0 - cd , 0 )"""
    return np.maximum(cf * S / S0 - cd, 0.0)


def simulate(S0, mu, sigma, T, seed):
    """
    Euler-Maruyama mensuel.
    t[0]=0 → début contrat (pas de coupon)
    t[1..N] → fin de chaque mois → coupon possible
    """
    np.random.seed(int(seed))
    N         = T * 12
    dt        = 1.0 / 12
    t         = np.linspace(0, T, N + 1)
    S         = np.empty(N + 1)
    S[0]      = S0
    for i in range(N):
        Z      = np.random.randn()
        S[i+1] = S[i] * (1 + mu * dt + sigma * np.sqrt(dt) * Z)
    return t, S


def cashflows(t, S, S0, cf, cd, er, ko_B):
    """
    Calcule les coupons mois par mois.
    Commence à i=1 (jamais i=0).
    S'arrête si S(t) >= ko_B (KO déclenché).
    """
    rows = []
    for i in range(1, len(t)):
        s = S[i]

        if s >= ko_B:                          # ← KO toujours actif
            rows.append({
                "Période":   f"Mois {i}",
                "t (ans)":   round(t[i], 4),
                "t (mois)":  i,
                "S(t)":      round(s, 2),
                "Coupon":    0.0,
                "PV Coupon": 0.0,
                "État":      "KO ❌"
            })
            break

        c  = payoff(s, S0, cf, cd)
        pv = c * np.exp(-er * t[i])
        rows.append({
            "Période":   f"Mois {i}",
            "t (ans)":   round(t[i], 4),
            "t (mois)":  i,
            "S(t)":      round(s, 2),
            "Coupon":    round(c, 6),
            "PV Coupon": round(pv, 6),
            "État":      "✅ Actif" if c > 0 else "🟡 Floor"
        })

    return pd.DataFrame(rows).set_index("Période")


# ═══════════════════════════════════════════════
#  CALCULS  (automatiques à chaque changement)
# ═══════════════════════════════════════════════
S_star = (cd / cf) * S0
t, S_path = simulate(S0, mu, sigma, T, seed)
df = cashflows(t, S_path, S0, cf, cd, er, ko_B)

ko_row  = df[df["État"] == "KO ❌"]
ko_date = ko_row.iloc[0]["t (ans)"] if not ko_row.empty else None

total_coupon = df["Coupon"].sum()
total_pv     = df["PV Coupon"].sum()


# ═══════════════════════════════════════════════
#  MÉTRIQUES
# ═══════════════════════════════════════════════
m1, m2, m3, m4 = st.columns(4)
m1.metric("Seuil S*",        f"{S_star:.2f}")
m2.metric("Σ Coupons bruts", f"{total_coupon:.5f}")
m3.metric("PV Coupons",      f"{total_pv:.5f}")

st.divider()


# ═══════════════════════════════════════════════
#  ONGLETS
# ═══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 Payoff", "📉 Trajectoire S(t)", "💰 Coupons"])


# ───────────────────────────────────────────────
#  ONGLET 1 — Payoff
# ───────────────────────────────────────────────
# ── Données du graphe ────────────────────────
S_range = np.linspace(S0 * 0.3, S0 * 2.5, 400)
C_range = payoff(S_range, S0, cf, cd)

# ✅ Après la barrière B → coupon = 0 (produit arrêté)
C_avec_ko = np.where(S_range >= ko_B, 0.0, C_range)

fig1, ax1 = plt.subplots(figsize=(8, 4))



# Zone teal — zone de profit (S* < S < B)
ax1.fill_between(S_range, C_avec_ko,
                 where=((S_range >= S_star) & (S_range < ko_B)),
                 color="teal", alpha=0.25,
                 label="Zone de profit  (Cᵢ > 0)")



# Courbe payoff avec arrêt à B
ax1.plot(S_range, C_avec_ko, color="teal", linewidth=2.5, label="Payoff PRDC")





# Lignes de référence
ax1.axvline(S_star, color="gold",      linestyle="--", label=f"Seuil S* = {S_star:.2f}")
ax1.axvline(S0,     color="steelblue", linestyle=":",  label=f"S₀ = {S0}")
ax1.axvline(ko_B,   color="red",       linestyle="-.", linewidth=2,
            label=f"Barrière KO = {ko_B}")
ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.4)

ax1.set_xlabel("Taux de change S")
ax1.set_ylabel("Coupon Cᵢ")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
st.pyplot(fig1)

# ───────────────────────────────────────────────
#  ONGLET 2 — Trajectoire S(t)
# ───────────────────────────────────────────────
with tab2:

    st.subheader("Trajectoire S(t) — Euler-Maruyama mensuel")

    st.latex(r"""
        S_{i+1} = S_i \Bigl(1 + \mu\,\Delta t + \sigma\sqrt{\Delta t}\,Z_i\Bigr),
        \quad Z_i \sim \mathcal{N}(0,1), \quad \Delta t = \text{un mois}
    """)

    st.info(
        r"- $\mu$ = drift annuel du FX" "\n\n"
        r"- $\sigma$ = volatilité annuelle" "\n\n"
        r"- $Z_i \sim \mathcal{N}(0,1)$ = choc aléatoire mensuel"
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t, S_path, color="teal", linewidth=2, label="S(t)")
    
    df_obs = df[df["État"] != "KO ❌"]
    ax2.scatter(df_obs["t (ans)"], df_obs["S(t)"],
                color="teal", s=15, alpha=0.4, zorder=4,
                label="Dates d'observation")
    ax2.axhline(S_star, color="gold",      linestyle="--", label=f"Seuil S* = {S_star:.1f}")
    ax2.axhline(S0,     color="steelblue", linestyle=":",  label=f"S₀ = {S0}")
    ax2.axhline(ko_B,   color="red",       linestyle="-.", linewidth=2,
                label=f"Barrière KO = {ko_B}")
    if ko_date:
        ax2.axvline(ko_date, color="red", linewidth=2,
                    label=f"KO à t = {ko_date} ans")
    ax2.set_xlabel("Temps (années)")
    ax2.set_ylabel("S(t)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)


# ───────────────────────────────────────────────
#  ONGLET 3 — Coupons
# ───────────────────────────────────────────────
with tab3:

    st.subheader("Coupons reçus — ce scénario")
    

    st.latex(r"""
        C_i = \max\!\left( c_f \cdot \frac{S(t_i)}{S_0} - c_d \;,\; 0 \right)
        \qquad
        \text{PV}_i = C_i \cdot e^{-er \cdot t_i}
    """)

    st.info(
        r"- $S(t_i) \geq S^*$ → coupon **positif**" "\n\n"
        r"- $S(t_i) < S^*$   → coupon **nul** (floor)" "\n\n"
        r"- $S(t_i) \geq B$  → produit **KO**, arrêt immédiat"
    )

    # Graphe coupons
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    colors = ["red" if c == 0 else "teal" for c in df["Coupon"]]
    ax3.bar(df["t (ans)"], df["Coupon"], color=colors, width=0.07)
    ax3.axvline(0, color="steelblue", linestyle=":", linewidth=1.5,
                label="t=0 : début contrat")
    if ko_date:
        ax3.axvline(ko_date, color="red", linewidth=2,
                    label=f"KO à t = {ko_date} ans")
    ax3.set_xlabel("Temps (années)")
    ax3.set_ylabel("Coupon C(t)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")
    st.pyplot(fig3)

    # Tableau
    st.markdown("#### 📋 Tableau des cashflows")
    st.dataframe(
        df.style
          .format({
              "t (ans)":   "{:.4f}",
              "t (mois)":  "{:.0f}",
              "S(t)":      "{:.2f}",
              "Coupon":    "{:.6f}",
              "PV Coupon": "{:.6f}",
          })
          .applymap(
              lambda v: "color: teal" if isinstance(v, float) and v > 0
                   else "color: red"  if isinstance(v, float) and v == 0
                   else "",
              subset=["Coupon"]
          ),
        use_container_width=True,
        height=400,
    )

    

    if ko_date:
        st.error(f"🔴 Produit KO à t = {ko_date} ans — plus de coupons après.")
    else:
        st.success("✅ Produit actif jusqu'à maturité.")
