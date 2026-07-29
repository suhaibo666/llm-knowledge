# -*- coding: utf-8 -*-
"""生成 SiTU-GLU 值域图（本库自绘）。
公式来源：Kimi K3 Technical Report 0797decb §2.3.2 / Appendix B Eq.18-19（pp.7-8, 43）。
输出：assets/kimi_k3_fig_situ_range.svg + .png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

B1, B2 = 4.0, 25.0
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
cap = lambda z, b: b * np.tanh(z / b)

gate_situ = lambda a: cap(a, B1) * sig(a)
gate_swish = lambda a: a * sig(a)
up_situ = lambda u: cap(u, B2)
f_situ = lambda x: gate_situ(x) * up_situ(x)
f_swiglu = lambda x: gate_swish(x) * x

C_SW, C_SI, C_BOUND, C_GRID = "#d1495b", "#1b6ca8", "#8d99ae", "#e6e6e6"

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
fig.subplots_adjust(hspace=0.34, wspace=0.24, left=0.07, right=0.975, top=0.90, bottom=0.075)

# ---------------- A: gate branch ----------------
ax = axes[0, 0]
a = np.linspace(-6, 20, 3000)
ax.plot(a, gate_swish(a), color=C_SW, lw=2, label=r"SwiGLU 门支  $a\,\sigma(a)$   （上界 $+\infty$）")
ax.plot(a, gate_situ(a), color=C_SI, lw=2, label=r"SiTU 门支  $\beta_1\tanh(a/\beta_1)\,\sigma(a)$")
ax.axhline(B1, color=C_BOUND, ls="--", lw=1.2)
ax.text(19.5, B1 + 0.45, r"$\beta_1=4$（上确界，取不到）", ha="right", color="#5a6472", fontsize=8.8)
ax.axhline(0, color="#bbb", lw=0.8)
ax.plot([-1.2187], [-0.269769], "o", color=C_SI, ms=5)
ax.plot([-1.2785], [-0.278465], "o", color=C_SW, ms=5)
ax.annotate("负向下确界几乎没变\n−0.2785 → −0.2698", xy=(-1.25, -0.274), xytext=(0.4, -3.1),
            fontsize=8.6, color="#3d4450",
            arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=1))
ax.set_xlim(-6, 20); ax.set_ylim(-4.2, 12)
ax.set_title("A  门支：cap 只压正向，负尾原样保留", loc="left", fontweight="bold")
ax.set_xlabel(r"门支预激活 $a=W_g x$"); ax.set_ylabel("门支输出")
ax.legend(loc="upper left", fontsize=8.6, framealpha=0.95)
ax.grid(color=C_GRID, lw=0.7)

# ---------------- B: up branch ----------------
ax = axes[0, 1]
u = np.linspace(-70, 70, 3000)
ax.plot(u, u, color=C_SW, lw=2, label=r"SwiGLU up 支  $u$   （双向无界）")
ax.plot(u, up_situ(u), color=C_SI, lw=2, label=r"SiTU up 支  $\beta_2\tanh(u/\beta_2)$")
for s in (1, -1):
    ax.axhline(s * B2, color=C_BOUND, ls="--", lw=1.2)
ax.text(69, B2 + 3, r"$\pm\beta_2=\pm 25$", ha="right", color="#5a6472", fontsize=8.8)
ax.axhline(0, color="#bbb", lw=0.8); ax.axvline(0, color="#bbb", lw=0.8)
ax.plot([B2], [up_situ(B2)], "o", color=C_SI, ms=5)
ax.annotate(r"$u=\beta_2$ 处只剩线性值的 76.2%", xy=(B2, up_situ(B2)), xytext=(-2, -54),
            fontsize=8.6, color="#3d4450",
            arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=1))
ax.set_xlim(-70, 70); ax.set_ylim(-70, 70)
ax.set_title("B  up 支：双向对称 cap，两支都不许独大", loc="left", fontweight="bold")
ax.set_xlabel(r"up 支预激活 $u=W_u x$"); ax.set_ylabel("up 支输出")
ax.legend(loc="upper left", fontsize=8.6, framealpha=0.95)
ax.grid(color=C_GRID, lw=0.7)

# ---------------- C: scalar response ----------------
ax = axes[1, 0]
x = np.linspace(-12, 60, 6000)
ax.plot(x, f_swiglu(x), color=C_SW, lw=2, label=r"SwiGLU（$\sim x^2$，无界）")
ax.plot(x, f_situ(x), color=C_SI, lw=2, label=r"SiTU-GLU（$\to \beta_1\beta_2=100$）")
ax.axhline(100, color=C_BOUND, ls="--", lw=1.3)
ax.text(-11, 113, r"$|f(x)|\leq\beta_1\beta_2=100$  (Eq. 19)", ha="left", color="#5a6472", fontsize=8.8)
ax.axhline(0, color="#bbb", lw=0.8)
ax.plot([10.0], [100.0], "o", color=C_SW, ms=5)
ax.annotate("SwiGLU 在 x=10 就已经到 100\n此后继续按 $x^2$ 涨", xy=(10, 100), xytext=(23, 36),
            fontsize=8.6, color="#3d4450",
            arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=1))
ax.set_xlim(-12, 60); ax.set_ylim(-28, 360)
ax.set_title("C  标量响应（两支同一输入，对应报告 Fig. 4 口径）", loc="left", fontweight="bold")
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$f(x)$")
ax.legend(loc="upper left", fontsize=8.8, framealpha=0.95)
ax.grid(color=C_GRID, lw=0.7)

axin = ax.inset_axes([0.585, 0.56, 0.385, 0.38])
xi = np.linspace(-4, 2.2, 1500)
axin.plot(xi, f_swiglu(xi), color=C_SW, lw=1.6)
axin.plot(xi, f_situ(xi), color=C_SI, lw=1.6, ls="--")
axin.axhline(0, color="#bbb", lw=0.7)
axin.set_title("原点附近：一阶重合", fontsize=8.2, pad=2)
axin.tick_params(labelsize=7.2); axin.grid(color=C_GRID, lw=0.6)

# ---------------- D: value-range ladder ----------------
ax = axes[1, 1]
rows = [
    ("输出  $f(x)$",        (-1e9, 1e9),   (-100, 100),      "有界化：$\\pm\\beta_1\\beta_2$"),
    ("up 支",               (-1e9, 1e9),   (-B2, B2),        "$\\pm\\beta_2$"),
    ("门支",                (-0.2785, 1e9), (-0.2698, B1),   "上确界 $\\beta_1$；下确界几乎不动"),
    ("预激活 $W_gx,\\;W_ux$", (-1e9, 1e9),  (-1e9, 1e9),     "不变（cap 作用在其后）"),
]
ylab = []
for i, (name, sw, si, note) in enumerate(rows):
    y = i
    ylab.append(name)
    ax.plot([max(sw[0], -600), min(sw[1], 600)], [y + 0.17] * 2, color=C_SW, lw=6, solid_capstyle="butt", alpha=0.85)
    ax.plot([si[0], si[1]], [y - 0.17] * 2, color=C_SI, lw=6, solid_capstyle="butt")
    if sw[1] >= 1e8:
        ax.add_patch(FancyArrowPatch((520, y + 0.17), (610, y + 0.17), arrowstyle="-|>",
                                     mutation_scale=11, color=C_SW, lw=0))
    if sw[0] <= -1e8:
        ax.add_patch(FancyArrowPatch((-520, y + 0.17), (-610, y + 0.17), arrowstyle="-|>",
                                     mutation_scale=11, color=C_SW, lw=0))
    if si[1] >= 1e8:
        ax.add_patch(FancyArrowPatch((520, y - 0.17), (610, y - 0.17), arrowstyle="-|>",
                                     mutation_scale=11, color=C_SI, lw=0))
        ax.add_patch(FancyArrowPatch((-520, y - 0.17), (-610, y - 0.17), arrowstyle="-|>",
                                     mutation_scale=11, color=C_SI, lw=0))
    ax.text(0, y + 0.355, note, fontsize=8.2, color="#3d4450", ha="center")

for v, lab in ((100, "100"), (-100, "−100"), (25, "25"), (-25, "−25"), (4, "4")):
    ax.axvline(v, color=C_BOUND, ls=":", lw=0.9, zorder=0)
ax.set_xscale("symlog", linthresh=1.0, linscale=0.55)
ax.set_xlim(-700, 700)
ax.set_xticks([-100, -25, -1, 0, 1, 4, 25, 100])
ax.set_xticklabels(["−100", "−25", "−1", "0", "1", "4", "25", "100"])
ax.set_yticks(range(len(rows))); ax.set_yticklabels(ylab)
ax.set_ylim(-1.15, len(rows) - 0.28)
ax.set_title("D  值域阶梯（横轴 symlog；箭头表示无界）", loc="left", fontweight="bold")
ax.set_xlabel("取值区间")
ax.grid(axis="x", color=C_GRID, lw=0.7)
h1, = ax.plot([], [], color=C_SW, lw=6, label="SwiGLU")
h2, = ax.plot([], [], color=C_SI, lw=6, label="SiTU-GLU")
ax.legend(handles=[h1, h2], loc="lower right", fontsize=8.8, framealpha=0.95)

fig.suptitle(r"SiTU-GLU：把 SwiGLU 的“两个无界因子相乘”改成“两个有界因子相乘”，"
             r"输出值域从 $\mathbb{R}$ 收到 $(-\beta_1\beta_2,\;\beta_1\beta_2)=(-100,100)$",
             fontsize=12, y=0.965, fontweight="bold")

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kimi_k3_fig_situ_range")
fig.savefig(out + ".svg", format="svg", bbox_inches="tight")
fig.savefig(out + ".png", format="png", dpi=170, bbox_inches="tight")
print("written:", out + ".svg / .png")
