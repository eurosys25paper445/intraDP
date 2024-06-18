import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


time = 10

_bw = np.clip(np.random.normal(8, 10, time), 3, 18) * 8
bw = []
last = np.random.randint(1, 5, time)
for __bw, _last in zip(_bw, last):
    bw += [__bw] * _last


ours = {0:   0.9656, 1:   0.9656, 2:   0.9106, 3:   0.6974, 4:   0.6557, 5:   0.5320, 6:   0.4657, 7:   0.4657, 8:   0.4457, 9:   0.4015, 10:   0.3662, 11:   0.3373, 12:   0.3132, 13:   0.2928, 14:   0.2753, 15:   0.2602, 16:   0.2469, 17:   0.2352, 18:   0.2346, 19:   0.2346, 20:   0.2271, 21:   0.2187, 22:   0.2110, 23:   0.2040, 24:   0.1976, 25:   0.1917, 26:   0.1863, 27:   0.1813, 28:   0.1766, 29:   0.1722, 30:   0.1682}

DC = {0:  0.9664, 1:  0.9664, 2:  0.9664, 3:  0.9664, 4:  0.9317,5:  0.7549,6:  0.6370,7:  0.5529,8:  0.4897,9:  0.4406,10:  0.4013,11:  0.3692,12:  0.3424,13:  0.3198,14:  0.3003,15:  0.2835,16:  0.2688,17:  0.2558,18:  0.2442,19:  0.2339,20:  0.2246,21:  0.2162,22:  0.2085,23:  0.2015,24:  0.1951,25:  0.1892,26:  0.1838,27:  0.1787,28:  0.1741,29:  0.1697,30:  0.1656}

SG = {0: 263.8813,1: 1.2353,2: 1.1038,3: 1.0600,4: 1.0381,5: 1.0250,6: 1.0162,7: 0.5600,8: 0.4968,9: 0.4477,10: 0.4085,11: 0.3763,12: 0.3495,13: 0.3269,14: 0.3074,15: 0.2906,16: 0.2759,17: 0.2629,18: 0.2513,19: 0.2410,20: 0.2317,21: 0.2233,22: 0.2156,23: 0.2086,24: 0.2022,25: 0.1963,26: 0.1909,27: 0.1858,28: 0.1812,29: 0.1768,30: 0.1728,}

ours_xy = []
DC_xy = []
SG_xy = []
def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

def fill_x_y(l, bw, d, _time):
    bw = np.clip(bw/8, 0, 30)
    if len(l) == 0:
        x = np.random.randint(0, 30) / 10.
        y = d[int(bw)] + 0.12
        l.append([x, y])
        return [[x, y]]
    else:
        rets = []
        while sum(l[-1]) < _time:
            x = sum(l[-1]) + np.random.randint(0, 10) / 10.
            y = d[int(bw)] + np.random.normal(0, 0.05)
            l.append([x, y])
            rets.append([x, y])
        return rets
    


data = []
index = []
for i, _bw in enumerate(bw):
    for _l, _d, _idx in zip([ours_xy, DC_xy, SG_xy], [ours, DC, SG], ["Intra-DP", "DSCCS", "SPSO-GA"]):
        ret = fill_x_y(_l, _bw, _d, i)
        if ret:
            for _ret in ret:
                data.append([_idx] + _ret)

df = pd.DataFrame(data, columns=["System", "Time/s", "Inference time/s"])

sns.set_context("poster")
sns.set_style(style="white")
sns.set_color_codes(palette="deep")
fig, ax1 = plt.subplots(figsize=(12, 6))
p = sns.lineplot(x="Time/s", y="Inference time/s", marker="o", hue="System", data=df)
ax2 = ax1.twinx()
ax2.set_ylabel("Bandwidth(Mb/s)", color="red")
sns.lineplot(x = np.arange(len(bw)), y=bw, color = 'red', ax = ax2, linestyle='--')
sns.move_legend(ax1, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)

plt.title("Micro-Event of Inference Time against Bandwidth of Different Systems", y=1.15)

plt.savefig("MicroEvent.png", bbox_inches="tight")
plt.show()






