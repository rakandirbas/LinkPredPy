import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import numpy as np
from mpltools import style
style.use('ggplot')


pr_dic = {"x": 0.3, "y": 0.6, "z": 0.8}
v1dataname = "AU-PR"
all_keys = ['x', "y", "z"]
pr_data = []


for key in all_keys:
    pr_data.append( pr_dic[key] )


pr_data = np.array(pr_data)

d = {v1dataname: Series(pr_data, index=all_keys)}

means = pd.DataFrame( d )

means.plot(kind='bar')

# plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

# plt.tight_layout()
plt.show()
# plt.savefig(filename)