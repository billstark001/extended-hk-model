import numpy as np
import matplotlib.pyplot as plt

# drafts


data_ = [x for x in data[9::9] if x['p_last'] >= -0.6]
plt.scatter([x['steepness']for x in data_], [x['active_step'] for x in data_], s=2)
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.show()