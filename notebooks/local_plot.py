#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:53:08 2026

@author: m484s199
"""

import matplotlib.pyplot as plt
import pandas as pd

surface_df = pd.read_csv('./petermann_20100420_03_surface.csv')
bottom_df = pd.read_csv('./petermann_20100420_03_bottom.csv')
hab_df = pd.read_csv('./petermann_20100420_03_HAB.csv')

#%%

# Plot layers using elevation data and slow_time
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
fig.tight_layout(pad=4.0)


# surface_df['wgs84'].plot(ax=ax1, x='along_track', linewidth=1, linestyle=':', label='surface')
# bottom_df['wgs84'].plot(ax=ax1, x='along_track', linewidth=1, linestyle=':', label='bottom')
ax1.plot(surface_df['along_track'], surface_df['wgs84'])
ax1.plot(bottom_df['along_track'], bottom_df['wgs84'])

gp_along_track = 93635.19516029 
gp_z = -425.34510601
ax1.scatter(gp_along_track, gp_z, color='r', s=20, label="Grounding Point")
ax1.set_title('Petermann Grounding Point Example 2010-04-20')

ax1.axvspan(95379.42502606, 97889.53351504, color='tab:green',
            alpha=0.5, label='2011-2015 gz')

ax2.plot(hab_df['along_track'], hab_df['wgs84'])
ax2.axvspan(95379.42502606, 97889.53351504, color='tab:green',
            alpha=0.5, label='2011-2015 gz')

ax2.set_title('HAB')
ax1.legend()

fig.savefig('/home/m484s199/grounding_point_figs/petermann_20100420_example_corrected.png', dpi=300)