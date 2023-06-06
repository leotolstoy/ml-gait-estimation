import numpy as np
import matplotlib.pyplot as plt
phase = np.linspace(0,2,1000)
cp = np.cos(2*np.pi*(phase-0.5))
sp = np.sin(2*np.pi*(phase-0.5))

x = np.arctan2(sp,cp)
print(x)
phase_p = ((x)/(2*np.pi)) + 0.5
print(phase_p)

fig, axs = plt.subplots(2,1)
axs[0].plot(2*np.pi*phase, phase%1)
axs[0].plot(2*np.pi*phase, phase_p)

axs[1].plot(2*np.pi*phase, cp)
axs[1].plot(2*np.pi*phase, sp)
plt.show()