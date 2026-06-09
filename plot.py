# load packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import telepot
telegramTOKEN = '5695741795:AAEhW_fX7sVAGolFamGgCZbi8MPii_l0MQ4'
import sys
import time
import telepot
chat_id = 63726018
text='data length'
import numpy as np
fig, ax = plt.subplots()
plt.close('all')
plt.figure()
data={}
data[1000] = 0.0051638631492122306
data[2000] = 0.005254690739202953
data[3000] = 0.005400548028290063
data[5000] = 0.006802678904417603
data[8000] = 0.005557449372505357
data[10000] = 0.006925031006641883
data[20000]= 0.007154005401984636
data[50000] = 0.0084732318972424
data[60000] = 0.009890225086661448
data[60000] = 0.009890225086661448
data[70000] = 0.011983961245805624
data[80000] = 0.012450280500146006
data[90000] = 0.012450280500146006
data[100000]= 0.012450280500146006
data[110000]= 0.014925421026211513
data[120000]= 0.015703190846067624
data[130000]= 0.01766779470460155
data[140000]= 0.017140910544692883
data[150000]= 0.01747861806492742
data[160000]= 0.02278732893752493
data[170000]= 0.02278732893752493
data[180000]= 0.02278732893752493
data[190000]= 0.02278732893752493
data[200000]=0.02278732893752493
#data[300000]=0.03396846544858323
#data[500000] = 0.056420899767309296
#data[800000]= 0.0635649549628059
#data[1000000] = 0.09064048575006717
#data[1200000] = 0.10292462584255865
#data[1500000] = 0.16251801879882288
#data[2000000] = 0.22693456204322354
plt.plot(data.keys(),data.values(), '*-')
plt.xlabel('Data length')
plt.ylabel('Average of finish time')
plt.grid()
ax.yaxis.set_label_coords(0.63,1.01)
ax.yaxis.tick_right()
ax.legend(frameon=False, loc='upper left',ncol=2,handlelength=4)
plt.savefig('plots/test.pdf',dpi=300)

plt.show()
telegrambot = telepot.Bot(telegramTOKEN)
a=telegrambot.getMe()
telegrambot.sendDocument(chat_id, document=open('plots/test.pdf', "rb"),caption=text)