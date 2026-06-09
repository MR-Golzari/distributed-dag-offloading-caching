# load packages
import matplotlib as mpl
import matplotlib.pyplot as plt

text='data length'
import numpy as np
fig, ax = plt.subplots()
plt.close('all')
plt.figure()
data={}
data['1'] = 0.04152714621726153
data['2'] = 0.021449062718895764
data['3'] = 0.034446659243260784
data['4'] = 0.04923105335605728

plt.bar(data.keys(),data.values(),width=0.3)
plt.xlabel('Number of hidden layers')
plt.ylabel('Average of finishing time')
plt.grid(axis='y')
#ax.yaxis.set_label_coords(0.63,1.01)
#ax.yaxis.tick_right()
#ax.legend(frameon=False, loc='upper left',ncol=2,handlelength=4)
plt.savefig('plots/hidden layers.pdf',dpi=300)
plt.show()
#telegrambot = telepot.Bot(telegramTOKEN)
#a=telegrambot.getMe()
#telegrambot.sendDocument(chat_id, document=open('plots/test.pdf', "rb"),caption=text)