# load packages
import matplotlib as mpl
import matplotlib.pyplot as plt

text='data length'
import numpy as np
fig, ax = plt.subplots()
plt.close('all')
plt.figure()
data={}
data['0.00001'] = 2.0310091611658443
data['0.0001'] = 1.0346785033128794
data['0.001'] = 0.02833301941336808
data['0.01'] = 0.048563203140245745
data['0.1'] = 2.0079935122506436


plt.bar(data.keys(),data.values(),width=0.3)
plt.xlabel('Learning rate')
plt.ylabel('Average of finishing time')
plt.yscale('log')
plt.grid(axis='y')
#ax.yaxis.set_label_coords(0.63,1.01)
#ax.yaxis.tick_right()
#ax.legend(frameon=False, loc='upper left',ncol=2,handlelength=4)
plt.savefig('plots/learningrates.pdf',dpi=300)
plt.show()
#telegrambot = telepot.Bot(telegramTOKEN)
#a=telegrambot.getMe()
#telegrambot.sendDocument(chat_id, document=open('plots/test.pdf', "rb"),caption=text)