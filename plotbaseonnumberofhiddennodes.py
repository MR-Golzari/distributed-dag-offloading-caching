# load packages
import matplotlib as mpl
import matplotlib.pyplot as plt

text='data length'
import numpy as np
fig, ax = plt.subplots()
plt.close('all')
plt.figure()
data={}
data['32'] = 0.027442307967813738
data['64'] = 0.020512604922825278
data['128'] = 0.03436425395010053
data['256'] = 0.059534102763025315
data['512'] = 0.03929583853445298
data['1024'] = 0.03979954762790919

plt.bar(data.keys(),data.values(),width=0.3)
plt.xlabel('Number of hidden nodes')
plt.ylabel('Average of finishing time')
plt.grid(axis='y')
#ax.yaxis.set_label_coords(0.63,1.01)
#ax.yaxis.tick_right()
#ax.legend(frameon=False, loc='upper left',ncol=2,handlelength=4)
plt.savefig('plots/hidden nodes.pdf',dpi=300)
plt.show()
#telegrambot = telepot.Bot(telegramTOKEN)
#a=telegrambot.getMe()
#telegrambot.sendDocument(chat_id, document=open('plots/test.pdf', "rb"),caption=text)