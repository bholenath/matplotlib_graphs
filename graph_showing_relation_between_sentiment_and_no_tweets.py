import MySQLdb
import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.ticker import FormatStrFormatter

# from scipy.optimize import linegress

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8", use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)


def func1(x, a, b, c):
    return np.log(c + x) - np.log(a + (b * x))


query0 = """select * from college_sentiment_average where no_rows >= 5000 order by no_rows asc"""
cursor1.execute(query0)
fetch_col = cursor1.fetchall()

sen_values = []
count_values = []
col_name = []

for row in fetch_col:
    sen_values.append(row[2])
    count_values.append(row[3])
    col_name.append(row[1][:20])

pearR = np.ma.corrcoef(count_values, sen_values)[1, 0]

f, ax1 = plt.subplots(1, 1, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)
plt.subplots_adjust(bottom=0.19)
# ax1.spines['left'].set_color('blue')
# ax1.set_zorder(1)
# print len(sen_values), " ",len(count_values), " ", len(date_values)
ax1.set_title('Correlation between Tweets Count and Sentiment', y=1.02, style='italic', family='monospace', weight='bold', size='large')

ax1.set_xlabel('Colleges', labelpad=3, family='monospace', weight='bold', size='medium')
ax1.set_ylabel('Colleges_Sentiment', labelpad=10, family='monospace', weight='bold', size='medium', color='b')

# ax1.xaxis.set_major_locator(MaxNLocator(nbins=len(date_values), prune=None))

yaxis_ticks = np.linspace(min(sen_values) - 0.2, max(sen_values) + 0.2, 20)
# minor_ticks = np.linspace(min(sen_values) - 0.1,max(sen_values) + 0.1,40)
ax1.set_yticks(yaxis_ticks)
# ax1.set_yticks(minor_ticks,minor=True)

ax1.set_yticklabels(sen_values,  zorder=1, color='blue', rotation='0',
                    va='center', ha='right', weight='roman', clip_on=True, axes=ax1, size='medium')
ax1.set_ylim(min(sen_values) - 0.2, max(sen_values) + 0.2)

col_arr = np.arange(1, len(col_name) + 1, 1)
x_new = np.linspace(col_arr[0], col_arr[-1], 1000)

try:
    popt1, pcov1 = opt.curve_fit(func1, col_arr, sen_values, p0=(1, 1e-6, 1))
    yLOG = func1(x_new, *popt1)
    plot00 = ax1.plot(x_new, yLOG, 'g--', lw=5, alpha=0.9, label='Curve Log Fit Sentiment', zorder=1)
    # print yLOG
except RuntimeError:
    pass

ax1.set_xticks(col_arr)
ax1.set_xticklabels(col_name,  zorder=1, color='black', rotation='90',
                    va='top', ha='center', weight='roman', clip_on=True, axes=ax1, size='medium')
ax1.set_xlim(col_arr[0] - 1, col_arr[-1] + 1)

ax1.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, colors='blue')
# ax1.tick_params(axis='y', which='minor', direction='out', length=3, width=0.5, labelsize=0, colors='blue')
ax1.tick_params(axis='x', which='both', direction='out', length=6, width=0.5, labelsize=8, colors='black', top='off')

plot_1 = ax1.plot(col_arr, sen_values, color='b', marker='o', linestyle='-', rasterized=True, antialiased=True, label='Tweets_Sentiment', zorder=1)

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1.grid(which='major', color='grey', linestyle='--', linewidth=1, alpha=0.8)
# ax1.grid(which='minor',color='grey', linestyle='--', linewidth=1,alpha=0.4)

ax2 = ax1.twinx()

ax2.set_zorder(1)
# post_addition = np.linspace(-0.4,0.4,len(col_arr))
ax2.set_ylabel('No._of_Tweets', labelpad=10, family='monospace', weight='bold', size='medium', color='r')

ax2.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, colors='red')
# ax2.tick_params(axis='y', which='minor', direction='in', length=2, width=0.5, colors='red')
# ax2.spines['right'].set_color('red')
yaxis_ticks_1 = np.linspace(min(count_values) - 100, max(count_values) + 5, 20)
# minor_ticks_1 = np.linspace(min(count_values) - 1,max(count_values) + 5,60,dtype='int32')
ax2.set_yticks(yaxis_ticks_1)
# ax2.spines['right'].set_color('red')

# try:
#     popt2, pcov2 = opt.curve_fit(func1, col_arr, count_values, p0=(1,1e-6,1))
#     yLOG1 = func1(x_new,*popt2)
#     plot2 = ax2.plot(x_new,yLOG1,'y--',lw=5, alpha=0.9, label='Curve Log Fit Count Tweets',zorder=10)
#     # print yLOG
# except RuntimeError:
#     pass

# ax2.set_yticks(minor_ticks_1,minor=True)
ax2.set_yticklabels(yaxis_ticks_1,  zorder=1, color='red', rotation='0',
                    va='center', ha='left', weight='roman', clip_on=True, axes=ax2, size='medium')
ax2.set_ylim(min(count_values) - 100, max(count_values) + 5)

ax2.bar(col_arr, count_values, width=0.8, color='r', rasterized=True, antialiased=True, label='Tweets_Count', align='center', zorder=0)
ax2.set_xlim(col_arr[0] - 1, col_arr[-1] + 1)

ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

lns = plot_1 + plot00
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center')

ax2.legend(loc='upper left')

text = "Correlation Coefficient : " + str(pearR)
f.text(0.01, 0.95, text, fontsize=14, bbox=dict(facecolor='grey', alpha=0.5, pad=7.0), weight='bold')

plt.show()
# plt.grid()
img_name = 'Correlation between Tweets Count and Sentiment.svg'
f.savefig(img_name, format='svg', dpi=300)
plt.close(f)
