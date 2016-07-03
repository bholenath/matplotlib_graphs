import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import matplotlib.finance as fin
# import sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from operator import itemgetter
# import random
# import matplotlib
# import csv

import MySQLdb

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8",
                        use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)

cnxn = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="college_info", charset="utf8",
                       use_unicode=True)
cursor = cnxn.cursor()
cnxn.autocommit(True)

query1 = "set names 'utf8mb4'"
query2 = "SET CHARACTER SET utf8mb4"

# making database accept utf8mb4 as the data format in their columns
cursor1.execute(query1)
cursor1.execute(query2)

# prev = 0
# count = 0
# all_college_sentiments = []
# query0 = "select count(id) from college_sentiment_average"
# cursor1.execute(query0)
# rows_count = cursor1.fetchone()
#
# for row in rows_count:
#     while prev < row:
query = "select college_name,sentiment_average from college_sentiment_average order by id"
cursor1.execute(query)
sent_val = cursor1.fetchall()
# all_college_sentiments.append(sent_val)
# prev += 30

sentiment_score = []
complete_aspects_score = []
college_name = []
# aspects_value = []

# query00 = "show columns from pub_college_aspects_ranking where Type = 'float(3,2)'"
# cursor.execute(query00)
# number_aspects = cursor.rowcount
#
# for name in cursor:
#     aspects_value.append(name[0])

for row in sent_val:

    sub_aspects_score = []

    query11 = """select * from pub_college_aspects_ranking where replace(replace(replace(replace(replace
                 (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
    cursor.execute(query11)
    val_pub = cursor.fetchone()

    if val_pub is not None:
        sentiment_score.append(row[1])
        # aspects_value.append(val_pub[0])
        sub_aspects_score.extend(
            [val_pub[2], val_pub[3], val_pub[4], val_pub[5], val_pub[6], val_pub[7], val_pub[8], val_pub[9], val_pub[10],
             val_pub[11], val_pub[12], val_pub[13], val_pub[14], val_pub[15], val_pub[16], val_pub[17], val_pub[18], val_pub[19],
             val_pub[20]])
        college_name.append(row[0][:20])

    else:
        query12 = """select * from pvt_college_aspects_ranking where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query12)
        val_pvt = cursor.fetchone()

        if val_pvt is not None:
            sentiment_score.append(row[1])
            sub_aspects_score.extend(
                [val_pvt[2], val_pvt[3], val_pvt[4], val_pvt[5], val_pvt[6], val_pvt[7], val_pvt[8], val_pvt[9], val_pvt[10],
                 val_pvt[11], val_pvt[12], val_pvt[13], val_pvt[14], val_pvt[15], val_pvt[16], val_pvt[17], val_pvt[18],
                 val_pvt[19], val_pvt[20]])
            college_name.append(row[0][:20])

        else:
            print "Can't be possible! Table not in saved universities.", row[0]
            continue

    complete_aspects_score.append(sub_aspects_score)

correlation_coefficients = []
min_val = 5
max_val = 0

index = 0

while index < len(complete_aspects_score):

    sub_correlation_coefficients = []

    for item in complete_aspects_score:
        # if item == complete_aspects_score[index]:
        #     continue
        sub_correlation_coefficients.append(np.ma.corrcoef(complete_aspects_score[index], item)[1, 0])

    if min_val > min(sub_correlation_coefficients):
        min_val = min(sub_correlation_coefficients)

    if max_val < max(sub_correlation_coefficients):
        max_val = max(sub_correlation_coefficients)

    indices, L_sorted = zip(*sorted(enumerate(sub_correlation_coefficients), key=itemgetter(1)))

    correlation_coefficients.append(sub_correlation_coefficients)
    index += 1

# x_arr = np.ones(len(correlation_coefficients))
# y_arr = x_arr
# n=20
# index=1
#
# for i in range(0, len(correlation_coefficients), n):
# print correlation_coefficients

f, ax = plt.subplots(1, 1, figsize=(200, 200), dpi=96)

mapp = ax.imshow(correlation_coefficients, cmap='RdBu', vmin=min_val, vmax=max_val, interpolation='none', origin='lower',
                 extent=[0, len(college_name), 0, len(college_name)], aspect='auto')
plt.colorbar(mapp, format='%.3f', fraction=0.10, pad=0.09, ticks=np.linspace(min_val, max_val, 10), spacing='proportional')

ax.set_title('Interpolation_Graph_Correlation_Aspects_Individual_Colleges', y=1.09, style='italic', family='monospace',
             weight='bold', size='small')
ax.set_xlabel('College_Name', labelpad=-0.6, family='monospace', weight='bold', size='xx-small')
ax.set_ylabel('College_Name', labelpad=0, family='monospace', weight='bold', size='xx-small')
# ax.set_xticks(len(college_name))
# ax.set_yticks(np.arange(min_value - 0.5, max_value + 0.5,0.3))

# ax.set_ylim(bottom=min_value - 0.2, top=max_value + 0.3, emit=True)

# ax.xaxis.set_minor_locator(MaxNLocator(nbins=len(correlation_coefficients), prune='lower'))
# ax.yaxis.set_minor_locator(MaxNLocator(nbins=len(correlation_coefficients), prune=None))
ax.yaxis.set_major_locator(MaxNLocator(nbins=len(college_name), prune=None))
ax.xaxis.set_major_locator(MaxNLocator(nbins=len(college_name), prune=None))

ax.set_xticklabels(college_name, fontdict='monospace', zorder=1, color='blue', rotation='vertical',
                   va='top', ha='left', weight='normal', clip_on=False, axes=ax, size='xx-small')

# ax.set_xticklabels(college_name, fontdict='monospace', minor = True, zorder=1, color='blue', rotation='20',
#                    va='top', ha='center', weight='roman',clip_on=True, axes=ax)

ax.set_yticklabels(college_name, fontdict='monospace', zorder=1, color='blue', rotation='0',
                   va='bottom', ha='right', weight='normal', clip_on=False, axes=ax, size='xx-small')

ax.tick_params(axis='y', which='both', width=0, labelsize=3, right='off', labelright='off')
ax.tick_params(axis='x', which='both', width=0, labelsize=4, top='off', labeltop='off')

new_tick_locations = np.arange(0, len(college_name), 1)

ax1 = ax.twinx()
# f.canvas.draw()
# ax1.plot()

ax1.set_yticks(new_tick_locations)
# plt.yticks(new_tick_locations,linespacing=0.5)
# ax1.yaxis.set_major_locator(MaxNLocator(prune='lower'))
# print college_name, len(college_name)
ax1.set_yticklabels(college_name, fontdict='monospace', zorder=1, color='blue', rotation='0',
                    va='bottom', ha='left', weight='normal', clip_on=True, axes=ax1, size='xx-small')
ax1.tick_params(axis='y', which='both', width=0, labelsize=3, left='off', labelleft='off')

# ax1.autoscale(enable=True,axis='y',tight=True)
# ax1.autoscale_view(tight=True,scaley=True)

ax2 = ax.twiny()
# f.canvas.draw()
# ax2.plot()
# new_tick_locations1 = np.arange(0,len(college_name),1)
ax2.set_xticks(new_tick_locations)
# ax2.xaxis.set_major_locator(MaxNLocator(prune='lower'))
# demi_college_name = college_name
# # col_len = len(college_name)
# # print col_len
# demi_college_name = [item.encode('ascii', 'ignore') for item in demi_college_name]
# ax2.set_xticks(demi_college_name)
# ax1.set_yticks(demi_college_name)
# ax2.xaxis.set_major_locator(MaxNLocator(nbins=len(college_name), prune=None))


ax2.set_xticklabels(college_name, fontdict='monospace', zorder=1, color='blue', rotation='vertical',
                    va='bottom', ha='left', weight='normal', clip_on=True, axes=ax2, size='xx-small')

# ax.set_yticklabels(college_name, fontdict='monospace', minor = True, zorder=1, color='blue', rotation='20',
#                    va='top', ha='center', weight='roman', clip_on=True, axes=ax)

ax2.tick_params(axis='x', which='both', width=0, labelsize=4, bottom='off', labelbottom='off')
# ax.tick_params(axis='x', which='both', , labelbottom='on', width=0)
# ax.margins(x=0.015, tight=True, pad=2)

# ax.xaxis.set_minor_locator(MaxNLocator(nbins=len(college_name), prune='lower'))

# ax2.autoscale(enable=True,axis='x',tight=True)
# ax2.autoscale_view(tight=True,scalex=True)

# plt.subplots_adjust(bottom=0.1, top=0.905, right=0.87)
# ax.grid()
# ax.legend(loc=2, borderaxespad=0.1, fontsize='medium')
plt.show()
img_name = "graph_correlation_pearson_individual_colleges_aspects.svg"
# index +=1
# ax.view_init(elev='10', azim='20')
f.savefig(img_name, format='svg', dpi=300)
plt.close(f)
pp = PdfPages('graph_correlation_pearson_individual_colleges_aspects.pdf')
pp.savefig(f)
# pp.savefig(plot2)
# pp.savefig(plot3)
pp.close()

# count = 0
# # print matplotlib.get_backend()
#
# while count < 5:
#     count +=1
# compare_data = random.sample(range(0,len(complete_aspects_score)),2)
# print compare_data
# print "Correlation between ",college_name[compare_data[0]], " & ", college_name[compare_data[1]] , " is : ", np.corrcoef(complete_aspects_score[compare_data[0]],complete_aspects_score[compare_data[1]])[1,0]
