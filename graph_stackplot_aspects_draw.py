import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import matplotlib.font_manager as fm
import matplotlib.cm as cmx
import matplotlib.colors as colors
# import matplotlib.finance as fin
# import sys
import numpy as np
from itertools import cycle
# import csv

import MySQLdb

cnxn1 = MySQLdb.connect(host="localhost", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8",
                        use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)

cnxn = MySQLdb.connect(host="localhost", user="root", passwd="harshit@123", db="college_info", charset="utf8", use_unicode=True)
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
query = "select college_name,sentiment_average from college_sentiment_average order by id limit %s,%s"
cursor1.execute(query, [0, 15])
sent_val = cursor1.fetchall()
# all_college_sentiments.append(sent_val)
# prev += 30

# sentiment_score = []
complete_aspects_score = []
college_name = []
aspects_value = []

query00 = "show columns from pub_college_aspects_ranking where Type = 'float(3,2)'"
cursor.execute(query00)
number_aspects = cursor.rowcount

for name in cursor:
    aspects_value.append(name[0])

for row in sent_val:

    sub_aspects_score = []

    query11 = """select * from pub_college_aspects_ranking where replace(replace(replace(replace(replace
                 (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
    cursor.execute(query11)
    val_pub = cursor.fetchone()

    if val_pub is not None:
        # sentiment_score.append(row[1])
        # aspects_value.append(val_pub[0])
        sub_aspects_score.extend(
            [val_pub[2], val_pub[3], val_pub[4], val_pub[5], val_pub[6], val_pub[7], val_pub[8], val_pub[9], val_pub[10],
             val_pub[11], val_pub[12], val_pub[13], val_pub[14], val_pub[15], val_pub[16], val_pub[17], val_pub[18], val_pub[19],
             val_pub[20]])
        college_name.append(row[0][:20].strip())

    else:
        query12 = """select * from pvt_college_aspects_ranking where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query12)
        val_pvt = cursor.fetchone()

        if val_pvt is not None:
            # sentiment_score.append(row[1])
            sub_aspects_score.extend(
                [val_pvt[2], val_pvt[3], val_pvt[4], val_pvt[5], val_pvt[6], val_pvt[7], val_pvt[8], val_pvt[9], val_pvt[10],
                 val_pvt[11], val_pvt[12], val_pvt[13], val_pvt[14], val_pvt[15], val_pvt[16], val_pvt[17], val_pvt[18],
                 val_pvt[19], val_pvt[20]])
            college_name.append(row[0][:20].strip())

        else:
            print "Can't be possible! Table not in saved universities.", row[0]
            continue

    complete_aspects_score.append(sub_aspects_score)

    # ind = np.arange(len(sentiment_score))
    # width = 0.6

# color_arr = []
f, ax = plt.subplots(1, 1, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)

# min_value = 5
# max_value = 0
min_values = []
max_values = []

index = 0
aspects_dd_list = []

# college_arr = np.arange(0,len(college_name),1)
# color_arr = cmx.rainbow(np.linspace(0,1,number_aspects))
while index < number_aspects - 5:

    # fig = plt.figure(figsize=(1600 / 95.8, 900 / 95.8), dpi=96)
    # ax = fig.add_subplot(111)

    # valid_show_aspects = []
    valid_show_aspects = []
    # valid_sentiment_aspects = []

    # writer = csv.writer(open('data_correlation_pearson_'+aspects_value[index]+'_sentiment.csv','w'))
    # writer.writerow(['id','college_name',aspects_value[index]+'_score','sentiment_score'])

    # col_count = 0
    # print "graph for : ", aspects_value[index], "\n"
    for item in complete_aspects_score:
        # college_aspects_score.append(valid_show_aspects)
        # writer.writerow([col_count+1,college_name[col_count],item[index],sentiment_score[col_count]])
        if item[index] >= 0:
            valid_show_aspects.append(item[index])
            # valid_sentiment_aspects.append(sentiment_score[col_count])
            # print "with aspect value : ", item[index], "equivalent sentiment of : ", sentiment_score[col_count]

            # col_count += 1

    # check_min = min(valid_show_aspects)
    # check_max = max(valid_show_aspects)
    # check_min += min(valid_show_aspects)
    # if check_min < min_value:
    #     min_value = check_min
    # if check_max > max_value:
    #     max_value = check_max
    max_values.append(max(valid_show_aspects))
    min_values.append(min(valid_show_aspects))

    valid_show_aspects = np.array(valid_show_aspects)
    # color_arr.append()
    aspects_dd_list.append(valid_show_aspects)
    # ax.bar(college_arr+(index*0.05),valid_show_aspects,width=0.05,color=color_arr[index],align='center',label=aspects_value[index])
    index += 1

    # data_add = min(valid_sentiment_aspects)

    # valid_sentiment_aspects = [value+abs(data_add) for value in valid_sentiment_aspects]



    # pearR = np.corrcoef(valid_show_aspects, np.array(valid_sentiment_aspects))[1,0]
    #
    # A = np.vstack([valid_show_aspects,np.ones(len(valid_show_aspects))]).T
    # m,c = np.linalg.lstsq(A,np.array(valid_sentiment_aspects))[0]
# ax.scatter(valid_show_aspects, valid_sentiment_aspects,label='Data Red',color='r')

# plt.plot(valid_show_aspects,valid_show_aspects*m+c,color='r',label="Fit %6s, r = %6.2e"%('RED',pearR))
# ax.set_yscale('log')
# ax.set_yscale('log', basey=1.1, posxy=)
# ax.set_xscale('log', basex=1.1)

# plt.scatter(valid_show_aspects, sentiment_score, color='r')

ax.set_title('Colleges_Aspects_StackPlot_Chart', y=1.02, style='italic', family='monospace', weight='bold')
ax.set_xlabel('College_Name', labelpad=-10, family='monospace', weight='bold')
ax.set_ylabel('Aspects_Score', labelpad=25, family='monospace', weight='bold')
# ax.set_xticks(len(college_name))
# ax.set_yticks(np.arange(min_value - 0.5, max_value + 0.5,0.3))

# ax.set_ylim(top=max_value + 0.3, emit=True)
ax.yaxis.set_major_locator(MaxNLocator(nbins=50, prune=None))

ax.xaxis.set_major_locator(MaxNLocator(nbins=len(college_name), prune=None))
ax.set_xticklabels(college_name, fontdict='monospace', zorder=1, color='blue', rotation='20',
                   va='top', ha='center', weight='roman', clip_on=False, axes=ax)

ax.tick_params(axis='both', width=1.5, direction='in')
# ax.margins(x=0, tight=True, pad=2)

# ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))

# print min_value, "", max_value

# ax.set_yscal  e('log',basey=1.1)
# width = ind * 3
# height = max(average_val_data) + 0.2
# print aspects_dd_list,'\n'
# arr_dd_stackplot =[[item] for item in aspects_dd_list]
# print arr_dd_stackplot
# print arr_dd_stackplot
# length = len(college_name)
# length_aspects_list = len(aspects_dd_list)
# print aspects_dd_list
color_norm = colors.Normalize(vmin=0, vmax=len(aspects_dd_list) - 1)
# print color_norm
scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
# print scalar_map
color_arr = scalar_map.to_rgba([n for n in range(len(aspects_dd_list))])
# color_arr = ['red', 'green', 'blue', 'yellow', 'black', 'grey', 'magenta']
#
# color_cycle = ax.set_color_cycle(color_arr)
# print color_cycle
# print len(color_arr)
# color_arr = np.array(color_arr)
# print aspects_dd_list[0]
# print "no of colleges : ",length, " no of aspects : ", length_aspects_list
# college_name_dd_array = [np.arange(1, len(college_name) + 1, 1) for _ in range(len(aspects_dd_list))]

# if all(len(list) == len(item) for list in aspects_dd_list for item in college_name_dd_array):
#     print "Success"

# # print college_name_dd_array
# # print college_name_dd_array
# # print length
# markers = np.arange(6, 2*(length_aspects_list+1), 2)
# labels = (aspects_value[i] for i in range(number_aspects))
# print labels
# print len(college_name), " :  ", len(aspects_dd_list[0])
# print aspects_dd_list
rects = ax.stackplot(np.arange(1, len(college_name) + 1, 1), aspects_dd_list, baseline='zero', colors=color_arr)

# side_arr_cycle = cycle([3, 4, 5, 6])
# linestyle_arr = cycle(['-'])
# angle_val = np.linspace(0, 90, len(aspects_dd_list))
# transparency = iter(np.linspace(1.0,0.17,len(aspects_dd_list)))
#
# # ax.plot(college_name_dd_array, aspects_dd_list, color=color_cycle, markersize=6, antialiased=True,
# #                         rasterized=True, label=aspects_value[0], linewidth=1, alpha=0.7, linestyle=linestyle_arr.next(), marker=(side_arr_cycle.next(), 3 % 4, angle_val[0]))
#
# for i in range(len(aspects_dd_list)):
#
#     axes_plot = ax.plot(college_name_dd_array[i], aspects_dd_list[i], color=color_cycle, markersize=6 + i, antialiased=True,
#                         rasterized=True, label=aspects_value[i], linewidth=1 + (i / 2.5), alpha=transparency.next(),
#                         linestyle=linestyle_arr.next(), marker=(side_arr_cycle.next(), i % 4, angle_val[i]))
#
# for k,item in enumerate(axes_plot):
#     val_store_arr = item.get_ydata()
#     for key,val in enumerate(val_store_arr):
#         for key1,val1 in range(val_store_arr[(key+1):]):
#             for j,q in zip(range(len(val)),range(len(val1))):
#                 if val[j] == val1[q]:
#                     item.set_markersize(40)


# ax.legend(iter(axes_plot), labels, loc=0)
# ax.legend(loc=0)
# size = fm.FontProperties()
# size.set_size('small')
# print aspects_value
# last_item = 0
multiply_factor = np.linspace(0.1,0.25,len(aspects_dd_list))

for i, rect in enumerate(rects):
    # print array_height
    # print min(array_height), " : ", max(array_height)
    # print rect.get_offsets(), "::", rect.get_offset_position()
    # fig_object = rect.get_figure()
    # height = fig_object.get_figheight()
    # height = 0
    #
    # for j in range(max_values[:i]):
    #     height += max_values[j]
    height = 0
    max_loc = aspects_dd_list[i].argmax()

    for k,val in enumerate(aspects_dd_list[:i]):
        # print val[max_loc]
        height += val[max_loc]
    # print aspects_dd_list
    height += aspects_dd_list[i][max_loc] * multiply_factor[i] #- (max_values[i] - min_values[i])

    plt.text(max_loc+1, height, '%s' % aspects_value[i], ha='center', va='bottom', rotation='5', weight='roman', size='medium', family='monospace', rasterized=True, style='italic')
    # ax.annotate(aspects_value[i][:20], xy=(aspects_dd_list[i].argmax(), height), xytext=(0, 0), textcoords='offset points', fontsize='small', ha='right',
    #             va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')

# ax.autoscale(enable=True, axis='both', tight=True)
# ax.autoscale_view(scalex=True, scaley=True, tight=True)

# plt.figure(figsize=('%s', '%s') % (width, height))
# check_overlap = [[-10, -10]],'o'
#
# for label, x, y in zip(college_name, college_aspects_score, sentiment_score):
#     count = 0
#     for i, content in enumerate(check_overlap):
#         print content
#         print "y : ", y, "x : ", x
#         if (abs(y - content[1]) > 0.1) and (abs(x - content[0]) > 0.2):
#             count += 1
#         else:
#             continue
#     if count >= 6:
#         print count, " : ", len(check_overlap)
#         print "-"
#         check_overlap.append([x, y])
#         ax.annotate(label[:20], xy=(x, y), xytext=(-20, 20), textcoords='offset points', fontsize='small', ha='right',
#                     va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6), arrowprops=dict(
#                     arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')
#     else:
#         print count, " : ", len(check_overlap)
#         print "+"
#         check_overlap.append([x, y])
#         ax.annotate(label[:20], xy=(x, y), xytext=(8, 20), textcoords='offset points', fontsize='small', ha='left',
#                     va='top', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6), arrowprops=dict(
#                     arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')

# plt.tight_layout()
# plt.grid()
# plt.legend(loc='best')

# fig = plt.figure(figsize=(1600 / 95.8, 900 / 95.8), dpi=96)
# plt.subplot(122)
# ax = plt.gca()
#
# plt.title('Colleges_CandleStick_Chart_of_'+aspects_value[index]+'_with_College_tweets_Sentiment', y=1.02, style='italic', family='monospace', weight='bold')
# plt.xlabel(''+aspects_value[index]+'_score', labelpad=25, family='monospace', weight='bold')
# plt.ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')
# # plt.xticks(np.arange((min(college_aspects_score) - 2), (max(college_aspects_score) + 2), 0.2))
# # plt.yticks(np.arange((min(sentiment_score) - 0.2), (max(sentiment_score) + 0.2), 0.03))
#
# # ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=25, prune=None))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15, prune=None))
# ax.set_ylim(bottom=min(valid_sentiment_aspects) - 0.02, top=max(valid_sentiment_aspects) + 0.1, emit=True)
#
# plt.margins(0.009)
# aspect_index_candle_chart = set(valid_show_aspects)
# aspect_index_candle_chart = list(aspect_index_candle_chart)
# # print aspect_index_candle_chart
# # sentiment_index_candle_chart = []
# quotes = []
# colors = []
#
# for data in aspect_index_candle_chart:
#     # print data
#     sentiment_sub_index_candle_chart = []
#     added_value = []
#     sentiment_sub_index_candle_chart.extend([data])
#     for key,val in enumerate(valid_show_aspects):
#         if val == data:
#             sentiment_sub_index_candle_chart.extend([valid_sentiment_aspects[key]])
#
#     max_val = max(sentiment_sub_index_candle_chart[1:])
#     min_val = min(sentiment_sub_index_candle_chart[1:])
#     avg_val = sum(sentiment_sub_index_candle_chart[1:]) / len(sentiment_sub_index_candle_chart[1:])
#
#     added_value.extend([sentiment_sub_index_candle_chart[0],min_val+((avg_val-min_val)/2),max_val-((max_val-avg_val)/2),max_val,min_val])
#     quotes.append(added_value)
#
#     # if avg_val > 1.5:
#     #     colors.append('b')
#     # else:
#     #     colors.append('r')
#     # sentiment_index_candle_chart.append(sentiment_sub_index_candle_chart)
#
# fin.candlestick_ochl(ax,quotes,colorup='b',colordown='r',alpha=0.7)

# plt.tight_layout()
# plt.legend(loc='best')
plt.grid()
plt.show()
img_name = "graph_real_stackplot_aspects.png"
f.savefig(img_name, dpi=95.9)
