import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import matplotlib.colors as colors
# import sys
import numpy as np
import MySQLdb

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8",
                        use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)

cnxn = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="college_info", charset="utf8", use_unicode=True)
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
cursor1.execute(query, [0, 200])
sent_val = cursor1.fetchall()
# all_college_sentiments.append(sent_val)
# prev += 30

aspects_value = []

query00 = "show columns from pub_college_aspects_ranking where Type = 'float(3,2)'"
cursor.execute(query00)
number_aspects = cursor.rowcount

for name in cursor:
    aspects_value.append(name[0])

complete_aspects_score = []
# sentiment_score = []
overall_college_score = []
college_name = []

for row in sent_val:

    sub_aspects_score = []

    query11 = """select overall_grade from pub_college_overall_rank where replace(replace(replace(replace(replace
                 (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
    cursor.execute(query11)
    val_pub1 = cursor.fetchone()

    if val_pub1 is not None:

        query11 = """select * from pub_college_aspects_ranking where replace(replace(replace(replace(replace
                 (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query11)
        val_pub = cursor.fetchone()

        sub_aspects_score.extend(
            [val_pub[2], val_pub[3], val_pub[4], val_pub[5], val_pub[6], val_pub[7], val_pub[8], val_pub[9], val_pub[10], val_pub[11], val_pub[12],
             val_pub[13], val_pub[14], val_pub[15], val_pub[16], val_pub[17], val_pub[18], val_pub[19], val_pub[20]])

        # college_name.append(row[0][:20].strip())
        # sentiment_score.append(row[1])
        overall_college_score.append(val_pub1[0])
        # college_name.append(row[0])

    else:
        query12 = """select overall_grade from pvt_college_overall_rank where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query12)
        val_pvt1 = cursor.fetchone()

        if val_pvt1 is not None:

            query12 = """select * from pvt_college_aspects_ranking where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
            cursor.execute(query12)
            val_pvt = cursor.fetchone()

            sub_aspects_score.extend(
                [val_pvt[2], val_pvt[3], val_pvt[4], val_pvt[5], val_pvt[6], val_pvt[7], val_pvt[8], val_pvt[9], val_pvt[10], val_pvt[11], val_pvt[12],
                 val_pvt[13], val_pvt[14], val_pvt[15], val_pvt[16], val_pvt[17], val_pvt[18], val_pvt[19], val_pvt[20]])

            # college_name.append(row[0][:20].strip())
            # sentiment_score.append(row[1])
            overall_college_score.append(val_pvt1[0])
            # college_name.append(row[0])

        else:
            print "Can't be possible! Table not in saved universities.", row[0]
            continue

    complete_aspects_score.append(sub_aspects_score)

    # ind = np.arange(len(sentiment_score))
    # width = 0.6

# min_value = 5
# max_value = 0

index = 0
aspects_dd_list = []

while index < number_aspects:

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
    # # check_min += min(valid_show_aspects)
    # if check_min < min_value:
    #     min_value = check_min
    # if check_max > max_value:
    #     max_value = check_max
    # max_values.append(check_max)
    # min_values.append(check_min)

    valid_show_aspects = np.array(valid_show_aspects)
    # color_arr.append()
    aspects_dd_list.append(valid_show_aspects)
    # ax.bar(college_arr+(index*0.05),valid_show_aspects,width=0.05,color=color_arr[index],align='center',label=aspects_value[index])
    index += 1

f, ax = plt.subplots(1, 1, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)
# ax = fig.add_subplot(111)

ax.set_title('Colleges_Correlation_Pearson_between_Aspects_&_Overall_Score', y=1.02, style='italic', family='monospace', weight='bold')
ax.set_xlabel('College_Score', labelpad=25, family='monospace', weight='bold')
ax.set_ylabel('Aspects_Score', labelpad=25, family='monospace', weight='bold')
# plt.xticks(np.arange((min(overall_college_score) - 2), (max(overall_college_score) + 2), 0.2))
# plt.yticks(np.arange((min(sentiment_score) - 0.2), (max(sentiment_score) + 0.2), 0.03))
# print len(set(overall_college_score))
# ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
# overall_college_score = list(set(overall_college_score))
# overall_college_score = ["%.2f" % item for item in overall_college_score]

ax.yaxis.set_major_locator(MaxNLocator(nbins=20, prune=None))
# ax.set_ylim(bottom=min_value - 0.2, top=max_value + 0.2, emit=True)
#
# ax.set_xticklabels(overall_college_score, fontdict='monospace', zorder=1, color='black', rotation='horizontal',
#                    va='top', ha='center', weight='roman', clip_on=False, axes=ax, x =0.015)

ax.xaxis.set_major_locator(MaxNLocator(nbins=20, prune=None))

ax.tick_params(axis='both', width=1.5, direction='in')
# ax.margins(0.015, tight=True, pad=2)

overall_college_score = np.array(overall_college_score)

color_arr = ['red', 'green', 'blue', 'yellow', 'black', 'grey', 'magenta', 'brown', 'turquoise', 'pink', 'violet', '#FE2E2E', 'cyan', '#996600', '#190707',
             '#5858FA', '#0B0B61']
# color_norm = colors.Normalize(vmin=0, vmax=len(aspects_dd_list) - 1)
# # # print color_norm
# scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmx.nipy_spectral)
# # # print scalar_map
# color_arr = scalar_map.to_rgba([n for n in range(len(aspects_dd_list))])

# color_cycle = ax.set_color_cycle(color_arr)

for key, item in enumerate(aspects_dd_list):

    # print np.ma.corrcoef(overall_college_score, item)
    compare_overall_college_score = overall_college_score

    # print len(item), " : ", len(overall_college_score)

    index = []
    for key1, val in enumerate(item):
        if val == 0.:
            index.append(key1)

    item = np.delete(item, index)
    compare_overall_college_score = np.delete(compare_overall_college_score, index)
    # print len(item), " : ", len(compare_overall_college_score)

    pearR = np.ma.corrcoef(compare_overall_college_score, item)[1, 0]

    A = np.vstack([compare_overall_college_score, np.ones(len(compare_overall_college_score))]).T
    m, c = np.linalg.lstsq(A, np.array(item))[0]

    # ax.scatter(overall_college_score, sentiment_score,label='Data Red',color='r')
    ax.plot(compare_overall_college_score, compare_overall_college_score * m + c, color=color_arr,
            label="" + aspects_value[key].capitalize() + " - r = %6.2e" % pearR, rasterized=True, antialiased=False, linestyle='-', linewidth=0.7)

# plt.scatter(overall_college_score, sentiment_score, color='r')

# ax.autoscale(enable=True, axis='both', tight=True)
# ax.autoscale_view(scalex=True, scaley=True, tight=True)

# width = ind * 3
# height = max(average_val_data) + 0.2

# plt.figure(figsize=('%s', '%s') % (width, height))
# check_overlap = [[-10, -10]]
#
# for label, x, y in zip(college_name, overall_college_score, sentiment_score):
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

# for i, rect in enumerate(rects):
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width(), height + 0.18, '(%s,%s)' % (sentiment_score[i], overall_college_score[i]),
#              ha='center', va='top', rotation='vertical')

# plt.tight_layout()
plt.grid()
ax.legend(loc='lower right', bbox_to_anchor=(0.999, 0.001), borderpad=0.01, fontsize='small', frameon=True, labelspacing=0.2, framealpha=0.3,
          fancybox=True)
# plt.setp(legend.get_label(),fontweight='xx-small')
plt.show()
img_name = "graph_correlation_pearson_overall_aspects.svg"
f.savefig(img_name, dpi=600)
plt.close(f)
