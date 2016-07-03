import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as plticker
from matplotlib.backends.backend_pdf import PdfPages
# import sys
import numpy as np
# import itertools
import MySQLdb
import datetime

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
query = """select college_name,sentiment_average,no_rows from college_sentiment_average order by id limit %s,%s"""
cursor1.execute(query, [0, 200])
sent_val = cursor1.fetchall()
no_col = cursor1.rowcount
# all_college_sentiments.append(sent_val)
# prev += 30

sentiment_score = []
overall_college_score = []
college_name = []
complete_aspects_score = []

for row in sent_val:

    sub_aspects_score = []

    query11 = """select * from pub_college_aspects_ranking where replace(replace(replace(replace(replace
                 (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
    cursor.execute(query11)
    val_pub = cursor.fetchone()

    if val_pub is not None:
        # sentiment_score.append(row[1])
        # aspects_value.append(val_pub[0])
        # tweets_analyzed.append(row[2])
        sub_aspects_score.extend(
            [val_pub[2], val_pub[3], val_pub[4], val_pub[5], val_pub[6], val_pub[7], val_pub[8], val_pub[9], val_pub[10],
             val_pub[11], val_pub[12], val_pub[13], val_pub[14], val_pub[15], val_pub[16], val_pub[17], val_pub[18], val_pub[19],
             val_pub[20]])
        # college_name.append(row[0][:20])

    else:
        query12 = """select * from pvt_college_aspects_ranking where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query12)
        val_pvt = cursor.fetchone()

        if val_pvt is not None:
            # sentiment_score.append(row[1])
            # tweets_analyzed.append(row[2])
            sub_aspects_score.extend(
                [val_pvt[2], val_pvt[3], val_pvt[4], val_pvt[5], val_pvt[6], val_pvt[7], val_pvt[8], val_pvt[9], val_pvt[10],
                 val_pvt[11], val_pvt[12], val_pvt[13], val_pvt[14], val_pvt[15], val_pvt[16], val_pvt[17], val_pvt[18],
                 val_pvt[19], val_pvt[20]])
            # college_name.append(row[0][:20])

        else:
            print "Can't be possible! Table not in saved universities.", row[0]
            continue

    complete_aspects_score.append(sub_aspects_score)

# query00 = """select college_name, sum(no_rows),avg(sentiment_average) from college_sentiment_average_per_day where sentiment_date between '2015-08-04' and date_add('2015-08-04', interval 30 day) group by college_name order by college_name"""
# cursor1.execute(query00)
# date_val_1 = cursor1.fetchall()
#
# coll_1_month_name = []
# coll_1_month_sen = []
# coll_1_month_rows = []
#
# for item in date_val_1:
#     coll_1_month_name.append(item[0])
#     coll_1_month_sen.append(item[2])
#     coll_1_month_rows.append(item[1])
#
# query01 = """select college_name, sum(no_rows),avg(sentiment_average) from college_sentiment_average_per_day where sentiment_date between '2015-09-04' and date_add('2015-09-04', interval 30 day) group by college_name order by college_name"""
# cursor1.execute(query01)
# date_val_2 = cursor1.fetchall()
#
#
# select Round((UNIX_TIMESTAMP(date_format(sentiment_date,"%Y-%m-%d"))) / (30*24*60*60)) as interval from college_sentiment_average_per_day group by college_name,interval
#
# coll_2_month_name = []
# coll_2_month_sen = []
# coll_2_month_rows = []
#
# select
#
# for item in date_val_2:
#     coll_2_month_name.append(item[0])
#     coll_2_month_sen.append(item[2])
#     coll_2_month_rows.append(item[1])
#
# query02 = """select college_name, sum(no_rows),avg(sentiment_average) from college_sentiment_average_per_day where sentiment_date between '2015-10-04' and date_add('2015-10-04', interval 30 day) group by college_name order by college_name"""
# cursor1.execute(query02)
# date_val_3 = cursor1.fetchall()
#
# coll_3_month_name = []
# coll_3_month_sen = []
# coll_3_month_rows = []
#
# for item in date_val_3:
#     coll_3_month_name.append(item[0])
#     coll_3_month_sen.append(item[2])
#     coll_3_month_rows.append(item[1])


query00 = """select sentiment_date, sum(no_rows), avg(sentiment_average), college_name, count(*) from college_sentiment_average_per_day group by round(UNIX_TIMESTAMP(date_format(sentiment_date,"%Y-%m-%d")) DIV (30*24*60*60)),college_name order by concat(sentiment_date, college_name)"""
cursor1.execute(query00)
no_of_rows = cursor1.rowcount
# query11 = """select overall_grade from pub_college_overall_rank where replace(replace(replace(replace(replace
#              (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
# cursor.execute(query11)
# val_pub = cursor.fetchone()
#
# if val_pub is not None:
#     sentiment_score.append(row[1])
#     overall_college_score.append(val_pub[0])
#     college_name.append(row[0])
#
# else:
#     query12 = """select overall_grade from pvt_college_overall_rank where replace(replace(replace(replace(replace
#                  (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
#     cursor.execute(query12)
#     val_pvt = cursor.fetchone()
#
#     if val_pvt is not None:
#         sentiment_score.append(row[1])
#         overall_college_score.append(val_pvt[0])
#         college_name.append(row[0])
#
#     else:
#         print "Can't be possible! Table not in saved universities.", row[0]
#         continue

# ind = np.arange(len(sentiment_score))
# width = 0.6

# overall_college_score = np.array(overall_college_score)
#
# pearR = np.corrcoef(overall_college_score, sentiment_score)[1,0]
#
# A = np.vstack([overall_college_score,np.ones(len(overall_college_score))]).T
# m,c = np.linalg.lstsq(A,np.array(sentiment_score))[0]
#

# ax.plot(overall_college_score,overall_college_score*m+c,color='r',label="Fit %6s, r = %6.2e"%('RED',pearR),rasterized=True,antialiased=True,linestyle='-',linewidth=1)

# plt.scatter(overall_college_score, sentiment_score, color='r')

for item in complete_aspects_score:
    for val in item:
        if val in (0, 0., 0.0, 0.00, 0.000):
            item.remove(val)
    item = np.array(item)
    mean = round(float(np.mean(item)), 2)
    overall_college_score.append(mean)

# prev = 0
last = 0
# print overall_college_score

while last < no_of_rows:

    query01 = query00 + """ limit {},{}""".format(last, no_col)
    cursor1.execute(query01)

    col_name = []
    col_sen = []
    # tweets_analyzed = []
    col_rows = []
    start_date = datetime.datetime.now().date()
    days_data = 0
    overall_college_score_sub = []

    for i, data in enumerate(cursor1):
        if data[2] is not None and data[1] >= 50:
            days_data = int(data[4])
            start_date = data[0]
            col_name.append(data[3])
            col_sen.append(data[2])
            col_rows.append(data[1])
            overall_college_score_sub.append(overall_college_score[i])

    last += no_col

    f, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)

    x_Formatter = plticker.FormatStrFormatter('%.2f')
    y_Formatter = plticker.FormatStrFormatter('%.3f')

    for axes1 in ax:
        for axes in axes1:
            axes.xaxis.set_major_formatter(x_Formatter)
            axes.xaxis.set_minor_formatter(x_Formatter)
            axes.yaxis.set_major_formatter(y_Formatter)
            axes.yaxis.set_minor_formatter(y_Formatter)

    ax[1, 0].spines['right'].set_visible(False)
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 1].spines['left'].set_visible(False)
    ax[1, 1].spines['top'].set_visible(False)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].spines['bottom'].set_visible(False)
    ax[0, 1].spines['left'].set_visible(False)
    ax[0, 1].spines['bottom'].set_visible(False)
    # axs[1].spines['bottom'].set_visible(False)
    # ax[0][0],ax[0][1] = ax
    # axes = ax
    # ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0, hspace=0)

    time_upto = start_date + datetime.timedelta(days=days_data)
    f.suptitle(
        'Colleges_Correlation_between_Sentiment_&_Overall_Score_for_time_' + str(start_date) + '_to_' + str(time_upto) + '_for colleges > 50 tweets',
        style='italic', family='monospace', weight='bold', size='large', ha='center', va='top')

    f.text(0.5, 0.01, 'College_Score', ha='center', family='monospace', weight='bold')
    f.text(0.01, 0.5, 'Avg_Sentiment_Value', va='center', rotation='vertical', family='monospace', weight='bold')

    # plt.xticks(np.arange((min(overall_college_score) - 2), (max(overall_college_score) + 2), 0.2))
    # plt.yticks(np.arange((min(sentiment_score) - 0.2), (max(sentiment_score) + 0.2), 0.03))
    # print len(set(overall_college_score))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins = ,prune='lower'))
    # overall_college_score_sub = overall_college_score
    # overall_college_score_sub = list(set(overall_college_score_sub))
    # overall_college_score_sub = ["%.2f" % item for item in overall_college_score_sub]

    # avg_sentiment_score = reduce(lambda x, y: x + y, sentiment_score) / len(sentiment_score)
    # avg_college_score = reduce(lambda x, y: x + y, overall_college_score) / len(overall_college_score)
    # median_sen = np.median(np.array(list(set(sentiment_score))))
    # print median_sen

    median_col = np.mean(np.array(list(set(overall_college_score_sub))))
    # print len(col_sen)
    # mid_y_val = min(col_sen) + (abs(min(col_sen)) + max(col_sen))/2
    mid_y_val = 0.0
    # mid_x_val = - 0.5 + median_col

    # print str(loc_sen), "", min(col_sen), "", str(mid_y_val)

    # col_arr = []
    # labels= ""
    grid_1_sen = []
    grid_2_sen = []
    grid_3_sen = []
    grid_4_sen = []
    grid_1_col = []
    grid_2_col = []
    grid_3_col = []
    grid_4_col = []
    college_1_grid = []
    college_2_grid = []
    college_3_grid = []
    college_4_grid = []
    college_rows_1_grid = []
    college_rows_2_grid = []
    college_rows_3_grid = []
    college_rows_4_grid = []
    count_grid_1_college, count_grid_2_college, count_grid_3_college, count_grid_4_college = 0, 0, 0, 0

    i = 0
    for sen, col in zip(col_sen, overall_college_score_sub):
        if sen <= mid_y_val and col <= median_col:
            # col_arr.append('red')
            grid_1_sen.append(sen)
            grid_1_col.append(col)
            count_grid_1_college += 1
            college_1_grid.append(col_name[i])
            college_rows_1_grid.append(col_rows[i])
            # labels.append('Expected Values')
        elif sen <= mid_y_val and col > median_col:
            # col_arr.append('violet')
            grid_2_sen.append(sen)
            grid_2_col.append(col)
            count_grid_2_college += 1
            college_2_grid.append(col_name[i])
            college_rows_2_grid.append(col_rows[i])
            # labels.append('UNEXPECTED VALUES')
        elif sen > mid_y_val and col > median_col:
            # col_arr.append('green')
            grid_3_sen.append(sen)
            grid_3_col.append(col)
            count_grid_3_college += 1
            college_3_grid.append(col_name[i])
            college_rows_3_grid.append(col_rows[i])
            # labels.append('Expected Values')
        elif sen > mid_y_val and col <= median_col:
            # col_arr.append('violet')
            grid_4_sen.append(sen)
            grid_4_col.append(col)
            count_grid_4_college += 1
            college_4_grid.append(col_name[i])
            college_rows_4_grid.append(col_rows[i])
            # labels.append('UNEXPECTED VALUES')
        i += 1

    loc_col = plticker.MultipleLocator(base=(median_col + 0.02))
    # val_tick = (abs(min(col_sen)) + max(col_sen))/2
    # locs = dict(vmin=val_tick,vmax=val_tick)
    # val_tick = mid_y_val
    loc_sen = plticker.FixedLocator(np.linspace((mid_y_val + 0.02), (mid_y_val + 0.02), 1))
    # print (abs(min(col_sen)) + max(col_sen))/2

    # ax[0][0].xaxis.set_minor_locator(loc_col)
    # ax[0][0].yaxis.set_minor_locator(loc_sen)
    # ax[0][0].set_xlabel('college_score', labelpad=25, family='monospace', weight='bold')
    # ax[0][0].set_ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')
    # ax[0][0].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen)-0.1,max(grid_1_sen+grid_2_sen)+0.1,20),minor=True)
    # print grid_4_sen
    if grid_4_sen:
        ax[0][0].set_yticks(np.linspace(min(grid_3_sen + grid_4_sen), max(grid_3_sen + grid_4_sen), 10))
        ax[0][0].set_ylim(mid_y_val - 0.02, max(grid_3_sen + grid_4_sen))

        ax[0][0].xaxis.tick_top()
        ax[0][0].xaxis.set_label_position("top")
        # ax.set_yticklabels(col_sen,size='small')

        # ax[0][0].set_yticklabels([''+str((abs(min(col_sen)) + max(col_sen))/2)],size='small',minor=True,visible=False)
        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[0][0].set_xticks(np.linspace(min(grid_1_col + grid_4_col), max(grid_1_col + grid_4_col), 5))
        ax[0][0].set_xlim(min(grid_1_col + grid_4_col), median_col + 0.02)

        ax[0][0].xaxis.set_minor_locator(loc_col)
        # ax[0][0].xaxis.set_minor_formatter(plticker.FormatStrFormatter('%.2f'))
        # ax[1][0].set_xticks([''+str(median_col)],minor=True)
        rep_x = [('' + str(round(median_col, 3)))]
        ax[0][0].set_xticklabels(np.repeat(rep_x, 2), weight='bold', size='large', minor=True)
        # ax[0][0].set_xticklabels(list(set(overall_college_score)),size='small')

        ax[0][0].scatter(grid_4_col, grid_4_sen, color='black',
                         label="Higher Sentiment, Lower Overall Score \n No. of Colleges : " + str(count_grid_4_college) + ", Avg. Sen : " + str(
                             round(np.mean(np.array(grid_4_sen)), 2)))

        ax[0][0].tick_params(axis='both', which='major', width=1.5, direction='in', right='off', bottom='off')
        ax[0][0].tick_params(axis='x', which='minor', width=1, direction='in', pad=20, right='off')

        ax[0][0].grid(which='major', axis='both', linewidth=1, linestyle='-', color='grey', alpha=0.7)
        ax[0][0].grid(which='minor', axis='x', linewidth=4, linestyle='-', color='b')

        for j, item in enumerate(grid_4_sen):
            if grid_4_col[j] <= ((median_col - min(grid_1_col + grid_4_col)) / 1.3) + min(grid_1_col + grid_4_col) and grid_4_sen[j] >= (
                0 + max(grid_3_sen + grid_4_sen)) / 1.3 or grid_4_col[j] <= ((median_col - min(grid_1_col + grid_4_col)) / 1.3) + min(
                            grid_1_col + grid_4_col) and grid_4_sen[j] <= (0 + max(grid_3_sen + grid_4_sen)) / 1.3:
                ax[0][0].annotate(college_4_grid[j] + " - " + str(college_rows_4_grid[j]), xy=(grid_4_col[j], grid_4_sen[j]), xytext=(1, 20),
                                  textcoords='offset points', weight='normal', fontsize='small', ha='center', va='center',
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')

        ax[0][0].legend(loc='upper right', fontsize='small', framealpha=0.3, borderpad=0.1)

    # ax[0][1].set_xlabel('college_score', labelpad=25, family='monospace', weight='bold')
    # ax[0][1].set_ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')

    if grid_3_sen:
        # ax[0][1].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen)-0.1,max(grid_1_sen+grid_2_sen)+0.1,20),minor=True)
        ax[0][1].set_yticks(np.linspace(min(grid_3_sen + grid_4_sen), max(grid_3_sen + grid_4_sen), 10))
        ax[0][1].set_ylim(mid_y_val - 0.02, max(grid_3_sen + grid_4_sen))

        ax[0][1].yaxis.tick_right()
        ax[0][1].yaxis.set_label_position("right")
        ax[0][1].xaxis.tick_top()
        ax[0][1].xaxis.set_label_position("top")
        # ax[0][1].xaxis.set_label_position("top")

        # ax.set_yticklabels(col_sen,size='small')
        # ax[0][0].set_yticklabels([''+str((abs(min(col_sen)) + max(col_sen))/2)],size='small',minor=True,visible=False)
        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[0][1].set_xticks(np.linspace(min(grid_2_col + grid_3_col), max(grid_2_col + grid_3_col), 5))
        ax[0][1].set_xlim(median_col - 0.02, max(grid_2_col + grid_3_col))
        # ax[0][0].set_xticklabels(list(set(overall_college_score)),size='small')

        ax[0][1].scatter(grid_3_col, grid_3_sen, color='green',
                         label="Higher Sentiment, Higher Overall Score \n No. of Colleges : " + str(count_grid_3_college) + ", Avg. Sen : " + str(
                             round(np.mean(np.array(grid_3_sen)), 2)))

        ax[0][1].tick_params(axis='both', which='major', width=1.5, direction='in', left='off', bottom='off')
        # ax[0][1].tick_params(axis='y', which='minor', width=1, direction='in', length=4, left='off')

        ax[0][1].grid(which='major', axis='both', linewidth=1, linestyle='-', color='grey', alpha=0.7)
        # ax[0][1].legend(loc=1, fontsize='small')

        for j, item in enumerate(grid_3_sen):
            if grid_3_col[j] >= ((max(grid_2_col + grid_3_col) - median_col) / 1.3) + median_col and grid_3_sen[j] >= (
                0 + max(grid_3_sen + grid_4_sen)) / 1.3 or grid_3_col[j] <= ((max(grid_2_col + grid_3_col) - median_col) / 1.3) + median_col and \
                            grid_3_sen[j] >= (0 + max(grid_3_sen + grid_4_sen)) / 1.1:
                # print grid_3_col[j], max(grid_2_col+grid_3_col) - median_col
                ax[0][1].annotate(college_3_grid[j] + " - " + str(college_rows_3_grid[j]), xy=(grid_3_col[j], grid_3_sen[j]), xytext=(5, 20),
                                  textcoords='offset points', weight='normal', fontsize='small', ha='center', va='center',
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')
        ax[0][1].legend(loc='lower left', fontsize='small', framealpha=0.3, borderpad=0.1)

    # buffer_arr = [mid_y_val for _ in grid_1_sen]
    # buffer_arr = np.array(buffer_arr)

    # ax[0][0].set_xlabel('college_score', labelpad=25, family='monospace', weight='bold')
    # ax[0][0].set_ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')

    # ax[0][1].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen)-0.1,max(grid_1_sen+grid_2_sen)+0.1,20),minor=True)
    # print min(col_sen), min(grid_1_sen+grid_2_sen)
    if grid_2_sen:
        ax[1][1].set_yticks(np.linspace(min(grid_1_sen + grid_2_sen), max(grid_1_sen + grid_2_sen), 10))
        ax[1][1].set_ylim(min(grid_1_sen + grid_2_sen), mid_y_val + 0.02)

        ax[1][1].yaxis.tick_right()
        ax[1][1].yaxis.set_label_position("right")

        ax[1][1].yaxis.set_minor_locator(loc_sen)
        # ax[1][0].set_yticks([''+str(mid_y_val)],minor=True)
        rep_y = ['' + str(mid_y_val)]
        ax[1][1].set_yticklabels(np.repeat(rep_y, 1), size='large', weight='bold', minor=True)

        # ax[0][1].xaxis.set_label_position("top")

        # ax.set_yticklabels(col_sen,size='small')
        # ax[0][0].set_yticklabels([''+str((abs(min(col_sen)) + max(col_sen))/2)],size='small',minor=True,visible=False)
        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[1][1].set_xticks(np.linspace(min(grid_2_col + grid_3_col), max(grid_2_col + grid_3_col), 5))
        ax[1][1].set_xlim(median_col - 0.02, max(grid_2_col + grid_3_col))
        # ax[0][0].set_xticklabels(list(set(overall_college_score)),size='small')

        ax[1][1].scatter(grid_2_col, grid_2_sen, color='black',
                         label="Lower Sentiment, Higher Overall Score \n No. of Colleges : " + str(count_grid_2_college) + ", Avg. Sen : " + str(
                             round(np.mean(np.array(grid_2_sen)), 2)))

        ax[1][1].tick_params(axis='both', which='major', width=1.5, direction='in', left='off', top='off')
        ax[1][1].tick_params(axis='y', which='minor', width=1, direction='in', left='off', pad=50)

        ax[1][1].grid(which='major', axis='both', linewidth=1, linestyle='-', color='grey', alpha=0.7)
        ax[1][1].grid(which='minor', axis='y', linewidth=1.5, linestyle='-', color='b')

        for j, item in enumerate(grid_2_sen):
            if grid_2_col[j] >= ((max(grid_2_col + grid_3_col) - median_col) / 1.3) + median_col and grid_2_sen[j] <= (
                0 + min(grid_1_sen + grid_2_sen)) / 1.3 or grid_2_col[j] <= ((max(grid_2_col + grid_3_col) - median_col) / 1.3) + median_col and \
                            grid_2_sen[j] <= (0 + min(grid_1_sen + grid_2_sen)) / 1.3 or grid_2_col[j] >= (
                (max(grid_2_col + grid_3_col) - median_col) / 1.5) + median_col and grid_2_sen[j] >= (0 + min(grid_1_sen + grid_2_sen)) / 1.3:
                ax[1][1].annotate(college_2_grid[j] + " - " + str(college_rows_2_grid[j]), xy=(grid_2_col[j], grid_2_sen[j]), xytext=(5, 20),
                                  textcoords='offset points', weight='normal', fontsize='small', ha='left', va='center',
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')

        ax[1][1].legend(loc='upper left', fontsize='small', framealpha=0.3, borderpad=0.1)
    # print buffer_arr
    # fill_arr_sen_grid_1 = np.ndarray(shape=(len(grid_1_sen)), dtype=np.float, buffer = np.array(buffer_arr), order='c')
    # # fill_arr_sen_grid_1.fill(mid_y_val)
    # print fill_arr_sen_grid_1
    if grid_1_sen:
        ax[1][0].set_yticks(np.linspace(min(grid_1_sen + grid_2_sen), max(grid_1_sen + grid_2_sen), 10))
        ax[1][0].set_ylim(min(grid_1_sen + grid_2_sen), mid_y_val + 0.02)

        ax[1][0].yaxis.set_minor_locator(loc_sen)
        # ax[1][0].set_yticks([''+str(mid_y_val)],minor=True)
        rep_y = ['' + str(mid_y_val)]
        ax[1][0].set_yticklabels(np.repeat(rep_y, 1), weight='bold', size='large', minor=True)
        # ax[0][1].yaxis.tick_right()
        # ax[0][1].yaxis.set_label_position("right")
        # ax[0][1].xaxis.set_label_position("top")

        # ax.set_yticklabels(sentiment_score,size='small')

        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[1][0].set_xticks(np.linspace(min(grid_1_col + grid_4_col), max(grid_1_col + grid_4_col), 5))
        ax[1][0].set_xlim(min(grid_1_col + grid_4_col), median_col + 0.02)

        ax[1][0].xaxis.set_minor_locator(loc_col)
        # ax[1][0].xaxis.set_minor_formatter(plticker.FormatStrFormatter('%.2f'))
        # ax[1][0].set_xticks([''+str(median_col)],minor=True)
        rep_x = [('' + str(round(median_col, 3)))]
        ax[1][0].set_xticklabels(np.repeat(rep_x, 2), weight='bold', size='large', minor=True)

        ax[1][0].scatter(grid_1_col, grid_1_sen, color='red',
                         label="Lower Sentiment, Lower Overall Score \n No. of Colleges : " + str(count_grid_1_college) + ", Avg. Sen : " + str(
                             round(np.mean(np.array(grid_1_sen)), 2)))

        ax[1][0].tick_params(axis='both', which='major', width=1.5, direction='in', right='off', top='off')
        ax[1][0].tick_params(axis='y', which='minor', width=1, direction='in', left='off', pad=50)
        ax[1][0].tick_params(axis='x', which='minor', width=1, direction='in', pad=20, right='off')
        # ax[0][1].tick_params(axis='y', which='minor', width=1, direction='in', length=4, left='off')

        ax[1][0].grid(which='major', axis='both', linewidth=1, linestyle='-', color='grey', alpha=0.7)
        ax[1][0].grid(which='minor', axis='x', linewidth=4, linestyle='-', color='b')
        ax[1][0].grid(which='minor', axis='y', linewidth=1.5, linestyle='-', color='b')

        for j, item in enumerate(grid_1_sen):
            if grid_1_col[j] <= ((median_col - min(grid_1_col + grid_4_col)) / 1.3) + min(grid_1_col + grid_4_col) and grid_1_sen[j] >= (
                0 + min(grid_1_sen + grid_2_sen)) / 1.3 or grid_1_col[j] >= ((median_col - min(grid_1_col + grid_4_col)) / 1.3) + min(
                            grid_1_col + grid_4_col) and grid_1_sen[j] <= (0 + min(grid_1_sen + grid_2_sen)) / 1.3 or grid_1_col[j] <= (
                (median_col - min(grid_1_col + grid_4_col)) / 1.3) + min(grid_1_col + grid_4_col) and grid_1_sen[j] <= (
                0 + min(grid_1_sen + grid_2_sen)) / 1.3:
                ax[1][0].annotate(college_1_grid[j] + " - " + str(college_rows_1_grid[j]), xy=(grid_1_col[j], grid_1_sen[j]), xytext=(1, 20),
                                  textcoords='offset points', weight='normal', fontsize='small', ha='right', va='center',
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')

        ax[1][0].legend(loc='upper right', fontsize='small', framealpha=0.3, borderpad=0.1)

    # ax[1][0].annotate(s=text,xy=(0.05,0.05),xytext=(0.05,0.05))


    # fill_arr_col_grid_1 = np.array([1,len(grid_1_col)])
    # fill_arr_col_grid_1.fill(median_col)
    # print grid_1_col,grid_1_sen, buffer_arr

    # ax.fill_between(grid_1_col,grid_1_sen,buffer_arr,where= grid_1_sen <= buffer_arr,facecolor='red',alpha=0.5)


    # for lines,ticks in zip(ax.yaxis.get_ticklabels(),ax.yaxis.get_gridlines()):
    #     lines.
    # ax.set_yticklabels(visible=False)
    # ax.set_xticklabels(visible=False)

    # ax.yaxis.set_major_locator(MaxNLocator(nbins=40, prune='lower'))
    # ax.set_ylim(bottom=min(sentiment_score) - 0.2, top=max(sentiment_score) + 0.2, emit=True)

    # ax.set_xticklabels(overall_college_score_sub, fontdict='monospace', zorder=1, color='black', rotation='horizontal',
    #                    va='top', ha='center', weight='roman', clip_on=False, axes=ax, x =0.015)

    # ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='lower'))

    # ax.grid(which='major',axis='both',linewidth=0.8,linestyle='-',color='grey',alpha=0.8)
    # H, xedges, yedges = np.histogram2d(overall_college_score, sentiment_score,
    #                                        bins=len(overall_college_score),
    #                                        range=[[min(sentiment_score),max(sentiment_score)],
    #                                               [min(overall_college_score),max(overall_college_score)]],
    #                                        normed=True)

    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # # print H
    # ax[0][1].set_xticks(np.linspace(min(overall_college_score)-.5,max(overall_college_score)+.5,10))
    # ax[0][1].set_xlim(min(overall_college_score)-.5,max(overall_college_score)+.5)
    # ax[0][1].set_yticks(np.linspace(min(sentiment_score),max(sentiment_score),20))
    # ax[0][1].set_ylim(min(sentiment_score),max(sentiment_score))
    # overall_college_score  = np.array(overall_college_score)
    # # sentiment_score = [item + abs(min(sentiment_score)) for item in sentiment_score]
    # sentiment_score = np.array(sentiment_score)
    # print overall_college_score
    # print "\n\n",str(len(overall_college_score)),"  ", str(len(sentiment_score))
    # print sentiment_score

    # xbins = np.linspace(0, 1, len(overall_college_score))
    # ybins = np.linspace(0, 1, len(sentiment_score))
    # xbins = [3.0,3.33,3.66,4.0,4.33]
    # ybins = [-0.75,-0.5,0,0.5,0.75]
    #
    # H, xedges, yedges,img = ax[0][1].hist2d(overall_college_score,sentiment_score,
    #                                        bins=len(overall_college_score)/4
    #                                        # range=[[min(sentiment_score),max(sentiment_score)],
    #                                        #        [min(overall_college_score),max(overall_college_score)]],
    #                                        # normed=True)
    #                                   )
    # # print H
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #
    # ax[0][1].imshow(H.T, interpolation='none',
    #                aspect='auto' ,cmap=plt.cm.YlOrRd_r,origin='lower',
    #                 extent= extent,vmax=max(sentiment_score),vmin = min(sentiment_score))

    # ax[0][1].xaxis.set_minor_locator(loc_col)
    # ax[0][1].yaxis.set_minor_locator(loc_sen)
    # ax[0][1].set_yticks([mid_y_val],minor=True)
    # ax[0][1].set_yticklabels([''+str((abs(min(sentiment_score)) + max(sentiment_score))/2)],size='small',minor=True,visible=False)
    # ax[0][1].grid(which='minor',axis='both',linewidth=2,linestyle='-',color='b')

    #
    # ax.colorbar(im, format='%.3f', fraction=0.10, pad=0.09, spacing='proportional')

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
    # plt.grid()
    # ax.legend(loc=0)
    print "Colleges in lower left grid_for_time_" + str(start_date) + "_to_" + str(time_upto) + ": \n"
    for i, val in enumerate(college_1_grid):
        print val, " - ", college_rows_1_grid[i]
        print "\n"

    print "Colleges in lower right grid_for_time_" + str(start_date) + "_to_" + str(time_upto) + " : \n"
    for i, val in enumerate(college_2_grid):
        print val, " - ", college_rows_2_grid[i]
        print "\n"

    print "Colleges in upper right grid_for_time_" + str(start_date) + "_to_" + str(time_upto) + " : \n"
    for i, val in enumerate(college_3_grid):
        print val, " - ", college_rows_3_grid[i]
        print "\n"

    print "Colleges in upper left grid_for_time_" + str(start_date) + "_to_" + str(time_upto) + " : \n",
    for i, val in enumerate(college_4_grid):
        print val, " - ", college_rows_4_grid[i]
        print "\n"

    text = "Total Colleges : " + str(len(col_sen)) + ", Total Avg. Sen : " + str(round(np.mean(np.array(col_sen)), 2))
    f.text(0.003, 0.93, text, fontsize=12, bbox=dict(facecolor='grey', alpha=0.5, pad=7.0, edgecolor='red'), weight='bold', color='black')

    plt.show()
    # f.tight_layout()
    img_name = "Colleges_Correlation_between_Sentiment_&_Overall_Score_for_time_" + str(start_date) + "_to_" + str(
        time_upto) + "for colleges > 50 tweets.svg"
    f.savefig('test_' + img_name, format='svg', dpi=600)
    pp = PdfPages("test_Colleges_Correlation_between_Sentiment_&_Overall_Score_for_time_" + str(start_date) + "_to_" + str(
        time_upto) + "for colleges > 50 tweets.pdf")
    pp.savefig(f)
    pp.close()

    # def plot_ranks_heatmap(returns_data,filename=None,col1='min_rank',col2='order'):
    #     returns_dataframe = returns_data.copy()
    # #     returns_dataframe[col1] = np.log10(returns_dataframe[col1])
    # #     returns_dataframe[col2] = np.log10(returns_dataframe[col2])
    #     returns_dataframe['order'] = returns_dataframe['order']-1
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.grid()
    #     H, xedges, yedges = np.histogram2d(returns_dataframe[col1].values,
    #                                        returns_dataframe[col2].values,
    #                                        bins=(
    # #             4**np.ceil(returns_dataframe[col1].max()),
    # #             4**np.ceil(returns_dataframe[col2].max())
    #             returns_dataframe[col1].max(),
    #             returns_dataframe[col2].max(),
    #
    #         ),
    #                                        range=[[returns_dataframe[col1].min(),returns_dataframe[col1].max()],
    #                                               [returns_dataframe[col2].min(),returns_dataframe[col2].max()]],
    #                                        normed=True)
    # #     extent = [10**xedges[0], 10**xedges[-1], 10**yedges[0], 10**yedges[-1]]
    #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #     im = ax.imshow(H.T, interpolation='none',
    #                    aspect='auto' ,cmap='jet',origin='lower',
    #                     extent= extent,
    #                    norm=LogNorm(),
    # #                    vmax=1.0e4,
    # #                    vmin = 0
    #                    )
    #     if 'zoomed' not in filename:
    #         fig.colorbar(im,
    #                      format=LogFormatterMathtext(),
    # #                      shrink=0.55,
    #
    # #                      label='$p(K_{f},K_{s}$)'
    #                     )
    #         # plt.plot(np.arange(-3,100),np.arange(-3,100),linewidth=5)
    # #     ax.set_xscale('log')
    # #     ax.set_yscale('log')
    #     ax.grid(False)
    # #     ax.set_xscale('log')
    # #     ax.set_yscale('log')
    #     ax.set_xlabel("$K_{f}$")
    #     ax.set_ylabel("$K_{s}$")
    #     ax.set_xlim((0,100))
    #     ax.set_ylim((0,100))
    #     fig.tight_layout()
    #     if filename:
    #         save_fig(fig,filename)
    #     else:
    #         return fig
