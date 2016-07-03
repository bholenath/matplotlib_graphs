import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

sentiment_score = []
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
            [val_pub[2], val_pub[3], val_pub[4], val_pub[5], val_pub[6], val_pub[7], val_pub[8], val_pub[9], val_pub[10], val_pub[11], val_pub[12],
             val_pub[13], val_pub[14], val_pub[15], val_pub[16], val_pub[17], val_pub[18], val_pub[19], val_pub[20]])
        # college_name.append(row[0])

    else:
        query12 = """select * from pvt_college_aspects_ranking where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query12)
        val_pvt = cursor.fetchone()

        if val_pvt is not None:
            # sentiment_score.append(row[1])
            sub_aspects_score.extend(
                [val_pvt[2], val_pvt[3], val_pvt[4], val_pvt[5], val_pvt[6], val_pvt[7], val_pvt[8], val_pvt[9], val_pvt[10], val_pvt[11], val_pvt[12],
                 val_pvt[13], val_pvt[14], val_pvt[15], val_pvt[16], val_pvt[17], val_pvt[18], val_pvt[19], val_pvt[20]])
            # college_name.append(row[0])

        else:
            print "Can't be possible! Table not in saved universities.", row[0]
            continue

    college_name.append(row[0])
    sentiment_score.append(row[1])
    complete_aspects_score.append(sub_aspects_score)

    # ind = np.arange(len(sentiment_score))
    # width = 0.6

fig = plt.figure(figsize=(1600 / 95.8, 900 / 95.8), dpi=96)
ax = fig.add_subplot(111)

index = 0

while index < number_aspects:

    college_aspects_score = []

    for item in complete_aspects_score:
        college_aspects_score.append(item[index])

    print "graph for : ", aspects_value[index], " with values : ", college_aspects_score
    college_aspects_score = np.array(college_aspects_score)

    pearR = np.corrcoef(college_aspects_score, sentiment_score)[1, 0]

    A = np.vstack([college_aspects_score, np.ones(len(college_aspects_score))]).T
    m, c = np.linalg.lstsq(A, np.array(sentiment_score))[0]
    plt.scatter(college_aspects_score, sentiment_score, label='Data Red', color='r')
    plt.plot(college_aspects_score, college_aspects_score * m + c, color='r', label="Fit %6s, r = %6.2e" % ('RED', pearR))

    # plt.scatter(college_aspects_score, sentiment_score, color='r')

    plt.title('Colleges_Correlation_between_' + aspects_value[index] + '_&_Overall_Score', y=1.02, style='italic', family='cursive', weight='bold')
    plt.xlabel('' + aspects_value[index] + '_score', labelpad=25, family='cursive', weight='bold')
    plt.ylabel('avg_sentiment_val', labelpad=25, family='cursive', weight='bold')
    # plt.xticks(np.arange((min(college_aspects_score) - 2), (max(college_aspects_score) + 2), 0.2))
    # plt.yticks(np.arange((min(sentiment_score) - 0.2), (max(sentiment_score) + 0.2), 0.03))

    # ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=30, prune=None))
    ax.set_ylim(bottom=min(sentiment_score) - 0.2, top=max(sentiment_score) + 0.2, emit=True)
    # width = ind * 3
    # height = max(average_val_data) + 0.2

    # plt.figure(figsize=('%s', '%s') % (width, height))
    # check_overlap = [[-10, -10]]
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

    # for i, rect in enumerate(rects):
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width(), height + 0.18, '(%s,%s)' % (sentiment_score[i], college_aspects_score[i]),
    #              ha='center', va='top', rotation='vertical')

    # plt.tight_layout()
    plt.grid()
    plt.legend(loc=3)
    plt.show()
    img_name = "graph_correlation_pearson_overall_sentiment.png"
    fig.savefig(img_name, dpi=96)
    index += 1
