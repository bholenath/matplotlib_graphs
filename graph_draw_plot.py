import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import numpy as np

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

prev = 0
count = 0
all_college_sentiments = []

query0 = "select count(id) from college_sentiment_average"
cursor1.execute(query0)
rows_count = cursor1.fetchone()

for row in rows_count:
    while prev < row:
        query = "select college_name,sentiment_average from college_sentiment_average order by id limit %s,%s"
        cursor1.execute(query, [prev, 30])
        sent_val = cursor1.fetchall()
        all_college_sentiments.append(sent_val)
        prev += 30

for item in all_college_sentiments:
    sentiment_score = []
    overall_college_score = []
    college_name = []
    count += 1

    for row in item:
        query11 = """select overall_grade from pub_college_overall_rank where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query11)
        val_pub = cursor.fetchone()

        if val_pub is not None:
            sentiment_score.append(row[1])
            overall_college_score.append(val_pub[0])
            college_name.append(row[0])

        else:
            query12 = """select overall_grade from pvt_college_overall_rank where replace(replace(replace(replace(replace
                         (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
            cursor.execute(query12)
            val_pvt = cursor.fetchone()

            if val_pvt is not None:
                sentiment_score.append(row[1])
                overall_college_score.append(val_pvt[0])
                college_name.append(row[0])

            else:
                print "Can't be possible! Table not in saved universities.", row[0]
                continue

    # ind = np.arange(len(sentiment_score))
    # width = 0.6

    fig = plt.figure(figsize=(1600 / 95.8, 900 / 95.8), dpi=96)
    ax = fig.add_subplot(111)

    plt.plot(overall_college_score, sentiment_score, color='r', marker='o', linestyle='')

    plt.title('Colleges_Correlation_between_Sentiment_&_Overall_Score', y=1.02, style='italic', family='cursive', weight='bold')
    plt.xlabel('college_score', labelpad=25, family='cursive', weight='bold')
    plt.ylabel('avg_sentiment_val', labelpad=25, family='cursive', weight='bold')
    # plt.xticks(np.arange((min(overall_college_score) - 2), (max(overall_college_score) + 2), 0.2))
    # plt.yticks(np.arange((min(sentiment_score) - 0.2), (max(sentiment_score) + 0.2), 0.03))

    # ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=50, prune=None))
    # width = ind * 3
    # height = max(average_val_data) + 0.2

    # plt.figure(figsize=('%s', '%s') % (width, height))
    check_overlap = [[-10, -10]]

    for label, x, y in zip(college_name, overall_college_score, sentiment_score):
        count = 0
        for i, content in enumerate(check_overlap):
            print content
            print "y : ", y, "x : ", x
            if (abs(y - content[1]) > 0.1) and (abs(x - content[0]) > 0.2):
                count += 1
            else:
                continue
        if count >= 6:
            print count, " : ", len(check_overlap)
            print "-"
            check_overlap.append([x, y])
            ax.annotate(label[:20], xy=(x, y), xytext=(-20, 20), textcoords='offset points', fontsize='small', ha='right',
                        va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6), arrowprops=dict(
                        arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')
        else:
            print count, " : ", len(check_overlap)
            print "+"
            check_overlap.append([x, y])
            ax.annotate(label[:20], xy=(x, y), xytext=(8, 20), textcoords='offset points', fontsize='small', ha='left',
                        va='top', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6), arrowprops=dict(
                        arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')

    # for i, rect in enumerate(rects):
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width(), height + 0.18, '(%s,%s)' % (sentiment_score[i], overall_college_score[i]),
    #              ha='center', va='top', rotation='vertical')

    # plt.tight_layout()
    plt.grid()
    plt.show()
    img_name = "graph_" + str(count) + "_correlation_overall_sentiment.png"
    fig.savefig(img_name, dpi=96)
