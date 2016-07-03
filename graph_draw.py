import MySQLdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from warnings import filterwarnings

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8",
                        use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)

cnxn = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="college_info", charset="utf8", use_unicode=True)
cursor = cnxn.cursor()
cnxn.autocommit(True)

query1 = "set names 'utf8mb4'"
query2 = "SET CHARACTER SET utf8mb4"

filterwarnings('ignore')

# making database accept utf8mb4 as the data format in their columns
cursor1.execute(query1)
cursor1.execute(query2)

prev = 0
count = 0
all_college_sentiments = []

# query0 = """select concat('select', 'count(1) "rowcount"', 'from', db, '.', tb) st from (select table_schema db, table_name tb
#             from information_schema.tables where table_schema='final_twitter_data' and table_name='college_sentiment_average')
#              t"""
query0 = "select count(id) from college_sentiment_average"
cursor1.execute(query0)
rows_count = cursor1.fetchone()

for row in rows_count:
    while prev < row:
        query = "select college_name,sentiment_average,no_rows from college_sentiment_average order by id limit %s,%s"
        cursor1.execute(query, [prev, 45])
        sent_val = cursor1.fetchall()
        all_college_sentiments.append(sent_val)
        prev += 45

for item in all_college_sentiments:

    college_data = []
    average_val_data = []
    tweets_analyzed = []
    count += 1

    for row in item:
        college_data.append(row[0][:43])
        average_val_data.append(row[1])
        tweets_analyzed.append(row[2])

    ind = np.arange(len(college_data))
    # width = 0.6

    colors = []

    for i, val in enumerate(average_val_data):
        if val > 0:
            colors.append('b')
        else:
            colors.append('r')

    # width = ind * 3
    # height = max(average_val_data) + 0.2

    f, ax = plt.subplots(1, 1, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)
    # fig = plt.figure(figsize=(1600 / 95.8, 900 / 95.8), dpi=96)
    # ax = fig.add_subplot(111)
    # ax.set_aspect('auto')
    # ax = plt.subplot(1, 1, 1)

    ax.set_title('Colleges_Sentiment_Average', y=1.02, style='italic', family='cursive', weight='bold')
    ax.set_xlabel('college_name', labelpad=25, family='cursive', weight='bold')
    ax.set_ylabel('avg_sentiment_val', labelpad=25, family='cursive', weight='bold')
    plt.xticks(ind + 0.53, [val for j, val in enumerate(college_data)], rotation='vertical', color='b',
               verticalalignment='bottom', horizontalalignment='center', weight='roman', family='cursive', y=0.03)
    # plt.yticks(np.arange(min(average_val_data) - 0.2, max(average_val_data) + 0.2, 0.1))
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    # ax.yaxis.set_ticks(np.arange(min(average_val_data) - 0.2, max(average_val_data) + 0.2, 0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    ax.set_ylim(bottom=min(average_val_data) - 0.2, top=max(average_val_data) + 0.2, emit=True)
    plt.margins(0.009)
    top = max(average_val_data) + 0.25
    plt.text(1.3, top, "() : tweets_analyzed", ha='center', family='serif', weight='bold', size='large', color='r')
    # plt.subplots_adjust(bottom=0.15)

    rects = plt.bar(ind, average_val_data, color=colors)

    for i, rect in enumerate(rects):
        height = rect.get_height()

        if abs(height) > 0.6:
            disp_height = height - 0.2
        else:
            disp_height = height + 0.267

        plt.text(rect.get_x() + 0.53, disp_height, '%s  (%s)' % ((round(float(average_val_data[i]), 3)), tweets_analyzed[i]),
                 ha='center', va='top', rotation='vertical')

    # plt.tight_layout()
    plt.show()
    img_name = "graph_" + str(count) + "_sentiment.png"
    fig.savefig(img_name, dpi=96)

    # for row in sent_val:
    #     query1 = """select * from pub_college_aspects_ranking where replace(college_name, ' ', '') = """ + row[0]
    #     cursor.execute(query1)
    #     val_pub = cursor.fetchone()
    #
    #     if val_pub is not None:
    #         pass
    #
    #     else:
    #         query2 = """select * from pvt_college_aspects_ranking where replace(college_name, ' ', '') = """ + row[0]
    #         cursor.execute(query2)
    #         val_pvt = cursor.fetchone()
    #
    #         if val_pvt is not None:
    #             pass
    #
    #         else:
    #             print "Can't be possible! Table not in saved universities."
    #             sys.exit()
