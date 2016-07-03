import MySQLdb
import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8", use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)

cnxn2 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="college_keywords_by_time", charset="utf8", use_unicode=True)
cursor2 = cnxn2.cursor()
cnxn2.autocommit(True)

query0 = """select * from college_sentiment_average where no_rows >= 200"""
cursor1.execute(query0)
fetch_col = cursor1.fetchall()

for row in fetch_col:

    sen_values = []
    count_values = []
    date_values = []

    try:
        query1 = "select * from college_sentiment_average_per_day where college_name in (%s) order by id"
        cursor1.execute(query1, [str(row[1])])
        for data in cursor1:
            if data[3] < 5:  # removing those days which have tweets less than 5
                continue
            else:
                if data[2] is None:
                    sen_values.append(0.0)
                else:
                    sen_values.append(data[2])
                count_values.append(int(data[3]))
                date_values.append(data[4])

        if len(date_values) < 10:  # removing those colleges which have less than 10 days to show after all calculations
            continue

        f, ax1 = plt.subplots(1, 1, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)
        plt.subplots_adjust(bottom=0.11)
        ax1.spines['left'].set_color('blue')

        # print len(sen_values), " ",len(count_values), " ", len(date_values)
        ax1.set_title('Plot_of_' + str(row[1]) + '_with_Sentiment_&_No_Tweets_per_Day', y=1.02, style='italic', family='monospace', weight='bold', size='large')

        ax1.set_xlabel('Date', labelpad=3, family='monospace', weight='bold', size='medium')
        ax1.set_ylabel('Tweets_Sentiment', labelpad=10, family='monospace', weight='bold', size='medium', color='b')

        # ax1.xaxis.set_major_locator(MaxNLocator(nbins=len(date_values), prune=None))

        yaxis_ticks = np.linspace(min(sen_values) - 0.1, max(sen_values) + 0.1, 20)
        # minor_ticks = np.linspace(min(sen_values) - 0.1,max(sen_values) + 0.1,40)
        ax1.set_yticks(yaxis_ticks)
        # ax1.set_yticks(minor_ticks,minor=True)

        ax1.set_yticklabels(yaxis_ticks, zorder=1, color='b', rotation='0', va='center', ha='right', weight='roman', clip_on=True, axes=ax1, size='medium')
        # print "fa"
        ax1.set_ylim(min(sen_values) - 0.1, max(sen_values) + 0.1)

        ax1.set_xticks(date_values)
        ax1.set_xticklabels(date_values, zorder=1, color='black', rotation='vertical',
                            va='top', ha='left', weight='roman', clip_on=True, axes=ax1, size='medium')
        ax1.set_xlim(date_values[0] - datetime.timedelta(days=2), date_values[-1] + datetime.timedelta(days=2))

        ax1.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, colors='blue')
        # ax1.tick_params(axis='y', which='minor', direction='out', length=3, width=0.5, labelsize=0, colors='blue')
        ax1.tick_params(axis='x', which='both', direction='out', length=6, width=0.5, labelsize=8, colors='black', top='off')

        plot_1 = ax1.plot_date(date_values, sen_values, color='b', marker='o', linestyle='-', rasterized=True, antialiased=True, label='Tweets_Sentiment',
                               zorder=1)

        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.grid(which='major', color='grey', linestyle='--', linewidth=1, alpha=0.8)
        ax1.grid(which='minor', color='grey', linestyle='--', linewidth=1, alpha=0.4)

        ax2 = ax1.twinx()

        ax2.set_ylabel('No._of_Tweets', labelpad=10, family='monospace', weight='bold', size='medium', color='r')

        ax2.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, colors='red')
        # ax2.tick_params(axis='y', which='minor', direction='in', length=2, width=0.5, colors='red')
        # ax2.spines['right'].set_color('red')
        yaxis_ticks_1 = np.linspace(min(count_values) - 1, max(count_values) + 5, 20)
        # minor_ticks_1 = np.linspace(min(count_values) - 1,max(count_values) + 5,60,dtype='int32')
        ax2.set_yticks(yaxis_ticks_1)
        ax2.spines['right'].set_color('red')

        # ax2.set_yticks(minor_ticks_1,minor=True)
        ax2.set_yticklabels(yaxis_ticks_1, zorder=1, color='red', rotation='0',
                            va='center', ha='left', weight='roman', clip_on=True, axes=ax2, size='medium')
        ax2.set_ylim(min(count_values) - 1, max(count_values) + 5)

        plot_2 = ax2.plot_date(date_values, count_values, color='r', marker='o', linestyle='-', rasterized=True, antialiased=True, label='Tweets_Count',
                               zorder=1)
        ax2.set_xlim(date_values[0] - datetime.timedelta(days=2), date_values[-1] + datetime.timedelta(days=2))

        ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        for j, item in enumerate(sen_values):
            if j == 0:
                continue
            else:
                if item - sen_values[j - 1] >= (max(sen_values) - min(sen_values)) / 3.0:
                    query99 = """select * from """ + row[
                        1] + """_tweets_keywords where date_tweet in (%s) order by key_frequency desc,mean_keyword_sentiment desc"""
                    cursor2.execute(query99, [str(date_values[j])])
                    text = ""
                    i = 0
                    for key in cursor2:
                        if key[3] >= 0.0:
                            if i <= 5:
                                text += key[1].title() + " / " + str(key[2]) + " / " + str(key[3]) + "\n"
                                # print date_values[j]
                                i += 1
                            else:
                                break
                    if i == 0:
                        for key in cursor2:
                            if i <= 5:
                                text += key[1].title() + " / " + str(key[2]) + " / " + str(key[3]) + "\n"
                                # print date_values[j]
                                i += 1
                            else:
                                break
                    ax1.annotate(text[:text.rfind('\n')], xy=(date_values[j], sen_values[j]), xycoords='data', xytext=(-20, 20),
                                 textcoords='offset points', weight='normal', fontsize='small', ha='right', va='center',
                                 bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.6),
                                 arrowprops=dict(facecolor='black', shrink=0.05), color='b',
                                 zorder=0)

                elif item - sen_values[j - 1] <= -((max(sen_values) - min(sen_values)) / 3.0):
                    query99 = """select * from """ + row[
                        1] + """_tweets_keywords where date_tweet in (%s) order by key_frequency desc,mean_keyword_sentiment asc"""
                    cursor2.execute(query99, [str(date_values[j])])
                    text = ""
                    i = 0
                    for key in cursor2:
                        if key[3] <= 0.0:
                            if i <= 5:
                                text += key[1] + " / " + str(key[2]) + " / " + str(key[3]) + "\n"
                                i += 1
                            else:
                                break
                    if i == 0:
                        for key in cursor2:
                            if i <= 5:
                                text += key[1].title() + " / " + str(key[2]) + " / " + str(key[3]) + "\n"
                                # print date_values[j]
                                i += 1
                            else:
                                break
                    ax1.annotate(text[:text.rfind('\n')], xy=(date_values[j], sen_values[j]), xycoords='data',
                                 xytext=(-20, 20),textcoords='offset points',
                                 weight='normal', fontsize='small', ha='right', va='center',
                                 bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.6),
                                 arrowprops=dict(facecolor='black', shrink=0.05), color='b',
                                 zorder=0)

        lns = plot_1 + plot_2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        plt.show()
        # plt.grid()
        img_name = 'annotated_Plot_of_' + str(row[1]) + '_with_Sentiment_&_No_Tweets_per_Day.svg'
        f.savefig('./per_day_tweets_for_each_college/' + img_name, format='svg', dpi=600)
        plt.close(f)

    except Exception, e:
        print "Error in plotting : ", e
        sys.exit(1)
