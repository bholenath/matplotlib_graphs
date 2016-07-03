import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter,MaxNLocator
import matplotlib.ticker as plticker
# import matplotlib.finance as fin
# import sys
import numpy as np
# import scipy.optimize as opt
# import csv
import datetime

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
query = "select college_name,sentiment_average from college_sentiment_average where no_rows >= 200 order by id limit %s," \
        "%s"
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

# def func(x,a,b,c1):
#     return a*np.exp(-b*x)-c1

for name in cursor:
    aspects_value.append(name[0])

for row in sent_val:

    sub_aspects_score = []

    query11 = """select * from pub_college_aspects_ranking where replace(replace(replace(replace(replace
                 (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
    cursor.execute(query11)
    val_pub = cursor.fetchone()

    if val_pub is not None:
        sentiment_score.append(row[1])
        # aspects_value.append(val_pub[0])
        sub_aspects_score.extend([val_pub[2],val_pub[3],val_pub[4],val_pub[5],val_pub[6],val_pub[7],val_pub[8],val_pub[9],val_pub[10],val_pub[11],val_pub[12],val_pub[13],val_pub[14],val_pub[15],val_pub[16],val_pub[17],val_pub[18],val_pub[19],val_pub[20]])
        college_name.append(row[0])

    else:
        query12 = """select * from pvt_college_aspects_ranking where replace(replace(replace(replace(replace
                     (college_name, ',', ''), ' ', ''), '-', ''), '&', ''), '.', '') = '""" + row[0] + """'"""
        cursor.execute(query12)
        val_pvt = cursor.fetchone()

        if val_pvt is not None:
            sentiment_score.append(row[1])
            sub_aspects_score.extend([val_pvt[2],val_pvt[3],val_pvt[4],val_pvt[5],val_pvt[6],val_pvt[7],val_pvt[8],val_pvt[9],val_pvt[10],val_pvt[11],val_pvt[12],val_pvt[13],val_pvt[14],val_pvt[15],val_pvt[16],val_pvt[17],val_pvt[18],val_pvt[19],val_pvt[20]])
            college_name.append(row[0])

        else:
            print "Can't be possible! Table not in saved universities.", row[0]
            continue

    complete_aspects_score.append(sub_aspects_score)

            # ind = np.arange(len(sentiment_score))
            # width = 0.6
index = 0

while index < number_aspects:

    # f,ax = plt.subplots(1,1,figsize=(1600 / 95.9, 900 / 95.9), dpi=96)
    # fig = plt.figure(figsize=(1600 / 95.8, 900 / 95.8), dpi=96)
    # ax = fig.add_subplot(111)

    f,ax = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(1600 / 95.9, 900 / 95.9), dpi=96)

    x_Formatter = plticker.FormatStrFormatter('%.2f')
    y_Formatter = plticker.FormatStrFormatter('%.3f')

    for axes1 in ax:
        for axes in axes1:
            axes.xaxis.set_major_formatter(x_Formatter)
            axes.xaxis.set_minor_formatter(x_Formatter)
            axes.yaxis.set_major_formatter(y_Formatter)
            axes.yaxis.set_minor_formatter(y_Formatter)

    ax[1,0].spines['right'].set_visible(False)
    ax[1,0].spines['top'].set_visible(False)
    ax[1,1].spines['left'].set_visible(False)
    ax[1,1].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].spines['bottom'].set_visible(False)
    ax[0,1].spines['left'].set_visible(False)
    ax[0,1].spines['bottom'].set_visible(False)
    # axs[1].spines['bottom'].set_visible(False)
    # ax[0][0],ax[0][1] = ax
    # axes = ax
    # ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0,hspace=0)

    f.suptitle('Colleges_Correlation_between_'+aspects_value[index].capitalize()+'_&_College_tweets_Sentiment', style='italic', family='monospace', weight='bold', size='large',ha='center',va='top')

    # f.set_title('Colleges_Correlation_between_'+aspects_value[index].capitalize()+'_&_College_tweets_Sentiment', y=1.02, style='italic', family='monospace', weight='bold')
    f.text(0.5, 0.01, ''+aspects_value[index].capitalize()+'_score', ha='center', family='monospace', weight='bold')
    f.text(0.01, 0.5, 'Avg_Sentiment_Value', va='center', rotation='vertical', family='monospace', weight='bold')
    # ax.set_xlabel(, labelpad=25, family='monospace', weight='bold')
    # ax.set_ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')

    valid_show_aspects = []
    valid_sentiment_aspects = []

    # writer = csv.writer(open('data_correlation_pearson_'+aspects_value[index]+'_sentiment.csv','w'))
    # writer.writerow(['id','college_name',aspects_value[index]+'_score','sentiment_score'])

    col_count = 0
    # print "graph for : ", aspects_value[index], "\n"
    for item in complete_aspects_score:
        # college_aspects_score.append(valid_show_aspects)
        # writer.writerow([col_count+1,college_name[col_count],item[index],sentiment_score[col_count]])
        if item[index] > 0:
            valid_show_aspects.append(item[index])
            valid_sentiment_aspects.append(sentiment_score[col_count])
            # print "with aspect value : ", item[index], "equivalent sentiment of : ", sentiment_score[col_count]

        col_count+=1

    # valid_show_aspects = []
    median_col = np.mean(np.array(list(set(valid_show_aspects))))
    # print len(col_sen)
    # mid_y_val = min(valid_sentiment_aspects) + (abs(min(valid_sentiment_aspects)) + max(valid_sentiment_aspects))/2
    mid_y_val = 0.0
    # valid_show_aspects = np.array(valid_show_aspects)
    #
    # # data_add = min(valid_sentiment_aspects)
    #
    # # making all sentiments positive
    # # valid_sentiment_aspects = [value+abs(data_add) for value in valid_sentiment_aspects]
    #
    pearR = np.corrcoef(np.array(valid_show_aspects), np.array(valid_sentiment_aspects))[1,0]
    # #
    # A = np.vstack([valid_show_aspects,np.ones(len(valid_show_aspects))]).T
    # m,c = np.linalg.lstsq(A,np.array(valid_sentiment_aspects))[0]

    grid_1_sen = []
    grid_2_sen = []
    grid_3_sen = []
    grid_4_sen = []
    grid_1_col = []
    grid_2_col = []
    grid_3_col = []
    grid_4_col = []
    count_grid_1_college,count_grid_2_college,count_grid_3_college,count_grid_4_college = 0,0,0,0

    for sen,col in zip(valid_sentiment_aspects,valid_show_aspects):
        if sen <= mid_y_val and col <= median_col:
            # col_arr.append('red')
            grid_1_sen.append(sen)
            grid_1_col.append(col)
            count_grid_1_college +=1
            # labels.append('Expected Values')
        elif sen <= mid_y_val and col > median_col:
            # col_arr.append('violet')
            grid_2_sen.append(sen)
            grid_2_col.append(col)
            count_grid_2_college +=1
            # labels.append('UNEXPECTED VALUES')
        elif sen > mid_y_val and col > median_col:
            # col_arr.append('green')
            grid_3_sen.append(sen)
            grid_3_col.append(col)
            count_grid_3_college +=1
            # labels.append('Expected Values')
        elif sen > mid_y_val and col <= median_col:
            # col_arr.append('violet')
            grid_4_sen.append(sen)
            grid_4_col.append(col)
            count_grid_4_college +=1
            # labels.append('UNEXPECTED VALUES')

    loc_col = plticker.MultipleLocator(base=(median_col+0.02))
    # val_tick = (abs(min(col_sen)) + max(col_sen))/2
    # locs = dict(vmin=val_tick,vmax=val_tick)
    # val_tick = mid_y_val
    loc_sen = plticker.FixedLocator(np.linspace((mid_y_val+0.02),(mid_y_val+0.02),1))

    if grid_4_sen:
        ax[0][0].set_yticks(np.linspace(min(grid_3_sen+grid_4_sen),max(grid_3_sen+grid_4_sen),10))
        ax[0][0].set_ylim(mid_y_val-0.02,max(grid_3_sen+grid_4_sen))

        ax[0][0].xaxis.tick_top()
        ax[0][0].xaxis.set_label_position("top")
        # ax.set_yticklabels(col_sen,size='small')

        # ax[0][0].set_yticklabels([''+str((abs(min(col_sen)) + max(col_sen))/2)],size='small',minor=True,visible=False)
        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[0][0].set_xticks(np.linspace(min(grid_1_col+grid_4_col),max(grid_1_col+grid_4_col),5))
        ax[0][0].set_xlim(min(grid_1_col+grid_4_col)-0.02,median_col+0.02)

        ax[0][0].xaxis.set_minor_locator(loc_col)
        # ax[0][0].xaxis.set_minor_formatter(plticker.FormatStrFormatter('%.2f'))
        # ax[1][0].set_xticks([''+str(median_col)],minor=True)
        rep_x = [(''+str(round(median_col,3)))]
        ax[0][0].set_xticklabels(np.repeat(rep_x,2), weight='bold', size='large', minor=True)
        # ax[0][0].set_xticklabels(list(set(overall_college_score)),size='small')

        ax[0][0].scatter(grid_4_col, grid_4_sen, color='black', label = "Higher Sentiment, Lower Aspect Score \n No. of Colleges : "+str(count_grid_4_college)+", Avg. Sen : "+str(round(np.mean(np.array(grid_4_sen)),2)))

        ax[0][0].tick_params(axis='both', which='major', width=1.5, direction='in', right='off', bottom = 'off')
        ax[0][0].tick_params(axis='x', which='minor', width=1, direction='in', pad=20, right='off')

        ax[0][0].grid(which='major',axis='both',linewidth=1,linestyle='-',color='grey',alpha=0.7)
        ax[0][0].grid(which='minor',axis='x',linewidth=4,linestyle='-',color='b')
        ax[0][0].legend(loc=2, fontsize='small')
    #
    # ax[0][1].set_xlabel('college_score', labelpad=25, family='monospace', weight='bold')
    # ax[0][1].set_ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')

    if grid_3_sen:
        # ax[0][1].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen)-0.1,max(grid_1_sen+grid_2_sen)+0.1,20),minor=True)
        ax[0][1].set_yticks(np.linspace(min(grid_3_sen+grid_4_sen),max(grid_3_sen+grid_4_sen),10))
        ax[0][1].set_ylim(mid_y_val-0.02,max(grid_3_sen+grid_4_sen))

        ax[0][1].yaxis.tick_right()
        ax[0][1].yaxis.set_label_position("right")
        ax[0][1].xaxis.tick_top()
        ax[0][1].xaxis.set_label_position("top")
        # ax[0][1].xaxis.set_label_position("top")

        # ax.set_yticklabels(col_sen,size='small')
        # ax[0][0].set_yticklabels([''+str((abs(min(col_sen)) + max(col_sen))/2)],size='small',minor=True,visible=False)
        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[0][1].set_xticks(np.linspace(min(grid_2_col+grid_3_col),max(grid_2_col+grid_3_col),5))
        ax[0][1].set_xlim(median_col-0.02,max(grid_2_col+grid_3_col)+0.02)
        # ax[0][0].set_xticklabels(list(set(overall_college_score)),size='small')

        ax[0][1].scatter(grid_3_col, grid_3_sen, color='green', label = "Higher Sentiment, Higher Overall Score \n No. of Colleges : "+str(count_grid_3_college)+", Avg. Sen : "+str(round(np.mean(np.array(grid_3_sen)),2)))

        ax[0][1].tick_params(axis='both', which='major', width=1.5, direction='in', left='off', bottom='off')
        # ax[0][1].tick_params(axis='y', which='minor', width=1, direction='in', length=4, left='off')

        ax[0][1].grid(which='major',axis='both',linewidth=1,linestyle='-',color='grey',alpha=0.7)
        ax[0][1].legend(loc=1, fontsize='small')

    # buffer_arr = [mid_y_val for _ in grid_1_sen]
    # buffer_arr = np.array(buffer_arr)

    # ax[0][0].set_xlabel('college_score', labelpad=25, family='monospace', weight='bold')
    # ax[0][0].set_ylabel('avg_sentiment_val', labelpad=25, family='monospace', weight='bold')

    # ax[0][1].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen)-0.1,max(grid_1_sen+grid_2_sen)+0.1,20),minor=True)
    # print min(col_sen), min(grid_1_sen+grid_2_sen)
    if grid_2_sen:
        ax[1][1].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen),max(grid_1_sen+grid_2_sen),10))
        ax[1][1].set_ylim(min(grid_1_sen+grid_2_sen),mid_y_val+0.02)

        ax[1][1].yaxis.tick_right()
        ax[1][1].yaxis.set_label_position("right")

        ax[1][1].yaxis.set_minor_locator(loc_sen)
        # ax[1][0].set_yticks([''+str(mid_y_val)],minor=True)
        rep_y = [''+str(mid_y_val)]
        ax[1][1].set_yticklabels(np.repeat(rep_y,1), size='large', weight='bold', minor=True)

        # ax[0][1].xaxis.set_label_position("top")

        # ax.set_yticklabels(col_sen,size='small')
        # ax[0][0].set_yticklabels([''+str((abs(min(col_sen)) + max(col_sen))/2)],size='small',minor=True,visible=False)
        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[1][1].set_xticks(np.linspace(min(grid_2_col+grid_3_col),max(grid_2_col+grid_3_col),5))
        ax[1][1].set_xlim(median_col-0.02,max(grid_2_col+grid_3_col)+0.02)
        # ax[0][0].set_xticklabels(list(set(overall_college_score)),size='small')

        ax[1][1].scatter(grid_2_col, grid_2_sen, color='black', label = "Lower Sentiment, Higher Overall Score \n No. of Colleges : "+str(count_grid_2_college)+", Avg. Sen : "+str(round(np.mean(np.array(grid_2_sen)),2)))

        ax[1][1].tick_params(axis='both', which='major', width=1.5, direction='in', left='off', top='off')
        ax[1][1].tick_params(axis='y', which='minor', width=1, direction='in', left='off', pad=50)

        ax[1][1].grid(which='major',axis='both',linewidth=1,linestyle='-',color='grey',alpha=0.7)
        ax[1][1].grid(which='minor',axis='y',linewidth=1.5,linestyle='-',color='b')
        ax[1][1].legend(loc=4, fontsize='small')
    # print buffer_arr
    # fill_arr_sen_grid_1 = np.ndarray(shape=(len(grid_1_sen)), dtype=np.float, buffer = np.array(buffer_arr), order='c')
    # # fill_arr_sen_grid_1.fill(mid_y_val)
    # print fill_arr_sen_grid_1
    if grid_1_sen:
        ax[1][0].set_yticks(np.linspace(min(grid_1_sen+grid_2_sen),max(grid_1_sen+grid_2_sen),10))
        ax[1][0].set_ylim(min(grid_1_sen+grid_2_sen),mid_y_val+0.02)

        ax[1][0].yaxis.set_minor_locator(loc_sen)
        # ax[1][0].set_yticks([''+str(mid_y_val)],minor=True)
        rep_y = [''+str(mid_y_val)]
        ax[1][0].set_yticklabels(np.repeat(rep_y,1), weight='bold', size='large', minor=True)
        # ax[0][1].yaxis.tick_right()
        # ax[0][1].yaxis.set_label_position("right")
        # ax[0][1].xaxis.set_label_position("top")

        # ax.set_yticklabels(sentiment_score,size='small')

        # ax.set_xticklabels([''+str(median_col)],size='small',minor=True,visible=True)
        ax[1][0].set_xticks(np.linspace(min(grid_1_col+grid_4_col),max(grid_1_col+grid_4_col),5))
        ax[1][0].set_xlim(min(grid_1_col+grid_4_col)-0.02,median_col+0.02)

        ax[1][0].xaxis.set_minor_locator(loc_col)
        # ax[1][0].xaxis.set_minor_formatter(plticker.FormatStrFormatter('%.2f'))
        # ax[1][0].set_xticks([''+str(median_col)],minor=True)
        rep_x = [(''+str(round(median_col,3)))]
        ax[1][0].set_xticklabels(np.repeat(rep_x,2), weight='bold', size='large', minor=True)

        ax[1][0].scatter(grid_1_col, grid_1_sen, color='red', label = "Lower Sentiment, Lower Aspect Score \n No. of Colleges : "+str(count_grid_1_college)+", Avg. Sen : "+str(round(np.mean(np.array(grid_1_sen)),2)))

        ax[1][0].tick_params(axis='both', which='major', width=1.5, direction='in', right='off', top='off')
        ax[1][0].tick_params(axis='y', which='minor', width=1, direction='in', left='off', pad=50)
        ax[1][0].tick_params(axis='x', which='minor', width=1, direction='in', pad=20, right='off')
        # ax[0][1].tick_params(axis='y', which='minor', width=1, direction='in', length=4, left='off')

        ax[1][0].grid(which='major',axis='both',linewidth=1,linestyle='-',color='grey',alpha=0.7)
        ax[1][0].grid(which='minor',axis='x',linewidth=4,linestyle='-',color='b')
        ax[1][0].grid(which='minor',axis='y',linewidth=1.5,linestyle='-',color='b')
        ax[1][0].legend(loc=3, fontsize='small')

    # popt,pcov = opt.curve_fit(func,valid_show_aspects,valid_sentiment_aspects)
    # # print popt
    # # print xx
    # xx = np.linspace(min(valid_show_aspects),max(valid_sentiment_aspects),1000)
    # yEXP = func(xx, *popt)
    # # loc_sen = plticker.FixedLocator(np.linspace(yEXP[0],yEXP[0],1))
    # # print yEXP
    # #
    # # print dd[0], date_values
    # # ax1.plot(dd,y,'r-',lw=1, alpha=0.9, label='poly fit')
    # ax.plot(xx,yEXP,'b--',lw=1.5, alpha=0.9, label='curve fit')
    # fit = np.polyfit(valid_show_aspects,valid_sentiment_aspects,1)
    # fit_fn = np.poly1d(fit)
    #
    # ax.plot(valid_show_aspects,fit_fn(valid_show_aspects),'--b', antialiased =True, rasterized = True,linewidth=0.8,label="Fit %6s, r = %6.2e"%('RED',pearR))

    # ax.scatter(valid_show_aspects, valid_sentiment_aspects,label='Data Red',color='r')

    # ax.plot(valid_show_aspects,valid_show_aspects*m+c,color='r',label="Fit %6s, r = %6.2e"%('RED',pearR), antialiased =
    #         True, rasterized = True,linewidth=0.8)
    # ax.set_yscale('log')
    # ax.set_yscale('log', basey=1.1, posxy=)
    # ax.set_xscale('log', basex=1.1)

    # plt.scatter(valid_show_aspects, sentiment_score, color='r')


    # ax.set_xticks(np.arange((min(valid_show_aspects)), (max(valid_show_aspects)), 0.2))
    # ax.set_yticks(np.arange((min(valid_sentiment_aspects) - 0.02), (max(valid_sentiment_aspects) + 0.02), 0.1))
    # ax.set_yticklabels(valid_sentiment_aspects,fontdict='monospace',zorder=3)
    # ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=25, prune=None))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=15, prune=None))
    # xticks_arr = np.linspace(min(valid_show_aspects)-0.05,max(valid_show_aspects)+0.05,15)
    # ax.set_xticks(xticks_arr)
    # ax.set_xticklabels(xticks_arr, fontdict='monospace', zorder=1, color='black', rotation='0',
    #                    va='top', ha='center', weight='roman', clip_on=True, axes=ax, size='medium')
    # ax.set_xlim(valid_show_aspects.min()-0.05,valid_show_aspects.max()+0.05)
    #
    # yticks_arr = np.linspace(min(valid_sentiment_aspects)-0.01,max(valid_sentiment_aspects)+0.01,25)
    # ax.set_yticks(yticks_arr)
    # ax.set_yticklabels(yticks_arr, fontdict='monospace', zorder=1, color='black', rotation='0',
    #                    va='center', ha='right', weight='roman', clip_on=True, axes=ax, size='medium')
    # ax.set_ylim(bottom=min(valid_sentiment_aspects) - 0.01, top=max(valid_sentiment_aspects) + 0.01, emit=True)
    #
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%.3f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    #
    # ax.tick_params(axis='both', which='major', direction='in', right='off', top='off', width=2, pad=5)
    # ax.set_yscale('log',basey=1.1)
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
    #     plt.text(rect.get_x() + rect.get_width(), height + 0.18, '(%s,%s)' % (sentiment_score[i], valid_show_aspects[i]),
    #              ha='center', va='top', rotation='vertical')

    # plt.tight_layout()
    # plt.grid()
    # ax.grid(which='major', color='grey', linestyle='-.', linewidth=0.5,alpha=0.9)
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

    # plt.grid()
    text = "r = " + str(pearR)
    f.text(0.01,0.93,text,fontsize=12,bbox=dict(facecolor='grey',alpha=0.5,pad=7.0,edgecolor='red'),weight='bold',color='k')

    plt.show()
    img_name = "Colleges_Correlation_between_"+aspects_value[index].capitalize()+"_&_College_tweets_Sentiment.svg"
    f.savefig('./sentiment_aspects_score_correlation/'+img_name, dpi=600)
    plt.close(f)
    index+=1


