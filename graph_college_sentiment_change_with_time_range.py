import MySQLdb
import datetime
import numpy as np
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# from scipy.stats import expon
import scipy.optimize as opt
from scipy.interpolate import interp1d

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8", use_unicode=True)
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)

cnxn2 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="college_keywords_by_time", charset="utf8", use_unicode=True)
cursor2 = cnxn2.cursor()
cnxn2.autocommit(True)

query0 = """select * from college_sentiment_average where no_rows >= 200"""
cursor1.execute(query0)
fetch_col = cursor1.fetchall()


# def func(x,a,b,c):
#     # print np.exp(-x)
#     return a*np.exp(b*x)+c


def func1(x, a, b, c):
    return np.log(c + x) - np.log(a + (b * x))

for row in fetch_col:

    sen_values = []
    count_values = []
    date_values = []
    overall_sentiment = 0.0
    overall_tweets = 0

    try:
        query1 = "select * from college_sentiment_average_per_day where college_name in (%s) order by id"
        cursor1.execute(query1, [str(row[1])])
        for i, data in enumerate(cursor1):
            if data[3] == 0:
                continue
            else:
                if data[2] is None:
                    print "why?"
                    overall_sentiment = ((overall_sentiment * overall_tweets) + (data[3] * 0.0)) / (overall_tweets + data[3])
                    sen_values.append(overall_sentiment)
                else:
                    overall_sentiment = ((overall_sentiment * overall_tweets) + (data[3] * data[2])) / (overall_tweets + data[3])
                    sen_values.append(overall_sentiment)
                overall_tweets += data[3]
                count_values.append(overall_tweets)
                date_values.append(data[4])

        if len(date_values) < 10:
            continue

        f, ax1 = plt.subplots(1, 1, figsize=(1600 / 95.9, 900 / 95.9), dpi=96)
        plt.subplots_adjust(bottom=0.11)
        # loc_sen = plticker.FixedLocator(np.linspace(0.0,0.0,1))

        # t_mean = np.mean(sen_values)
        # t_std = np.std(sen_values)
        #
        # mean,var,skew,kurt = expon.stats(moments='mvsk')
        # x = np.linspace(expon.ppf(min(sen_values)),expon.ppf(max(sen_values)),len(date_values))
        # ax1.plot(x, expon.pdf(x),'r-', lw=5, alpha=0.6, label='expon pdf')

        # ax1[0].spines['bottom'].set_visible(False)
        # ax1[0].spines['top'].set_visible(False)

        # print len(sen_values), " ",len(count_values), " ", len(date_values)
        ax1.set_title('Plot_of_' + str(row[1]) + '_with_Accumulated_Tweets_Sentiment_till_the_Day', y=1.02, style='italic', family='monospace',
                      weight='bold', size='large')

        ax1.set_xlabel('Date', labelpad=4, family='monospace', weight='bold', size='medium')
        ax1.set_ylabel('Accumulated_Tweets_Sentiment_till_day', labelpad=10, family='monospace', weight='bold', size='medium', color='black')

        # print xx
        dates_conv = mdates.date2num(date_values)
        # dates_conv1 = [float((i-date_values[0]).days)+1.0 for i in date_values]
        # print dates_conv
        # print dates_conv1
        # print np.exp(-dates_conv)
        # for i in range(10):
        #     start = np.random.uniform(-10, 10, size=4)
        #     # Get parameters estimate
        #     try:
        #         popt2, pcov2 = curve_fit(func2, xdata, ydata, p0=start)
        #     except RuntimeError:
        #         continue
        #     err = ((ydata - func2(xdata, *popt2))**2).sum()
        #     if err < err_last:
        #         err_last = err
        #         print err
        #         best = popt2
        # print dates_conv
        # print dates_conv
        # trialX = np.linspace(xData[0], xData[-1], 1000)
        #
        # z = np.polyfit(dates_conv,sen_values,2)
        # # # print z
        # poln = np.poly1d(z)
        # y = np.zeros(len(xx))
        # for i in range(len(z)):
        #    y += z[i]*xx**i

        # # # print dd
        # #
        # # # ax1.yaxis.set_minor_locator(loc_sen)
        # #
        # # # comp_array = np.arange(len(date_values))

        # print popt
        # xx = np.linspace(dates_conv[0]-10, dates_conv[-1]+10, 1000) #int((date_values[0]-date_values[-1]).days))
        # dd = mdates.num2date(xx)
        # print xx

        # start = np.random.uniform(-10, 10, size=3)
        # print start[0],start[1],start[2]
        # popt, pcov = opt.curve_fit(func1, dates_conv1, sen_values, p0=(1,0.5,1))
        # yEXP = func1(dates_conv, *popt)
        # print popt
        # print yEXP
        # # loc_sen = plticker.FixedLocator(np.linspace(yEXP[0],yEXP[0],1))
        # # print yEXP
        # x_conv = [item.date() for item in dd]
        #
        # print dd[0], date_values
        # popt1, pcov1 = opt.curve_fit(func1, dates_conv1, sen_values, p0=(1,-1.0,1))
        # yEXP1 = func1(dates_conv1, *popt1)
        # print popt1
        # print yEXP1

        # ax1.plot(dates_conv,yEXP,'b-',lw=1, alpha=0.9, label='expon curve fit')
        # ax1.plot(dates_conv1,yEXP1,'r-',lw=1, alpha=0.9, label='expon curve fit1')
        # ax1.plot(date_values,poln(dates_conv),'b--',lw=1.5, alpha=0.7, label='Poly Fit')
        # ax1.yaxis.set_minor_locator(loc_sen)

        # rep_y = [''+str(yEXP[0])]
        # ax1.set_yticklabels(np.repeat(rep_y,1), weight='bold', size='large', minor=True)
        func_interp = interp1d(dates_conv, sen_values, kind='cubic')

        # z = np.polyfit(dates_conv,sen_values,3)
        # poln = np.poly1d(z)

        # start = np.random.uniform(-10, 10, size=3)
        x_new = np.arange(dates_conv[0], dates_conv[-1] + 1, 1)
        dd = mdates.num2date(x_new)
        # try:
        #     popt, pcov = opt.curve_fit(func, dates_conv, sen_values, p0=(1,1e-6,1))
        #     yEXP = func(x_new,*popt)
        #     ax1.plot(dd,yEXP,'r-.',lw=5, alpha=0.7, label='Curve Exp Fit')
        #     print yEXP
        # except RuntimeError:
        #     pass
        try:
            popt1, pcov1 = opt.curve_fit(func1, dates_conv, sen_values, p0=(1, 1e-6, 1))
            yLOG = func1(x_new, *popt1)
            plot00 = ax1.plot(dd, yLOG, 'g--', lw=5, alpha=0.9, label='Curve Log Fit', zorder=1)
            # print yLOG
        except RuntimeError:
            pass

        # print x_new
        # dd1 = [datetime.datetime.date(item) for item in dd]
        # print dd1
        plot0 = ax1.plot(dd, func_interp(x_new), 'r-', lw=1, alpha=0.9, label='Interpolation Fit', zorder=1)

        # ax1.plot(dd,poln(x_new),'k-',lw=3, alpha=0.9, label='Poly Fit')
        # print yEXP
        plot_1 = ax1.plot_date(date_values, sen_values, fmt='ro', rasterized=True, antialiased=True, label='Accumulated Tweets Sentiment per Range',
                               zorder=1)

        yaxis_ticks = np.linspace(min(sen_values) - 0.01, max(sen_values) + 0.01, 20)
        # minor_ticks = np.linspace(min(sen_values) - 0.1,max(sen_values) + 0.1,40)
        ax1.set_yticks(yaxis_ticks)
        # ax1.set_yticks(minor_ticks,minor=True)

        ax1.set_yticklabels(yaxis_ticks, fontdict='monospace', zorder=1, color='black', rotation='0',
                            va='center', ha='right', weight='roman', clip_on=True, axes=ax1, size='medium')
        ax1.set_ylim(min(sen_values) - 0.01, max(sen_values) + 0.01)

        # rep_y = [(''+str(0.0))]
        # ax1.set_yticklabels(np.repeat(rep_y,1), weight='bold', size='large', minor=True)

        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # ax1.yaxis.set_minor_formatter(FormatStrFormatter('%.3f'))

        ax1.set_xticks(date_values)
        ax1.set_xticklabels(date_values, fontdict='monospace', zorder=1, color='black', rotation='vertical',
                            va='top', ha='left', weight='roman', clip_on=True, axes=ax1, size='medium')
        ax1.set_xlim(date_values[0] - datetime.timedelta(days=1), date_values[-1] + datetime.timedelta(days=1))

        ax1.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, right='off')
        # ax1.tick_params(axis='y', which='minor', direction='out', length=8, width=2, pad=50)
        # ax1.tick_params(axis='y', which='minor', direction='out', length=3, width=0.5, labelsize=0, colors='blue')
        ax1.tick_params(axis='x', which='major', direction='out', length=6, width=0.5, labelsize=8, top='off')

        # print date_values, sen_values, count_values

        ax1.grid(which='major', color='grey', linestyle='--', linewidth=1, alpha=0.8)
        # ax1.grid(which='minor',axis='y',linewidth=1.5,linestyle='-',color='b')
        # ax1.grid(which='minor', axis='y', color='blue', linestyle='-', linewidth=2,alpha=0.7)

        ax2 = ax1.twinx()

        ax2.set_ylabel('No._of_Tweets', labelpad=10, family='monospace', weight='bold', size='medium', color='r')

        ax2.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, colors='red', left='off')
        # ax2.tick_params(axis='y', which='minor', direction='in', length=2, width=0.5, colors='red')
        # ax2.spines['right'].set_color('red')
        yaxis_ticks_1 = np.linspace(min(count_values), max(count_values), 20)
        # ticks_count =
        # minor_ticks_1 = np.linspace(min(count_values) - 1,max(count_values) + 5,60,dtype='int32')
        ax2.set_yticks(yaxis_ticks_1)
        ax2.spines['right'].set_color('red')

        # ax2.set_yticks(minor_ticks_1,minor=True)
        ax2.set_yticklabels(yaxis_ticks_1, fontdict='monospace', zorder=1, color='red', rotation='0',
                            va='center', ha='left', weight='roman', clip_on=True, axes=ax2, size='medium')
        ax2.set_ylim(min(count_values) - 5, max(count_values) + 5)

        plot_2 = ax2.plot_date(date_values, count_values, color='b', marker='o', linestyle='-', linewidth=1, rasterized=True, antialiased=True,
                               label='Tweets_Count', zorder=1)
        ax2.set_xlim(date_values[0] - datetime.timedelta(days=1), date_values[-1] + datetime.timedelta(days=1))

        ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        for j, item in enumerate(sen_values):
            if j == 0:
                continue
            else:
                if item - sen_values[j - 1] >= (max(sen_values) - min(sen_values)) / 4.0:
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
                    if i == 0:
                        for key in cursor2:
                            if i <= 5:
                                text += key[1].title() + " / " + str(key[2]) + " / " + str(key[3]) + "\n"
                                # print date_values[j]
                                i += 1
                    ax1.annotate(text[:text.rfind('\n')], xy=(date_values[j], sen_values[j]), xycoords='data', xytext=(-20, 20),
                                 textcoords='offset points', weight='normal', fontsize='small', ha='right', va='center',
                                 bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.6), arrowprops=dict(facecolor='black', shrink=0.05), color='b',
                                 zorder=0)

                elif item - sen_values[j - 1] <= -((max(sen_values) - min(sen_values)) / 4.0):
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
                    if i == 0:
                        for key in cursor2:
                            if i <= 5:
                                text += key[1].title() + " / " + str(key[2]) + " / " + str(key[3]) + "\n"
                                # print date_values[j]
                                i += 1
                    ax1.annotate(text[:text.rfind('\n')], xy=(date_values[j], sen_values[j]), xycoords='data', xytext=(-20, 20),
                                 textcoords='offset points', weight='normal', fontsize='small', ha='right', va='center',
                                 bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.6), arrowprops=dict(facecolor='black', shrink=0.05), color='b',
                                 zorder=0)
        # popt = ()


        # ax2 = ax1.twiny()
        #
        # ax2.set_ylabel('No._of_Tweets', labelpad=5, family='monospace', weight='bold', size='medium',color='r')
        #
        # ax2.tick_params(axis='y', which='major', direction='out', length=6, width=0.5, colors='red')
        # # ax2.tick_params(axis='y', which='minor', direction='in', length=2, width=0.5, colors='red')
        # # ax2.spines['right'].set_color('red')
        # yaxis_ticks_1 = np.linspace(min(count_values) - 1,max(count_values) + 5,20)
        # # minor_ticks_1 = np.linspace(min(count_values) - 1,max(count_values) + 5,60,dtype='int32')
        # ax2.set_yticks(yaxis_ticks_1)
        #
        # # ax2.set_yticks(minor_ticks_1,minor=True)
        # ax2.set_yticklabels(yaxis_ticks_1, fontdict='monospace', zorder=1, color='red', rotation='0',
        #                    va='center', ha='left', weight='roman', clip_on=True, axes=ax2, size='medium')
        # ax2.set_ylim(min(count_values) - 1, max(count_values) + 5)
        #
        # plot_2 = ax2.plot_date(date_values,count_values,color='r',marker='o',linestyle='-',rasterized=True,antialiased=True,label='Tweets_Count')
        #
        # ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        #
        # lns = plot_1 + plot_2
        # labs = [l.get_label() for l in lns]
        # for i,val in enumerate(date_values):
        #     ax1.annotate("Tweets Count : " + str(count_values[i]), xy=(val,sen_values[i]), xytext=(5, 20), textcoords='offset points', weight='normal', fontsize='small', ha='center',va='center', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), color='b')
        lns = plot00 + plot0 + plot_1 + plot_2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')
        # ax1.legend(loc='best')

        plt.show()
        # ax.grid()
        img_name = 'annotated_Plot_of_' + str(row[1]) + '_with_Accumulated_Tweets_Sentiment_till_the_Day.svg'
        f.savefig('./test_all_college_sentiment_amalgamate_with_time/' + img_name, format='svg', dpi=600)
        plt.close(f)
        # f.close()
        # ax1.close()

    except Exception, e:
        print "Error in plotting : ", e
        sys.exit(1)
