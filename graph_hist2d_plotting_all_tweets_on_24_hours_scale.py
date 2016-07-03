# from multiprocessing import Pool
# import random
import time
# import unicodedata
import sys
from matplotlib.backends.backend_pdf import PdfPages
import MySQLdb
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as clr
# from matplotlib import cm
import datetime
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# import re

# ckey = 'ZuOG4EnYSaytbMJrjujkE4m0G'
# csecret = '5kHVA3iQKUJ2xEwWi3OfnDnXePgr5qUrEoY8PljYChoOLARp5O'
# atoken = '99918081-CzzDcnKJPO8MR2OIcpHDEHoY9ti7wGAggcME3ke4h'
# asecret = 'NhyIvJLTygJwpHL9r0Ic2Gu2sPDl44dDvYtK5xfxMXWLS'

# alchemy_api_keys = ['99ffde4abbe13ff7d46577fb1b32b989dcaba79f', 'c4d7b3ef2a002a3067f342dcdcd375e529ccb6e5',
#                     '9c12219c575293afaf896ee23c40f077e71bb9bb', '14f4bb53b16e90d0c7f917aad0941e43516ee84f']

cnxn1 = MySQLdb.connect(host="163.118.78.202", user="root", passwd="harshit@123", db="final_twitter_data", charset="utf8", use_unicode=True)
# cnxn1.set_character_set('utf8mb4')
# cnxn1.character_set_name('utf8mb4')
cursor1 = cnxn1.cursor()
cnxn1.autocommit(True)


# cnxn = MySQLdb.connect(host="localhost", user="root", passwd="harshit@123", db="college_info", charset="utf8",
#                        use_unicode=True)
# # cursor = cnxn.cursor()
# # cnxn.autocommit(True)
#
# query00 = "select * from time_conversion_scale"
# cursor = cnxn.cursor()
# cursor.execute(query00)
# time_check_data = cursor.fetchall()
#
# cursor.close()
# cnxn.close()
#
# query1 = "set names 'utf8mb4'"
# query2 = "SET CHARACTER SET utf8mb4"

# making database accept utf8mb4 as the data format in their columns
# cursor1.execute(query1)
# cursor1.execute(query2)

# filterwarnings('ignore')

# various api links which would be called form requests library
# alchemy_url = "http://gateway-a.watsonplatform.net/calls/text/TextGetTextSentiment"
# # google_get_local_time = "https://maps.googleapis.com/maps/api/timezone/json"
#
# # error_sentiment = open('error_in_getting_sentiment.txt', 'a')
# convert_count = open('count_of_conversion_for_each_college.txt', 'a')


def func1(x, a, b, c):
    return np.log(c + x) - np.log(a + (b * x))


def tables_calculate(count):
    try:
        query0 = """select table_name from information_schema.tables where table_schema = 'final_twitter_data' and table_name like '%_tweets' order by table_name limit {},{}""".format(count, 200)
        # print query0
        cursor1.execute(query0)
        table_data = cursor1.fetchall()
        return table_data
    except Exception, e:
        print "Error while getting college names ", e


def adding_sentiment_time():
    count = 0

    # str = '00:00:00'
    # val.append(str)
    time_num = []
    # val_time = []
    for j in range(3):
        if j == 2:
            ex = 4
        else:
            ex = 10
        for i in range(ex):
            val = str(j) + "" + str(i) + ":00:00"
            # val_time.append(val)
            time_num.append(time.mktime(time.strptime("2015-01-01 " + val, "%Y-%m-%d %H:%M:%S")))
        time_num.append(time.mktime(time.strptime("2015-01-01 23:59:59", "%Y-%m-%d %H:%M:%S")))

    time_arr_quarter = np.linspace(time_num[0], time_num[-1], (24 * 4) + 1)

    # print time_arr_quarter, len(time_arr_quarter)

    val_time = []
    for item in time_arr_quarter:
        val_time.append(datetime.datetime.fromtimestamp(int(item)).strftime('%H:%M:%S'))

    # print val_time
    # ax1.tick_params(axis='y', which='minor', direction='out', length=3, width=0.5, labelsize=0, colors='blue')
    # ax1.tick_params(axis='x', which='both', direction='out', length=6, width=0.5, labelsize=8, colors='black', top='off')

    try:
        # current_alchemy_api_key = '4c31dc7176e538e5f2ebc3d662e728270e30efa4'
        # final = previous + 10
        table_data = tables_calculate(count)
        # print table_data
        # plt.ion()

        all_sen = []
        all_time = []
        for row1 in table_data:
            # print row1[0]
            time_arr = []
            date_arr = []
            sen_arr = []
            # col_arr = []
            # print row1[0]
            query = """select local_tweet_date,local_tweet_time,sentiment_score from """ + row1[0] + """ order by id"""
            cursor1.execute(query)
            conversion_data = cursor1.fetchall()
            # print "\n\nNew table started : -> ", row1[0], "\n\n"
            # count += 1
            i = 0
            for row in conversion_data:
                i += 1
                if i < 10000:
                    if row[1] not in (None, 'None', 'NULL', '', ' ') and row[2] not in (None, 'None', 'NULL', '', ' '):
                        # print dt.date2num(datetime.datetime.strptime(str(row[1]),"%H:%M:%S"))
                        # print time.mktime(time.strptime("2015-01-01 "+str(row[1]),"%Y-%m-%d %H:%M:%S"))
                        time_arr.append(time.mktime(time.strptime("2015-01-01 " + str(row[1]), "%Y-%m-%d %H:%M:%S")))
                        date_arr.append(row[0])
                        sen_arr.append(row[2])
                        # if row[2] > 0.0:
                        #     col_arr.append('blue')
                        # elif row[2] < 0.0:
                        #     col_arr.append('r')
                        # else:
                        #     col_arr.append('grey')

            # ax1.xaxis.set_major_locator(MaxNLocator(nbins=len(date_values), prune=None))
            # print time_arr
            # print sen_arr

            # ax1.plot(time_arr, sen_arr, marker='o', color='b', linestyle='', markersize=4.0, rasterized=True, antialiased=True, zorder=1)
            # print time_arr
            all_sen.extend(sen_arr)
            all_time.extend(time_arr)

            # print "success"
            # ax1.draw(plot1)
            # f.canvas.draw()
            # plt.draw()

        # H, xedges, yedges, img = ax1.hist(all_time, all_sen, bins=[time_num, np.arange(min(all_sen), max(all_sen), 0.1)], range=[[min(time_num), max(time_num)], [min(all_sen), max(all_sen)]], normed=True)

        # print time_num

        # print min(all_sen), max(all_sen)
        all_time = np.array(all_time)
        all_sen = np.array(all_sen)

        yaxis_ticks = np.linspace(np.amin(all_sen), np.amax(all_sen), 90)

        H, xedges, yedges = np.histogram2d(all_time, all_sen, bins=(time_arr_quarter, yaxis_ticks),
                                           range=[[np.amin(time_arr_quarter), np.amax(time_arr_quarter)], [np.amin(all_sen), np.amax(all_sen)]],
                                           normed=False)

        # print xedges, yedges
        # print H.T
        min_val = 100000.
        max_val = -1.
        for item in H:
            if np.amin(item) < min_val:
                min_val = np.amin(item)
            if np.amax(item) > max_val:
                max_val = np.amax(item)

        # print min_val, max_val

        f, ax1 = plt.subplots(1, 1, figsize=(100, 200), dpi=96)
        plt.subplots_adjust(bottom=0.12)
        f.set_size_inches(100, 200, forward=True)
        # ax1, ax2 = ax[0, 0], ax[0, 1]
        # ax1.spines['left'].set_color('blue')
        # for axes in ax:
        #     ax1=
        # ax1,ax2 = ax

        # print len(sen_values), " ",len(count_values), " ", len(date_values)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # jet = plt.get_cmap('Reds')
        # for i in range(jet)
        # cmap1 = cmx.Blues
        cmap = cmx.RdBu_r
        # extract all colors from the .jet map
        # cmaplist1 = [cmap1(i) for i in range(cmap1.N)]
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be grey
        # print len(cmaplist)
        cmaplist = cmaplist[-240::-1] + cmaplist[240:]
        # cmaplist1 = cmaplist1[127:]
        # list_con = cmaplist1 + cmaplist
        # print len(list_con)
        # create the new map
        # print cmap.N
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        print cmap.N
        bounds = np.linspace(min_val, max_val, cmap.N+1, endpoint=True)
        print len(bounds)
        norm = clr.BoundaryNorm(bounds, cmap.N)

        mapp = ax1.imshow(H.T, interpolation='none', aspect='auto', cmap=cmap, origin='lower', extent=extent, norm=norm)
        #
        plt.colorbar(mapp, orientation='vertical', format='%d', fraction=0.05, pad=0.07, spacing='proportional',
                     ticks=bounds, boundaries=bounds)

        ax1.set_title('Tweets Plotting with Sentiment over 24 hour Period', y=1.02, style='italic', family='monospace', weight='bold', size='large')

        ax1.set_xlabel('Time', labelpad=3, family='monospace', weight='bold', size='small')
        ax1.set_ylabel('Tweets_Sentiment', labelpad=10, family='monospace', weight='bold', size='small')

        ax1.set_xticks(time_arr_quarter)

        ax1.set_xticklabels(val_time, zorder=1, color='black', rotation='vertical',
                            va='top', ha='center', weight='roman', clip_on=True, axes=ax1, size='x-small')
        ax1.set_xlim(time_arr_quarter[0], time_arr_quarter[-1])

        # minor_ticks = np.linspace(min(sen_values) - 0.1,max(sen_values) + 0.1,40)
        ax1.set_yticks(yaxis_ticks)
        # ax1.set_yticks(minor_ticks,minor=True)

        ax1.set_yticklabels(yaxis_ticks, zorder=1, color='blue', rotation='0',
                            va='center', ha='right', weight='roman', clip_on=True, axes=ax1, size='xx-small')
        ax1.set_ylim(np.amin(all_sen), np.amax(all_sen))

        ax1.tick_params(axis='both', which='major', direction='out', pad=5, length=4, width=0.5, colors='k', top='off', right='off')
        # ax1.tick_params(axis='x', which='major', direction='in', pad=5, length=4, width=2, colors='k', top='off')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
        # ax1.grid(which='minor', color='grey', linestyle='-', linewidth=1, alpha=0.4)

        ax2 = ax1.twinx()
        # f.canvas.draw()
        # ax1.plot()

        ax2.set_yticks(yaxis_ticks)
        # plt.yticks(new_tick_locations,linespacing=0.5)
        # ax1.yaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax2.set_yticklabels(yaxis_ticks, zorder=1, color='k', rotation='0',
                            va='center', ha='left', weight='normal', clip_on=True, size='xx-small')
        # print college_name, len(college_name)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.tick_params(axis='y', which='major', direction='out', pad=5, length=4, width=0.5, colors='k', left='off', labelleft='off')
        ax2.set_ylim(np.amin(all_sen), np.amax(all_sen))

        # ax1.autoscale(enable=True,axis='y',tight=True)
        # ax1.autoscale_view(tight=True,scaley=True)

        # ax3 = ax1.twiny()
        # # f.canvas.draw()
        # # ax2.plot()
        # # new_tick_locations1 = np.arange(0,len(college_name),1)
        # ax3.set_xticks(time_arr_quarter)
        # # ax2.xaxis.set_major_locator(MaxNLocator(prune='lower'))
        # # demi_college_name = college_name
        # # # col_len = len(college_name)
        # # # print col_len
        # # demi_college_name = [item.encode('ascii', 'ignore') for item in demi_college_name]
        # # ax2.set_xticks(demi_college_name)
        # # ax1.set_yticks(demi_college_name)
        # # ax2.xaxis.set_major_locator(MaxNLocator(nbins=len(college_name), prune=None))
        #
        # ax3.set_xticklabels(val_time, zorder=1, color='k', rotation='vertical',
        #                     va='bottom', ha='center', weight='normal', clip_on=True, size='x-small')
        #
        # # ax.set_yticklabels(college_name, fontdict='monospace', minor = True, zorder=1, color='blue', rotation='20',
        # #                    va='top', ha='center', weight='roman', clip_on=True, axes=ax)
        #
        # ax3.tick_params(axis='x', which='major', direction='out', pad=5, length=4, width=0.5, colors='k', bottom='off', labelbottom='off')
        # ax3.set_xlim(time_arr_quarter[0], time_arr_quarter[-1])

        # ax2.pcolormesh(len(time_num), len(np.arange(min(all_sen), max(all_sen), 0.1)),H)

        img_name = "Tweets Plotting with Sentiment over 24 hour Period.svg"
        # index +=1
        # ax.view_init(elev='10', azim='20')
        # lns = plot1
        # j=0
        # labs = [l.get_label() for l in lns if j < 1]
        # pcoff = np.ma.corrcoef(np.array(all_time), np.array(all_sen))[1, 0]
        # print min(all_time), max(all_time)
        # all_time = [item / 100 for item in all_time]
        # x_new = np.linspace(min(all_time), max(all_time), 200000)
        # # print len(all_sen),len(all_time)
        # try:
        #     popt1, pcov1 = opt.curve_fit(func1, all_time, all_sen, p0=(1, 1e-200, 1))
        #     yLOG = func1(x_new, *popt1)
        #     x_new = [item * 100 for item in x_new]
        #     plot00 = ax1.plot(x_new, yLOG, 'g--', lw=5, alpha=0.9, label='Curve Log Fit', zorder=1)
        #     # print yLOG
        # except RuntimeError:
        #     pass
        # ax1.plot([], [], marker='o', color='b', linestyle='', label='Tweets Sentiment')
        # # ax1.plot([],[],marker='o',color='r',linestyle='',label='Negative')
        # # ax1.plot([],[],marker='o',color='grey',linestyle='',label='Neutral')
        # # lns = plot1 + plot00
        # # labs = [l.get_label() for l in lns]
        # ax1.legend(loc='upper center')
        # lns = [for lns in plot]
        # ax1.legend(loc='upper center')
        ax1.autoscale(tight=True)
        # plt.tight_layout()
        plt.show()
        # ax1.legend(loc='upper left')
        # text = "Pearson Correlation Coefficient : " + str(round(pcoff, 3))
        # f.text(0.01, 0.95, text, fontsize=14, bbox=dict(facecolor='grey', alpha=0.5, pad=7.0), weight='bold')

        f.savefig(img_name, format='svg', dpi=600, bbox_inches='tight')
        pp = PdfPages('Tweets Plotting with Sentiment over 24 hour Period.pdf')
        pp.savefig(f)
        # pp.savefig(plot2)
        # pp.savefig(plot3)
        pp.close()
        plt.close(f)

    except Exception, err:
        print "Error while calling query : ", err
        sys.exit(1)


if __name__ == "__main__":
    # pool = Pool(processes=4)
    # pool.map(adding_sentiment_time, range(250, 1550, 100))
    # pool.close()
    # pool.join()
    adding_sentiment_time()
