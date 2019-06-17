# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:43:06 2019

@author: jingwenken
"""

#%%
'''
Seaborn heatmap calendar graph
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
import numpy as np
import datetime as dt
booksv2 = pd.read_csv('book1.csv')

booksv2['Start'] = booksv2['Start'].apply(lambda x: dt.datetime.strptime(str(x),'%d/%m/%Y'))
booksv2['End'] = booksv2['End'].apply(lambda x: dt.datetime.strptime(str(x),'%d/%m/%Y'))
min_date = min(list(booksv2['Start'])+list(booksv2['End']))
max_date = max(list(booksv2['Start'])+list(booksv2['End']))
#year = min_date.year
year = 2018

def date_generator(from_date,to_date):
    while from_date<=to_date:
        yield from_date
        from_date = from_date + dt.timedelta(days=1)
        
# create a new df with 2 columns
# col1 : Title, col2: DateRead
Title = []
Date = []
for index,row in booksv2.iterrows():
    date_temp = list(date_generator(row['Start'],row['End']))
    title_temp = [row['Title']]*len(date_temp)
    print(date_temp,title_temp)
    Title += title_temp
    Date += date_temp

booksv2 = pd.DataFrame(np.array([Title,Date]),index=['Title','Date']).T

# create a df with date from 1st to last day of year from min_year
dateList = pd.DataFrame(list(date_generator(dt.datetime(year,1,1,0,0,0),dt.datetime(year,12,31,0,0,0))),columns=['Date'])
dateList.Date = dateList.Date.astype('O')

# merge the 2 dataframes
bookDf = pd.merge(dateList,booksv2,how='left',on='Date')
bookDf.fillna('nil')

# Adding needed columns
bookDf['DateOrig'] = bookDf['Date']
bookDf['Day'] = bookDf['Date'].apply(lambda x: x.day)
bookDf['Month'] = bookDf['Date'].apply(lambda x: dt.datetime.strftime(x,'%b'))
bookDf['DOW'] = bookDf['Date'].apply(lambda x: dt.datetime.strftime(x,'%a'))
bookDf['Month_num'] = bookDf['Date'].apply(lambda x: x.month)
bookDf['DOW_num'] = bookDf['Date'].apply(lambda x: x.weekday())
bookDf['Week_num'] = bookDf['Date'].apply(lambda x: int(dt.datetime.strftime(x,'%W')))

#add proxy for different colours
bookDf['color_proxy'] = bookDf[['Month_num','Title']].apply(lambda x: x[0] if str(x[1])!='nan' else 0,axis=1)
# columnmn for Y labels
bookDf['y_ticklabels'] = bookDf[['Day','Month']].apply(lambda x: x[1] if x[0]==1 else '',axis=1)
y_ticklabels = bookDf[['y_ticklabels','Week_num']].drop_duplicates()

y_withLabels = list(y_ticklabels[y_ticklabels['y_ticklabels']!='']['Week_num'])
y_ticklabels = y_ticklabels[(y_ticklabels['y_ticklabels']!='') | (y_ticklabels['Week_num'].apply(lambda x: int(x) not in list(y_withLabels)))]
y_ticklabels = list(y_ticklabels['y_ticklabels'])

# set colors
cmap=['white','red','orange','yellow','green','blue',
      'indigo','violet','purple','grey','pink',
      'brown','black']

f, ax = plt.subplots(figsize=(6, 18))

# drop duplicates for bookDf **End of book A is the start of book B
df = bookDf.copy()
df.drop_duplicates(['Date'],inplace=True)

df = df.pivot('Week_num','DOW_num','color_proxy')

sns.heatmap(df, linewidths=.5,
            ax=ax,
            cmap=cmap,
            cbar=False,
            vmin = 0,
            vmax = 12,
            square = True,
            yticklabels = y_ticklabels)

ax.set_yticklabels(ax.get_yticklabels(),rotation=0)