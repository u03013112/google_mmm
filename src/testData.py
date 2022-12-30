# 尝试将一些自己的数据放进来测试，看看能够得到什么样的结果
# 先测试按天的，从5月到11月

import numpy as np
import pandas as pd


from odps import ODPS
import sys
sys.path.append('/src')

from src.config import accessId,secretAccessKey,defaultProject,endPoint
def execSql(sql):
    o = ODPS(accessId, secretAccessKey, defaultProject,
            endpoint=endPoint)
    with o.execute_sql(sql).open_reader(tunnel=True) as reader:
        pd_df = reader.to_pandas()
        return pd_df

def getDataFromODPS():
    sql = '''
        select
            install_day,
            mediasource,
            sum(kpi_impressions) as impressions,
            sum(kpi_clicks) as clicks,
            sum(kpi_installs) as installs,
            sum(cost_value_usd) as cost
            from dwd_base_summary_cost
        where
            day >= 20220501 and day < 20221201
            and app_package = 'id1479198816'
        group by
            install_day,
            mediasource
        ;
    '''

    print(sql)
    # return
    pd_df = execSql(sql)
    # 在这里简单过滤一下安装日期，因为获取的内容全部应该是当天就完整，所以不用多获取
    df = pd_df.loc[(pd_df.install_day >= '2022-05-01') & (pd_df.install_day < '2022-12-01')]
    return df

# 媒体数据处理步骤1
def mediaDataStep1(df):
    # 按照主要媒体，做分类，非主要媒体暂时忽略
    mediaList = [
        {'name':'google','codeList':['googleadwords_int']},
        {'name':'applovin','codeList':['applovin_int']},
        {'name':'bytedance','codeList':['bytedanceglobal_int']},
        {'name':'unity','codeList':['unityads_int']},
        {'name':'apple','codeList':['Apple Search Ads']},
        {'name':'facebook','codeList':['Facebook Ads']},
        {'name':'snapchat','codeList':['snapchat_int']},
    ]
    df.insert(df.shape[1],'media_group','unknown')
    for media in mediaList:
        name = media['name']
        for code in media['codeList']:
            df.loc[df.mediasource == code,'media_group'] = name
    
    # 然后按照媒体与日期做汇总，将缺少的部分补充
    install_day_list = df['install_day'].unique()
    for install_day in install_day_list:
        # print(install_day)
        dfInstallDay = df.loc[(df.install_day == install_day)]
        dataNeedAppend = {
            'install_day':[],
            'impressions':[],
            'clicks':[],
            'installs':[],
            'cost':[],
            'media_group':[]
        }
        
        for media in mediaList:
            name = media['name']
            dfFind = dfInstallDay.loc[(dfInstallDay.media_group == name)]
            if len(dfFind) == 0:
                dataNeedAppend['install_day'].append(install_day)
                dataNeedAppend['impressions'].append(0)
                dataNeedAppend['clicks'].append(0)
                dataNeedAppend['installs'].append(0)
                dataNeedAppend['cost'].append(0)
                dataNeedAppend['media_group'].append(name)

        if len(dataNeedAppend['install_day']) > 0:
            print('补充数据：',install_day,len(dataNeedAppend['install_day']))
            df = df.append(pd.DataFrame(data=dataNeedAppend))
    return df

def getMediaData(df):
    # 按顺序是如下8个媒体：apple,applovin,bytedance,facebook,google,snapchat,unity,unknown
    df = df.loc[:,~df.columns.str.match('Unnamed')]
    df = df.sort_values(by=['install_day','media_group'])
    df = df.groupby(['install_day','media_group']).agg('sum')
    
    # print(df)
    # df.to_csv('/src/data/data2.csv')
    npArray = np.array(df['cost']).reshape((-1,8))
    # print(npArray)
    
    # 第8个是unknown，不要，只要前7个，
    npArray = npArray[:,:-1]
    # print(npArray.shape)
    return npArray

def getMediaCost(df):
    df = df.loc[:,~df.columns.str.match('Unnamed')]
    df = df.sort_values(by=['install_day','media_group'])
    df = df.groupby(['install_day','media_group']).agg('sum')
    
    # print(df)
    # df.to_csv('/src/data/data2.csv')
    npArray = np.array(df['cost']).reshape((-1,8))
    # print(npArray)
    
    # 第8个是unknown，不要，只要前7个，
    npArray = npArray[:,:-1]
    # print(npArray.shape)
    return np.sum(npArray, axis=0)

def getRevenueFromODPS():
    sql='''
        select
            install_date,
            sum(if(life_cycle <= 0, revenue_value_usd, 0)) as r1usd,
            sum(if(life_cycle <= 6, revenue_value_usd, 0)) as r7usd
        from(
            select
                game_uid as uid,
                to_char(
                    to_date(install_day, "yyyymmdd"),
                    "yyyy-mm-dd"
                ) as install_date,
                revenue_value_usd,
                DATEDIFF(
                    to_date(day, 'yyyymmdd'),
                    to_date(install_day, 'yyyymmdd'),
                    'dd'
                ) as life_cycle
            from
                dwd_base_event_purchase_afattribution
            where
                app_package = "id1479198816"
                and app = 102
                and zone = 0
                and window_cycle = 9999
                and install_day >= 20220501
                and install_day < 20221201
        )
        group by
            install_date         
        ;
    '''
    print(sql)
    pd_df = execSql(sql)
    df = pd_df.sort_values(by = 'install_date',ignore_index=True)
    return df
    

# df = getDataFromODPS()
# df.to_csv('/src/data/data0.csv')
# df = pd.read_csv('/src/data/data0.csv')
# df = mediaDataStep1(df)
# df.to_csv('/src/data/data1.csv')

df = pd.read_csv('/src/data/data1.csv')
mediaData = getMediaData(df)

mediaCost = getMediaCost(df)
# df = getRevenueFromODPS()
# df.to_csv('/src/data/data3.csv')
revenueDf = pd.read_csv('/src/data/data3.csv')


print(mediaData.shape,revenueDf['r1usd'].shape,mediaCost.shape)