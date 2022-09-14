import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from zipfile import ZipFile

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

import warnings
warnings.filterwarnings("ignore")
# -------------------------------------------------------------------


BIDS_FILE       = 'data/android_bids_us.csv'
APPS_ZIP_FILE   = 'data/play_apps.zip'


def process_zip(apps_zip_file):
    # loading the "play_apps" zip file look at one sample to see what it contains
    app_details = ZipFile(apps_zip_file)
    app_file = 'play_apps/a008.com.fc2.blog.androidkaihatu.datecamera2'
    app = pickle.loads(app_details.read(app_file))
    print(app)

    # The relevant data from "play_apps.zip" is: category, score and free
    # "category" can be a list, so first loop is to see the whole list of categories

    # first, I prepare a list of all application categories
    app_category_list = []
    # application data
    app_details = ZipFile(apps_zip_file)
    app_files = ZipFile.namelist(app_details)

    # create data frame to that links app id to features
    for idx, app_file in enumerate(app_files):
        try:
            category = pickle.loads(app_details.read(app_file))['category']
            app_category_list = list(set( app_category_list+category ) )
        except:
            continue
    print('Number of categories is ' + str(len(app_category_list)) )


    # Now that I know the categories, I can extract the application data for all apps
    # "apps_df" will be a DataFrame in which each row represent the data of a single application - id, score, free, and all
    # the categories (each one in a single column).
    #
    # NOTICE - The reason that I dont put the application category in a single column is that
    # some of the application have more than one category
    # Prepare the Dataframe
    print(f"parsing {apps_zip_file} to dataframe")
    app_details = ZipFile(apps_zip_file)
    app_files = ZipFile.namelist(app_details)
    cols_name = ['app_id','app_score','app_free'] + app_category_list
    apps_id_df = pd.DataFrame('0', index = np.arange(len(app_files)) , columns = ['app_id'] )
    apps_df = apps_id_df
    app_score_df = pd.DataFrame(float(0), index = np.arange(len(app_files)) , columns = ['app_score'] )
    apps_df['app_score'] = app_score_df['app_score']
    app_free_df = pd.DataFrame(False, index = np.arange(len(app_files)) , columns = ['app_free'] )
    apps_df['app_free'] = app_free_df['app_free']
    app_category_df = pd.DataFrame(0, index = np.arange(len(app_files)) , columns = app_category_list )
    for category in app_category_list:
        apps_df[category] = app_category_df[category]
    # Fill the dataframe
    print('number of app in data file:' + str(len(app_files)))


    # create data frame to that links app id to features
    for idx, app_file in enumerate(app_files):
        try:
            af_det = pickle.loads(app_details.read(app_file))
            for category in af_det['category']:
                apps_df.at[idx, category] = 1
            score = pickle.loads(app_details.read(app_file))['score']
            apps_df.at[idx, 'app_score'] = float(score)
            free = pickle.loads(app_details.read(app_file))['free']
            apps_df.at[idx, 'app_free'] = free
            app_id = pickle.loads(app_details.read(app_file))['app_id']
            apps_df.at[idx, 'app_id'] = app_id
        except:
            continue

    print('Number of applications in apps_df = ' + str(apps_df.shape[0]) + ', number of columns = ' + str(apps_df.shape[1]))
    apps_df.sample(3)
    return apps_df


def process_merge(data, apps_df):
    # merge the application data with the android_bids_us data
    data = pd.merge(data, apps_df, how='left', on='app_id')
    print('Shape of data after merging application data: ' + str(data.shape[0]) + ' x ' + str(data.shape[1]))

    # Some "bidid" appears twice - duplicates should be dropped
    print('filter duplicate "bidid"')
    data = data.drop_duplicates(subset='bidid', keep="first")
    data.reset_index(drop=True, inplace=True)
    print("data size: ", data.shape)


    # data2 = pd.merge(data, apps_df, how='inner', on='app_id')
    # data2.dropna(inplace=True)
    # data2.reset_index(drop=True, inplace=True)
    # print('Shape of data after merging application data: ' + str(data2.shape[0]) + ' x ' + str(data2.shape[1]))
    # data = data2


    # The merging gives us some NaN values.
    print('replacing missing values')
    print(str(round(data['GAME_ACTION'].isnull().sum()/len(data['GAME_ACTION']), 3)*100) + '% of the data is NaN')
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    imputer = imputer.fit(data)
    data.iloc[:,:] = imputer.transform(data)

    # I replace all nan frequent value of each feature
    # for col in data.columns:
    #     data[col] = imputer.fit_transform(data[[col]])
        # frequent_val = s.value_counts().index[0]
        # s.replace(to_replace=['unknown', np.nan], value=frequent_val, inplace=True)
        # data[col] = s


    # check is all NaN were replaced
    if data.isnull().values.any():
        print('There is still Nan in the data')

        for col in data.columns:
            missing = data[col].isnull().sum()
            if missing == 0:
                continue

            print(col, "missing values: ", missing, 100*missing/len(data))
            data[col] = data[col].fillna(data[col].value_counts().index[0])
            #data[col] = imputer.fit_transform(data[[col]])
    else:
        print('All NaN were replaced')


    # The time in the data is given by utc time,
    # the code below converts it to local time (using the state of the user)and extract the day and hour
    # us states timezone table
    print('convert UTC_datetime to local')
    state2timezone = { 'AK': 'US/Alaska', 'AL': 'US/Central', 'AR': 'US/Central', 'AS': 'US/Samoa', 'AZ': 'US/Mountain', 'CA': 'US/Pacific', 'CO': 'US/Mountain', 'CT': 'US/Eastern', 'DC': 'US/Eastern', 'DE': 'US/Eastern', 'FL': 'US/Eastern', 'GA': 'US/Eastern', 'GU': 'Pacific/Guam', 'HI': 'US/Hawaii', 'IA': 'US/Central', 'ID': 'US/Mountain', 'IL': 'US/Central', 'IN': 'US/Eastern', 'KS': 'US/Central', 'KY': 'US/Eastern', 'LA': 'US/Central', 'MA': 'US/Eastern', 'MD': 'US/Eastern', 'ME': 'US/Eastern', 'MI': 'US/Eastern', 'MN': 'US/Central', 'MO': 'US/Central', 'MP': 'Pacific/Guam', 'MS': 'US/Central', 'MT': 'US/Mountain', 'NC': 'US/Eastern', 'ND': 'US/Central', 'NE': 'US/Central', 'NH': 'US/Eastern', 'NJ': 'US/Eastern', 'NM': 'US/Mountain', 'NV': 'US/Pacific', 'NY': 'US/Eastern', 'OH': 'US/Eastern', 'OK': 'US/Central', 'OR': 'US/Pacific', 'PA': 'US/Eastern', 'PR': 'America/Puerto_Rico', 'RI': 'US/Eastern', 'SC': 'US/Eastern', 'SD': 'US/Central', 'TN': 'US/Central', 'TX': 'US/Central', 'UT': 'US/Mountain', 'VA': 'US/Eastern', 'VI': 'America/Virgin', 'VT': 'US/Eastern', 'WA': 'US/Pacific', 'WI': 'US/Central', 'WV': 'US/Eastern', 'WY': 'US/Mountain', '' : 'US/Pacific', '--': 'US/Pacific' }
    data["UTC_datetime"] = pd.to_datetime(round(data['utc_time']/1000), unit='s')
    days = ['mon','tue','wed','thu','fri','sat','sun']

    local_time_vec = []
    local_time_vec_naive_vec = []
    weekday_vec = []
    hour_vec = []
    for idx, t in enumerate(data["UTC_datetime"]):
        local_with_tz = t.tz_localize(tz=state2timezone[data['user_state'][idx]], ambiguous=True)
        utc_time = str(local_with_tz)[0:19]
        utc_time = datetime.strptime(utc_time, '%Y-%m-%d %H:%M:%S')
        offset = str(local_with_tz)[19:22]
        local_time_naive = utc_time + timedelta(hours = int(offset))
        local_time_vec.append(local_with_tz)
        local_time_vec_naive_vec.append(local_time_naive)
        hour_vec.append(local_time_naive.hour)
        weekday_vec.append(days[local_time_naive.weekday()])
    data['local_time_naive'] = local_time_vec_naive_vec
    data['local_time'] = local_time_vec
    data['local hour'] = hour_vec
    data['local_weekday'] = weekday_vec

    # data.drop(['bidid', 'UTC_datetime', 'marketplace'], axis=1, inplace=True)
    return data


# ## Data Exploration - understanding the content of the data
def process_data(data):
    # print(f'reading data from {data_file}')
    # data = pd.read_csv('data/data.csv')
    data.dropna(inplace=True)

    data['local_time_naive'] = pd.to_datetime(data['local_time_naive'])   # convert column to Datetime format
    print('data size: ', data.shape)

    print('columns: ', data.columns)

    # there are 74 columns. If I want to divide to types:
    # 1. 11 basic features: 'bidid', 'utc_time', 'app_id', 'user_state', 'user_isp', 'device_maker',
    #        'device_model', 'device_osv', 'device_height', 'device_width',
    #        'marketplace'
    # 2. 1 target: 'click'
    # 3. 2 app features: 'app_score', 'app_free'
    # 4. 55 app categories: 'MUSIC_AND_AUDIO',
    #        'BUSINESS', 'EVENTS', 'BEAUTY', 'ENTERTAINMENT', 'FAMILY_CREATE',
    #        'LIBRARIES_AND_DEMO', 'SPORTS', 'GAME_BOARD', 'GAME_ACTION',
    #        'GAME_PUZZLE', 'GAME_SIMULATION', 'FOOD_AND_DRINK', 'TRAVEL_AND_LOCAL',
    #        'PARENTING', 'GAME_SPORTS', 'GAME_ADVENTURE', 'FAMILY_MUSICVIDEO',
    #        'FINANCE', 'HOUSE_AND_HOME', 'FAMILY_EDUCATION', 'SHOPPING',
    #        'NEWS_AND_MAGAZINES', 'COMICS', 'FAMILY_ACTION', 'GAME_ARCADE',
    #        'BOOKS_AND_REFERENCE', 'PRODUCTIVITY', 'PERSONALIZATION',
    #        'GAME_EDUCATIONAL', 'VIDEO_PLAYERS', 'SOCIAL', 'GAME_RACING',
    #        'GAME_CARD', 'DATING', 'LIFESTYLE', 'ART_AND_DESIGN',
    #        'FAMILY_BRAINGAMES', 'GAME_CASINO', 'AUTO_AND_VEHICLES',
    #        'FAMILY_PRETEND', 'HEALTH_AND_FITNESS', 'COMMUNICATION', 'MEDICAL',
    #        'EDUCATION', 'GAME_MUSIC', 'PHOTOGRAPHY', 'GAME_WORD',
    #        'MAPS_AND_NAVIGATION', 'GAME_ROLE_PLAYING', 'GAME_TRIVIA',
    #        'GAME_CASUAL', 'TOOLS', 'GAME_STRATEGY', 'WEATHER'
    # 5. 5 time features: 'UTC_datetime',
    #        'local_time_naive', 'local_time', 'local hour', 'local_weekday'


    # ## Data Exploration - Dropping irrelevant columns
    # data = data.drop(['bidid','utc_time','app_id','user_isp','device_maker','device_model','device_osv','marketplace','local_time','UTC_datetime'], axis=1)
    data = data.drop(['bidid', 'utc_time', 'user_isp', 'device_osv', 'marketplace', 'local_time', 'UTC_datetime'], axis=1)

    # ## 2 - Data Exploration - See how imbalanced the data of impressions is

    # Group dat by the click parameter
    click_data = data.groupby('click')['user_state'].count()
    print('click_data: ', click_data)


    basic_click_rate = click_data[1] / (click_data[0] + click_data[1])
    print('basic click rate is :' + str(round(basic_click_rate, 3)))


    # The data is highly imbalanced. That should be taken into consideration while training


    # There are 55 categories, but most of the data is contained in the top categories.
    # I will take only the categories that contains 99% of  the data, and in that way I will reduce the number of categories
    # significantly without losing data
    cat = ['MUSIC_AND_AUDIO',
           'BUSINESS', 'EVENTS', 'BEAUTY', 'ENTERTAINMENT', 'FAMILY_CREATE',
           'LIBRARIES_AND_DEMO', 'SPORTS', 'GAME_BOARD', 'GAME_ACTION',
           'GAME_PUZZLE', 'GAME_SIMULATION', 'FOOD_AND_DRINK', 'TRAVEL_AND_LOCAL',
           'PARENTING', 'GAME_SPORTS', 'GAME_ADVENTURE', 'FAMILY_MUSICVIDEO',
           'FINANCE', 'HOUSE_AND_HOME', 'FAMILY_EDUCATION', 'SHOPPING',
           'NEWS_AND_MAGAZINES', 'COMICS', 'FAMILY_ACTION', 'GAME_ARCADE',
           'BOOKS_AND_REFERENCE', 'PRODUCTIVITY', 'PERSONALIZATION',
           'GAME_EDUCATIONAL', 'VIDEO_PLAYERS', 'SOCIAL', 'GAME_RACING',
           'GAME_CARD', 'DATING', 'LIFESTYLE', 'ART_AND_DESIGN',
           'FAMILY_BRAINGAMES', 'GAME_CASINO', 'AUTO_AND_VEHICLES',
           'FAMILY_PRETEND', 'HEALTH_AND_FITNESS', 'COMMUNICATION', 'MEDICAL',
           'EDUCATION', 'GAME_MUSIC', 'PHOTOGRAPHY', 'GAME_WORD',
           'MAPS_AND_NAVIGATION', 'GAME_ROLE_PLAYING', 'GAME_TRIVIA',
           'GAME_CASUAL', 'TOOLS', 'GAME_STRATEGY', 'WEATHER']
    print('Number of categories before of 100% of the data:' + str(len(cat)))

    a = np.sum(data[cat])
    cat_list = np.cumsum(a.sort_values(ascending=False)/a.sum())
    # plt.figure()
    # cat_list.plot.bar(rot=0)
    # plt.xlabel('Category')
    # plt.ylabel('Cumsum over relative number of appearance')
    # plt.title('Cumulative sum over number of appearance vs. Category')
    # plt.show()


    small_cat_to_drop = list(cat_list[cat_list > 0.98].index)
    print(f'Number of category to drop and to keep 99% of the data: {len(small_cat_to_drop)}')
    data = data.drop(small_cat_to_drop, axis=1)
    # we dropped 33 categories (out of 55), while dropping only 1% of the data


    # ## 2 - Data Exploration - Understanding the reducing number of time categories
    # I first check the CLICK RATE (clicks divided by all impressions)
    # from now on "CLICK RATE" will be written as CR for comfort
    data.groupby('local hour')[['click']].mean().plot(kind='bar', rot=50, ylim=(0.030,0.04))
    #plt.show()


    # I can see a big difference CR parameter - much higher during the morning and afternoon hours
    # Instead of using it as a categorical feature I will divide the time to 4 (slightly overlapping) periods of the day
    # night: 20:00-7:00
    # morning: 6:00-14:00
    # noon: 13:00-16:00
    # afternoon: 15:00-21:00
    data['night']       = ( (data['local hour'] > 20).astype('int') + (data['local hour'] <= 7).astype('int') ) > 0
    data['morning']     = ( (data['local hour'] > 7).astype('int') + (data['local hour'] <= 14).astype('int') ) == 2
    data['noon']        = ( (data['local hour'] > 13).astype('int') + (data['local hour'] <= 16).astype('int') ) == 2
    data['afternoon']   = ( (data['local hour'] > 15).astype('int') + (data['local hour'] <= 21).astype('int') ) == 2
    data = data.drop(['local hour'], axis=1)

    # try to see the distribution of click rate through the weekday
    data.groupby('local_weekday')[['click']].mean().plot.bar(ylim=(0.030,0.04))


    # The CR parameters changes between days - high in the weekend (Saturday and Sunday), low on Thursday and Friday.
    # I will use it later as categorical feature


    # ## 2 - Data Exploration - Understanding the State category

    # I look if the user state gives more info about the CR
    data.groupby('user_state')[['click']].mean().plot(kind='bar', rot=50, ylim=(0.015,0.045))
    # It looks like a significant parameter - there is a big difference in CR between states

    # I want to see the distribution of impressions between states
    states_impressions = data.groupby('user_state')[['click']].count()
    states_impressions.sort_values('click')

    states_impressions_cs = np.cumsum(states_impressions.sort_values('click', ascending = False)/states_impressions.sum())
    plt.figure()
    states_impressions_cs.plot.bar(rot=0)
    plt.xlabel('States')
    plt.ylabel('Cumsum Impression')
    plt.title('Cumulative sum Impressions vs. State')
    plt.show()


    # The number of impressions in not evenly distributed between states, but in order to get ~99% of the data we need
    # almost all the states, so no reduction will be made


    # ## 2 - Data Exploration - Understanding the application free/notfree feature

    # First I want to see how many impressions come from free applications and how many are not
    #print(data.groupby('app_free')[['click']].count())
    # There is almost no information on non-free applications

    #print(data.groupby('app_free')[['click']].mean())

    # Looking at the CR, its looks informative, but since there is almost no data I will drop it
    #data = data.drop(['app_free'], axis=1)


    # First we see the amount of impressions per score bin (using histogram)
    data.groupby('app_score')[['click']].count().plot(kind='bar', rot=50)


    # There are relatively low number of impressions in apps below score of 3, so I won't take them into consideration

    # fill the score below 3 with median score value
    data.loc[data.app_score < 3, 'app_score'] = np.nan
    s = data['app_score']
    frequent_val = s.value_counts().index[0]
    s.replace(to_replace=['uknown', np.nan], value=frequent_val, inplace=True)
    data['app_score'] = s


    # Histogram of clck rate per score bin
    data.groupby('app_score')[['click']].mean().plot(kind='bar', rot=50 )

    # CR vs score looks informative

    target_encoder = TargetEncoder(cols=['app_id', 'device_model', 'user_state'])
    te_df = target_encoder.fit_transform(data[['app_id', 'device_model', 'user_state']], data['click'])
    data['app_id'] = te_df['app_id']
    data['user_state'] = te_df['user_state']
    data['device_model'] = te_df['device_model']

    print('result columns: ', data.columns)

    # save data to csv file
    csv_file = 'data/data.csv'
    print(f"saving to {csv_file}")
    data.to_csv(csv_file, index=False)
    print(f'file {csv_file} ready')
    return data



def run():
    # loading the "android_bids_us" csv file look at one sample to see what it contains
    data = pd.read_csv(BIDS_FILE)
    print('data size: ' + str(data.shape[0]) + ' samples , ' + str(data.shape[1]) + ' columns')
    print('sample row:\n', data.sample(1))

    apps_df = process_zip(APPS_ZIP_FILE)
    data = process_merge(data, apps_df)
    csv_file = process_data(data)

    print("done")


run()
