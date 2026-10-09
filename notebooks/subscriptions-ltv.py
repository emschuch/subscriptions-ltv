#!/usr/bin/env python
# coding: utf-8

# # Subscriptions LTV
# 
# Exploration of simulated subscriptions data to calculate LTV.

# In[1]:


import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 500


# In[2]:


repo_path = os.path.split(os.getcwd())[0]

sys.path.insert(0, repo_path + "/data")


# In[3]:


subs = pd.read_csv(repo_path + "/data/subscriptions.csv")


# In[4]:


subs.head()


# In[5]:


subs.tail()


# In[6]:


subs.describe()


# In[7]:


# convert date columns to year-month format
def month_truncate(df, column):
    df['year_month_' + column] = df[column].map(lambda x: '-'.join(x.split('-')[:-1]))


# In[8]:


date_cols = ['created_at', 'ended_at']

# fill NAs with the max date
subs['ended_at'].fillna('2026-06-30', inplace=True)

for col in date_cols:
    month_truncate(subs, col)


# In[9]:


# offset cohort and end_month to be the last day of the month
subs['cohort'] = pd.to_datetime(subs['year_month_created_at'], format='%Y-%m') + pd.tseries.offsets.MonthEnd(0)
subs['end_month'] = pd.to_datetime(subs['year_month_ended_at'], format='%Y-%m') + pd.tseries.offsets.MonthEnd(0)
subs['end_date'] = pd.to_datetime(subs['ended_at'], format='%Y-%m-%d')


# In[10]:


set(subs.channel)


# In[11]:


set(subs.utm_campaign)


# In[12]:


set(subs.end_reason)


# In[13]:


def concat_channel(row):
    try:
        return '-'.join([row['channel'], row['utm_campaign']])
    except TypeError:
        return row['channel']


# In[14]:


# create column to capture channel + paid social campaigns
subs['channel_ext'] = subs.apply(concat_channel, axis=1)

# fill NAs for active accounts
subs['end_reason'].fillna('active_account', inplace=True)

# add monthly revenue for the plan the user is subscribed to
subs['monthly_revenue'] = [15.0 if x=='monthly' else 150.0/12 for x in subs['plan']]


# In[15]:


# create list of months since the user subscribes to the max month
# this will ensure that users are still counted as part of a cohort even if they unsubscribe, but not before they subscribed
end_month = max(subs.end_month)

subs['month_since_subscribe'] = subs.cohort.map(lambda x: pd.date_range(start=x, end=end_month, freq='M'))
subs['month_num_since_subscribe'] = subs.month_since_subscribe.map(lambda x: range(len(x)))


# In[16]:


subs.shape


# In[17]:


subs.head()


# In[18]:


# expand the dataframe so that a user has multiple rows for each month since the user subscribed
subs_exp = subs.explode(['month_since_subscribe', 'month_num_since_subscribe'])


# In[19]:


subs_exp.shape


# In[20]:


subs_exp.head()


# In[ ]:


# flag which months the user was actually subscribed and multiply by monthly revenue
# slightly different logic is necessary for monthly vs. annual accounts
subs_exp['active_month'] = subs_exp.apply(lambda x: 1 if x.end_reason == 'active_account'
                                          else 1 if x.end_date == x.month_since_subscribe and x.plan == 'monthly'
                                          else 1 if x.month_since_subscribe < x.end_date
                                          else 0, axis=1)

# multiply the active month flag by revenue to get actual revenue for that month
subs_exp['active_revenue'] = subs_exp.active_month * subs_exp.monthly_revenue


# In[21]:


subs_exp.columns


# In[22]:


subs_exp[subs_exp.active_month == 0].head()


# In[23]:


# aggregate for usage in Tableau
group_columns = ['cohort', 'month_since_subscribe', 'month_num_since_subscribe',
                 'channel', 'channel_ext', 'end_reason', 'plan']

subs_agg = subs_exp.groupby(group_columns).agg(user_count=('subscription_id', 'nunique'),
                                               total_revenue=('active_revenue', 'sum')
                                              ).reset_index()


# In[24]:


subs_agg.columns = ['cohort', 'month', 'month_num'] + list(subs_agg.columns)[3:]


# In[25]:


# reminder, users and total revenue can exist in month 0 even when end reason is 'payment_failed'
# this simply means that payment failed in a subsequest month, not in month 0
subs_agg.head()


# In[26]:


# QA check: the original dataset had 160,000 users, is that still the case?
# we can check if the cohort is the same size each month (it is)
# then filter to month 0 and sum across cohorts 
# looks good! 👍

subs_agg.groupby(['cohort', 'month']).user_count.sum().tail(10)


# In[27]:


sum(subs_agg[subs_agg['month_num'] == 0].groupby('cohort').user_count.sum())


# In[28]:


# create another dataset for visualizing subscription length
group_columns2 = ['cohort', 'subscription_id', 'created_at', 'end_date',
                  'channel', 'channel_ext', 'end_reason', 'plan']

subs_freq = subs_exp.groupby(group_columns2).agg(subscription_length=('active_month', 'sum')).reset_index()


# In[29]:


subs_freq.head()


# In[30]:


group_columns3 = ['cohort', 'channel', 'channel_ext', 'end_reason', 'plan', 'subscription_length']

subs_freq_agg = subs_freq.groupby(group_columns3).agg(num_users=('subscription_id', 'nunique')).reset_index()


# In[31]:


subs_freq_agg.head(10)


# In[32]:


# some annual accounts have a subscription length of 25 months?
# guessing that this is due to recovery tries for about a month
subs_freq[subs_freq['subscription_length'] == 25].head()


# In[33]:


# save outputs to csv
subs_agg.to_csv('../outputs/subs_aggregate.csv', index=False)
subs_freq_agg.to_csv('../outputs/subs_frequency.csv', index=False)


# In[34]:


# convert notebook to script
get_ipython().system('jupyter nbconvert --to script subscriptions-ltv.ipynb')


# In[ ]:




