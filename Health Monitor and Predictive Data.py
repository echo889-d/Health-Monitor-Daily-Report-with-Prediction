##### import libraries #####
#reference:
#https://plotly.com/python/time-series/
#https://plotly.com/python/table-subplots/
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, metrics

import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import plotly.figure_factory as ff
import numpy as np
import statsmodels.api as sm

from google.cloud import bigquery
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('path/to/file.json')
project_id = 'my-bq'
client = bigquery.Client(credentials= credentials,project=project_id)


#Step 1: Query the results
query_job = client.query("""
WITH 
recent_launch_stage AS (
  SELECT 
    id AS promotion_id,
    ds,
    stage AS promotion_stage,
    EXTRACT(date FROM TIMESTAMP_SECONDS(stage_since_timestamp) AT TIME ZONE "America/Los_Angeles") AS stage_since_timestamp,
    in_campaign_automation,
  FROM daily_promotion 
  WHERE active = 1 
    AND is_volume = 0
    AND EXTRACT(date FROM TIMESTAMP_SECONDS(stage_since_timestamp) AT TIME ZONE "America/Los_Angeles") BETWEEN DATE(DATE_ADD(CURRENT_DATE(), INTERVAL -45 DAY)) AND DATE(DATE_ADD(CURRENT_DATE(), INTERVAL -1 DAY))
  GROUP BY 1,2,3,4,5
),

earliest_launch_stage_with_stage AS(
  SELECT
    DISTINCT
    promotion_id,
    ds,
    promotion_stage AS earliest_stage,
    in_campaign_automation,
    MIN(stage_since_timestamp) OVER(PARTITION BY promotion_id) AS earliest_stage_date
  FROM recent_launch_stage
),

earliest_launch_stage AS(
  SELECT 
    * EXCEPT(row_number)
  FROM (
    SELECT 
      promotion_id,
      earliest_stage, 
      earliest_stage_date,
      in_campaign_automation,
      ROW_NUMBER() OVER(PARTITION BY promotion_id ORDER BY ds ASC) AS row_number
    FROM earliest_launch_stage_with_stage
    )
  WHERE row_number = 1
),

ops_owner AS (
    SELECT 
        login_id,
        CONCAT(first_name, ' ',last_name) AS ops_owner_name
    FROM admin_user 
),

biz_owner AS (
    SELECT 
        login_id,
        CONCAT(first_name, ' ',last_name) AS biz_owner_name
    FROM admin_user
),

company AS (
  SELECT 
    app.app_id,
    app.company_id,
    company.company_name,
    management_region AS region
  FROM app AS app
  LEFT JOIN company AS company ON app.company_id=company.company_id
  LEFT JOIN sfdc_accounts_users  AS account ON app.company_id = account.company_id
  WHERE dt = DATE_ADD(CURRENT_DATE(), INTERVAL -2 DAY)
),

current_input AS (
  SELECT 
    promotion.id AS promotion_id,
    promotion.name AS promotion_name,
    CONCAT('(', promotion.id, ') ', promotion.name) AS promotion,
    promotion.is_volume,
    company.company_name,
    campaign.id AS campaign_id,
    campaign.name AS campaign_name,
    campaign.studio,
    campaign.tier,
    campaign.platform,
    biz_owner.biz_owner_name,
    ops_owner.ops_owner_name,
    promotion.in_campaign_automation AS current_in_ca,
    region,
    CASE WHEN promotion.id IN (SELECT promotion_id FROM earliest_launch_stage) THEN 1 ELSE 0 END AS new_launch
    
  FROM promotion AS promotion
  LEFT JOIN campaign AS campaign ON campaign.id = promotion.campaign_id
  LEFT JOIN biz_owner ON campaign.business_owner = biz_owner.login_id
  LEFT JOIN ops_owner ON campaign.ops_owner = ops_owner.login_id
  LEFT JOIN company AS company ON company.app_id = campaign.advertiser_app
),

ie AS (
    SELECT
      dt,
      SAFE_CAST(promotion_id AS INT) AS promotion_id,
      model_strategy,
      model_version,
      adexchange,
      
      SUM (impressions) AS impressions,
      SUM(installs) AS installs,
      SUM(money_paid/1000) AS money_paid,
      SUM(money_collected) AS money_collected,
      SUM(money_collected) - SUM(money_paid/1000) AS net,
      SUM(d1_iap_revenue) AS d1_iap,
      SUM(d3_iap_revenue) AS d3_iap,
      SUM(d7_iap_revenue) AS d7_iap,
    FROM daily_uber_aggr dua
    WHERE dt >= DATE(DATE_ADD(CURRENT_DATE(), INTERVAL -80 DAY))
    AND model_strategy NOT IN ('unified_d7_roas')
    GROUP BY 1,2,3,4,5
    --ORDER BY dt DESC
),

d1 AS(
    SELECT
        dt,
        model_strategy,
        adexchange,
        model_version,
        promotion_id,
        promotion_name,
        in_campaign_automation,
        is_volume,
        type,
        studio,
        platform,
        tier,
        attribution_provider,
        sum(impressions) impressions,
        sum(installs) installs,
        sum(money_paid) money_paid,
        sum(money_collected) money_collected,
        sum(net) net,
        sum(d1_iap) d1_iap,
        sum(d1_spend_capacity) d1_spend_capacity,
    FROM dsp_hourly_impression_enhanced_v2 
     --daily_impression_enhanced
    WHERE SAFE_CAST(hour_pst_now AS INT) = 22 --8am for AMS; 15pm for Beijing
    AND date_pst_now = dt 
    AND dt >= current_date() - 80
    AND dt NOT IN ('2023-12-01','2023-12-02') --unusual data
    AND model_strategy NOT IN ('unified_d7_roas')
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13
),

overbid AS (
SELECT 
  type_id, 
  ds, 
  AVG(value) AS overbid 
  FROM daily_overbid 
  WHERE type = 'promotion' AND goal_type = 'd7_roas' 
  GROUP BY 1,2
),

daily_number AS (
  SELECT 
    DISTINCT
    ie.dt AS dt,
    ie.promotion_id,
    dsb.amount AS customer_budget,
    dsb.target_percentage AS spend_target,
    dsb.amount * dsb.target_percentage AS padded_daily_budget,
    COALESCE(dsb.d7/100, 0) AS d7_roas_goal,
    --dua.* EXCEPT (dt, promotion_id),
    --da.max_cpi AS daily_max_cpi,
    --da.min_cpi AS daily_min_cpi,
    --p.margin_target AS daily_margin_target,
    --da.padding_factor AS daily_padding_factor,
    eig.mode AS daily_eig_mode,
    eig.install_goal AS daily_install_goal,
    eig.ecpi_goal AS daily_ecpi_goal,
    overbid.overbid AS promotion_overbid,
    
  FROM ie
  LEFT JOIN (SELECT * FROM daily_shared_budget WHERE type = 'daily' ) AS dsb ON SAFE_CAST(ie.promotion_id AS INT) = SAFE_CAST(dsb.promotion_id AS INT) AND ie.dt = dsb.ds
  LEFT JOIN daily_promotion AS p ON SAFE_CAST(ie.promotion_id AS INT) = SAFE_CAST(p.id AS INT) AND ie.dt = p.ds
  LEFT JOIN (SELECT * FROM daily_automation WHERE strategy = 'cpi_updater' ) AS da ON SAFE_CAST(ie.promotion_id AS INT) = SAFE_CAST(da.promotion_id AS INT) AND ie.dt = da.ds
  LEFT JOIN daily_eig AS eig ON SAFE_CAST(eig.promotion_id AS INT) = SAFE_CAST(ie.promotion_id AS INT) AND ie.dt = eig.ds 
  LEFT JOIN overbid AS overbid ON overbid.type_id = SAFE_CAST(ie.promotion_id AS INT) AND overbid.ds = ie.dt
  WHERE ie.dt >= DATE(DATE_ADD(CURRENT_DATE(), INTERVAL -80 DAY))
)



SELECT
    ie.dt,
    FORMAT_TIMESTAMP('%A', ie.dt) AS DOW, 
    d1.model_strategy,
    d1.adexchange,
    d1.model_version,
    d1.promotion_id,
    d1.promotion_name,
    d1.in_campaign_automation,
    d1.is_volume,
    d1.type,
    d1.studio,
    d1.platform,
    d1.tier,
    d1.attribution_provider,
    
    --dimension

    d1.impressions AS d1_impression_22h,
    d1.installs AS d1_install_22h,
    d1.money_paid AS d1_paid_22h,
    d1.money_collected AS d1_spend_22h,
    d1.net AS d1_net_22h,
    d1.d1_iap AS d1_iap_22h,
    CASE WHEN d1.installs=0 then 0 ELSE d1.money_collected/d1.installs END AS d1_cpi,
    d1_spend_capacity AS d1_spend_capacity_22h,
    --d1_23h
    
    ie.impressions AS realized_impression,
    ie.installs AS realized_install,
    ie.money_paid AS realized_paid,
    ie.money_collected AS realized_spend,
    ie.net AS realized_net,
    ie.d1_iap AS realized_d1_iap,
    ie.d3_iap AS realized_d3_iap,
    ie.d7_iap AS realized_d7_iap,
    ie.d1_iap/NULLIF(d7_roas_goal,0) AS realized_d1_spend_capacity,
    ie.d3_iap/NULLIF(d7_roas_goal,0) AS realized_d3_spend_capacity,
    ie.d7_iap/NULLIF(d7_roas_goal,0) AS realized_d7_spend_capacity,
    --realized
    current_input.company_name,
    current_input.region,
    current_input.new_launch,
    current_input.ops_owner_name,
    
    daily_number.customer_budget,
    daily_number.spend_target,
    daily_number.padded_daily_budget,
    
    SUM(d1.net) OVER(PARTITION BY d1.dt,d1.promotion_id) AS sum_d1_net,
    SUM(d1.money_collected) OVER(PARTITION BY d1.dt,d1.promotion_id) AS sum_d1_spend,
    SUM(d1.d1_iap) OVER(PARTITION BY d1.dt,d1.promotion_id) AS sum_d1_iap,
    SUM(d1_spend_capacity) OVER(PARTITION BY d1.dt,d1.promotion_id) AS sum_d1_capacity,
FROM ie
LEFT JOIN d1 
    ON ie.dt= d1.dt
    AND SAFE_CAST(ie.promotion_id AS INT) = d1.promotion_id 
    AND ie.model_strategy = d1.model_strategy
    AND ie.adexchange = d1.adexchange
    AND ie.model_version = d1.model_version
LEFT JOIN daily_number ON SAFE_CAST(ie.promotion_id AS INT) = daily_number.promotion_id AND ie.dt = daily_number.dt
LEFT JOIN current_input ON SAFE_CAST(current_input.promotion_id AS INT) = SAFE_CAST(ie.promotion_id AS INT)
ORDER BY ie.dt DESC
""")

datasets = query_job.result()

#Step 2: transfrom query result to dataframe
Xy_all = datasets['D1_&_realized']

##update date column to date
Xy_all['dt'] = pd.to_datetime(Xy_all['dt'], format='%Y-%m-%d') 
print(Xy_all.columns)
###Xy_all.head()

##overall data: group by date then sum
def define_mask(Xy_all_sum):
  mask = Xy_all_sum['dt'].isin(['2023-12-01','2023-12-02'])
  Xy_all_sum = Xy_all_sum[~mask]
  return Xy_all_sum
  
Xy_all_sum = Xy_all[['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby('dt',as_index=False).sum()
Xy_all_sum = define_mask(Xy_all_sum)

###data: group by date and in_ca, and is_volume == 0
Xy_in_CA_1 = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['in_campaign_automation'] == 1)][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()
Xy_in_CA_0 = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['in_campaign_automation'] == 0)][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()

Xy_in_CA_1 = define_mask(Xy_in_CA_1)
Xy_in_CA_0 = define_mask(Xy_in_CA_0)

###data: group by date and model strategy, and is_volume == 0
Xy_model_strategy_value = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['model_strategy'] == 'go_linear_d7_roas')][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()
Xy_model_strategy_volume = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['model_strategy'] == 'go_linear_ecpi')][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()
Xy_model_strategy_other = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['model_strategy'] != 'go_linear_ecpi') & (Xy_all['model_strategy'] != 'go_linear_d7_roas')][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()

Xy_model_strategy_value = define_mask(Xy_model_strategy_value)
Xy_model_strategy_value = define_mask(Xy_model_strategy_value)
Xy_model_strategy_value = define_mask(Xy_model_strategy_value)


###data: group by date and platform, and is_volume == 0
Xy_GP = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['platform'] == 'Google Play')][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()
Xy_iOS = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['platform'] == 'iOS')][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()
Xy_AMZ = Xy_all.loc[(Xy_all['is_volume'] == 0) & (Xy_all['platform'] == 'Amazon')][['dt','d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h','realized_impression','realized_install','realized_paid','realized_spend','realized_net','realized_d1_iap','realized_d3_iap','realized_d7_iap','realized_d1_spend_capacity','realized_d3_spend_capacity','realized_d7_spend_capacity']].groupby(by = ['dt'],as_index=False).sum()

Xy_GP = define_mask(Xy_GP)
Xy_iOS = define_mask(Xy_iOS)
Xy_AMZ = define_mask(Xy_AMZ)

###preview promotion level information
Xy_all_sum.head()
Xy_GP.head()

##Split dataframe to training (mature data, days = [-45,-8]) and test (immature data, days = [-7,0]), and validation data to check test errors
def split_train_test(all_data = Xy_all_sum):
  Xy_train = all_data.loc[(all_data['dt'] <= all_data['dt'].max() + timedelta(days=-8)) & (all_data['dt'] >= all_data['dt'].max() + timedelta(days=-45))]
  Xy_valid = all_data.loc[all_data['dt'] < all_data['dt'].max() + timedelta(days=-45)]
  Xy_test = all_data.loc[(all_data['dt'] > all_data['dt'].max() + timedelta(days=-8))]
  return Xy_train, Xy_valid, Xy_test

###preview the train and test from the dataframe group by date
Xy_train, Xy_valid, Xy_test = split_train_test(Xy_all_sum)
print(Xy_train.columns)
print('Training data period: ', Xy_train['dt'].max(), Xy_train['dt'].min())
print('Testing data period: ', Xy_test['dt'].max(), Xy_test['dt'].min())
print('Validating data period: ', Xy_valid['dt'].max(), Xy_valid['dt'].min())


###preview the train and test from the dataframe group by date and in_ca
Xy_train_ca_1, Xy_valid_ca_1, Xy_test_ca_1 = split_train_test(Xy_in_CA_1)
Xy_train_ca_0, Xy_valid_ca_0, Xy_test_ca_0 = split_train_test(Xy_in_CA_0)
#Xy_train_ca.head()

###preview the train and test from the dataframe group by date and in_ca
Xy_train_value, Xy_valid_value, Xy_test_value = split_train_test(Xy_model_strategy_value)
Xy_train_volume, Xy_valid_volume, Xy_test_volume = split_train_test(Xy_model_strategy_volume)

###preview the train and test from the dataframe group by date and in_ca
Xy_train_GP, Xy_valid_GP, Xy_test_GP = split_train_test(Xy_GP)
Xy_train_iOS, Xy_valid_iOS, Xy_test_iOS = split_train_test(Xy_iOS)
Xy_train_AMZ, Xy_valid_AMZ, Xy_test_AMZ = split_train_test(Xy_AMZ)

#Step3: Build Linear Models
###get the data frame column names
X_list = ['d1_impression_22h', 'd1_install_22h', 'd1_paid_22h','d1_spend_22h', 'd1_net_22h', 'd1_iap_22h', 'd1_spend_capacity_22h']
y_list = ['realized_impression', 'realized_install', 'realized_paid','realized_spend', 'realized_net','realized_d7_iap','realized_d7_spend_capacity']
y_expected_list = ['expected_impression', 'expected_install', 'expected_paid','expected_spend', 'expected_net','expected_d7_iap','expected_d7_spend_capacity']

def lr_models(Xy_df=Xy_all_sum, df_train=Xy_train, X="d1_impression_22h", y="realized_impression",y_expected="expected_impression"):
  LR = LinearRegression().fit(df_train[[X]], df_train[[y]])
  Xy_df[[y_expected]] = LR.predict(Xy_df[[X]])
  return Xy_df

###predict overall numbers for impression, install, spend. Net and capacity use different models.
for i in range(len(X_list)):
  Xy_all_sum = lr_models(Xy_all_sum, Xy_train, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  
  #Xy_in_CA_1 = lr_models(Xy_in_CA_1, Xy_train_ca_1, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  #Xy_in_CA_0 = lr_models(Xy_in_CA_0, Xy_train_ca_0, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  
  Xy_model_strategy_value = lr_models(Xy_model_strategy_value, Xy_train_value, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  Xy_model_strategy_volume = lr_models(Xy_model_strategy_volume, Xy_train_volume, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  
  Xy_GP = lr_models(Xy_GP, Xy_train_GP, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  Xy_iOS = lr_models(Xy_iOS, Xy_train_iOS, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])
  Xy_AMZ = lr_models(Xy_AMZ, Xy_train_AMZ, X = X_list[i], y= y_list[i] , y_expected = y_expected_list[i])

####df = Xy_all_sum.loc[Xy_all_sum["dt"].isin(["2023-06-20","2023-06-19","2023-06-13","2023-06-06","2023-05-30"])][["dt","d1_spend_22h","d1_net_22h","d1_iap_22h","d1_spend_capacity_22h","expected_spend","expected_net","expected_d7_iap","expected_d7_spend_capacity"]]
df = Xy_all_sum[["dt","d1_spend_22h","d1_net_22h","d1_iap_22h","d1_spend_capacity_22h","expected_spend","expected_net","expected_d7_iap","expected_d7_spend_capacity"]]
####mode.export_csv(df)
####df.tail()

#Step 4: Predictions and Visualize
def visulize_charts(name='impression', x=Xy_all_sum[['dt']], y_d1=Xy_all_sum[['d1_impression_22h']],y_realized=Xy_all_sum[['realized_impression']], y_expected=Xy_all_sum[['expected_impression']],y_dtick=1000000, width=1000, height=400, title_text = "impression"):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y_d1,
                    mode='lines',
                    name= ("D1_22h " + name),
                    line=dict(color=px.colors.qualitative.Set1[2]),
                    hovertemplate = '%{y:f} <br>%{x}'
                    ))
  fig.add_trace(go.Scatter(x=x, y=y_realized,
                    mode='lines',
                    name= ('realized ' + name),
                    line=dict(color=px.colors.qualitative.Set1[1]),
                    hovertemplate = '%{y:f} <br>%{x}'
                    ))
  fig.add_trace(go.Scatter(x=x, y=y_expected,
                    mode='lines', 
                    name= ('expected ' + name),
                    line=dict(color=px.colors.qualitative.Set1[4]),
                    hovertemplate = '%{y:f} <br>%{x}'
                    ))
  fig.update_yaxes(dtick=y_dtick)
  fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=14, label="2w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
  fig.update_layout(width=width,height=height,title=dict(text= title_text))
  fig.show()


##impression
visulize_charts(name='impression', x=Xy_all_sum['dt'], y_d1=Xy_all_sum['d1_impression_22h'],y_realized=Xy_all_sum['realized_impression'], y_expected=Xy_all_sum['expected_impression'],y_dtick=2 * 10 ** 6,width=550,height=350, title_text = "impression")

##install
visulize_charts(name='install', x=Xy_all_sum['dt'], y_d1=Xy_all_sum['d1_install_22h'],y_realized=Xy_all_sum['realized_install'], y_expected=Xy_all_sum['expected_install'],y_dtick= 2 * 10 ** 4, width=550,height=350, title_text = "install")

##spend
visulize_charts(name='spend', x=Xy_all_sum['dt'], y_d1=Xy_all_sum['d1_spend_22h'],y_realized=Xy_all_sum['realized_spend'], y_expected=Xy_all_sum['expected_spend'],y_dtick= 3 * 10 ** 4, title_text = "overall spend")

##net
X = Xy_train[['d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h']]
#X = Xy_train[['d1_net_22h']]
y = Xy_train[['realized_net']]
olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
y_repd = olsres.predict(Xy_all_sum[['d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h']])
#y_repd = olsres.predict(Xy_all_sum[['d1_net_22h']])
#plt.plot(y_repd)
#plt.plot(Xy_all_sum[["realized_net"]])

#print(np.sqrt(metrics.mean_squared_error(Xy_all_sum[['realized_net']], y_repd)))
#print(olsres.summary())
visulize_charts(name='net', x=Xy_all_sum['dt'], y_d1=Xy_all_sum['d1_net_22h'],y_realized=Xy_all_sum['realized_net'], y_expected=y_repd,y_dtick= 2 * 10 ** 4, title_text = "Overall Net")

Xy_all_sum['d1_22h_margin'] = Xy_all_sum['d1_net_22h']/Xy_all_sum['realized_spend']
Xy_all_sum['expected_margin'] = y_repd/Xy_all_sum['expected_spend']
Xy_all_sum['realized_margin'] = Xy_all_sum['realized_net']/Xy_all_sum['realized_spend']

name = "margin"
fig = go.Figure()
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['d1_22h_margin'],
                  mode='lines',
                  name= ("D1_22h " + name),
                  line=dict(color=px.colors.qualitative.Set1[2]),
                  hovertemplate = '%{y:0.3f} <br>%{x}'
                  ))
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['realized_margin'],
                  mode='lines',
                  name= ('realized ' + name),
                  line=dict(color=px.colors.qualitative.Set1[1]),
                  hovertemplate = '%{y:0.3f} <br>%{x}'
                  ))
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['expected_margin'],
                  mode='lines', 
                  name= ('expected ' + name),
                  line=dict(color=px.colors.qualitative.Set1[4]),
                  hovertemplate = '%{y:0.3f} <br>%{x}'
                  ))
fig.update_yaxes(dtick= 0.2)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=14, label="2w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
fig.update_layout(width=1000,height=400,title=dict(text= "Margin"))
fig.show()              

##iap
X = Xy_train[['d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h']]
y = Xy_train[['realized_d3_iap']]
olsmod = sm.OLS(y, X)
olsres = olsmod.fit()

y_train = olsres.predict(Xy_train[['d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h']])
y_sum = olsres.predict(Xy_all_sum[['d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h']])

X_d1 = y_train
y_d1 = Xy_train[['realized_d7_iap']]
olsmod_d1 = sm.OLS(y_d1, X_d1)
olsres_d1 = olsmod_d1.fit()
y_repd = olsres_d1.predict(y_sum)

"""
from sklearn.linear_model import LarsCV
reg = LarsCV(cv=6).fit(X, y)
y_repd = reg.predict(Xy_all_sum[['d1_impression_22h','d1_install_22h','d1_paid_22h','d1_spend_22h','d1_net_22h','d1_iap_22h','d1_spend_capacity_22h']])
"""

name = "iap"
fig = go.Figure()
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['d1_iap_22h'],
                  mode='lines',
                  name= ("D1_22h " + name),
                  line=dict(color=px.colors.qualitative.Set1[2]),
                  hovertemplate = '%{y:f} <br>%{x}'
                  ))
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['realized_d7_iap'],
                  mode='lines',
                  name= ('realized ' + name),
                  line=dict(color=px.colors.qualitative.Set1[1]),
                  hovertemplate = '%{y:f} <br>%{x}'
                  ))
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=y_repd,
                  mode='lines', 
                  name= ('expected ' + name),
                  line=dict(color=px.colors.qualitative.Set1[4]),
                  hovertemplate = '%{y:f} <br>%{x}'
                  ))
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['realized_d1_iap'],
                  mode='lines', 
                  name= ('realized_d1 ' + name),
                  line=dict(color=px.colors.qualitative.Set1[6]),
                  hovertemplate = '%{y:f} <br>%{x}'
                  ))
fig.add_trace(go.Scatter(x=Xy_all_sum['dt'], y=Xy_all_sum['realized_d3_iap'],
                  mode='lines', 
                  name= ('realized_d3 ' + name),
                  line=dict(color=px.colors.qualitative.Set1[8]),
                  hovertemplate = '%{y:f} <br>%{x}'
                  ))    
fig.update_yaxes(dtick= 5 * 10 ** 3)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=14, label="2w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
fig.update_layout(width=1000,height=400,title=dict(text= "Overall IAP"))
fig.show()