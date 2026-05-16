"""
Olist 데이터를 로드하고 전처리하는 통합 파이프라인 모듈 (버전 2).
위경도(Geolocation) 데이터를 포함하여 SCM 물류 맵 시각화를 지원하며,
가설 검증, 코호트 분석, 객단가(AOV) 등 종합 분석 파생 변수를 모두 생성합니다.
"""
import pandas as pd
import numpy as np
import os
import streamlit as st

@st.cache_data(show_spinner="대규모 Olist 데이터를 병합하고 전처리 중입니다... (최초 로딩 시간 소요)")
def load_and_preprocess_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    
    # 1. 파일 로드 (압축된 ZIP 파일 직접 로드)
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv.zip"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv.zip"))
    reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv.zip"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv.zip"))
    sellers = pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv.zip"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv.zip"))
    translations = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv.zip"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv.zip"))
    
    # 지리(Geolocation) 데이터 로드 및 집계 (동일 우편번호의 평균 좌표)
    geo = pd.read_csv(os.path.join(data_dir, "olist_geolocation_dataset.csv.zip"))
    geo_agg = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    
    # 2. 고객/셀러 위치 매핑
    customers = customers.merge(geo_agg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    customers.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    
    sellers = sellers.merge(geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    sellers.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    
    # 3. 날짜 컬럼 형변환 및 파생변수 생성
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])
        
    orders['delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.total_seconds() / (24*3600)
    
    # 4. 리뷰 및 결제 데이터 병합
    reviews_agg = reviews.groupby('order_id')['review_score'].mean().reset_index()
    orders = orders.merge(reviews_agg, on='order_id', how='left')
    orders = orders.merge(customers[['customer_id', 'customer_unique_id', 'customer_state', 'customer_lat', 'customer_lng']], on='customer_id', how='left')
    
    # 주문별 총 결제 금액 계산 (AOV 분석용)
    payments_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
    orders = orders.merge(payments_agg, on='order_id', how='left')
    
    # 5. 아이템, 카테고리, 셀러 병합 (마스터 테이블 생성)
    items_prod = order_items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    items_prod = items_prod.merge(translations, on='product_category_name', how='left')
    items_seller = items_prod.merge(sellers[['seller_id', 'seller_state', 'seller_lat', 'seller_lng']], on='seller_id', how='left')
    
    df_merged = items_seller.merge(
        orders[['order_id', 'customer_unique_id', 'order_purchase_timestamp', 'delay_days', 'review_score', 
                'payment_value', 'customer_state', 'customer_lat', 'customer_lng', 'order_status']], 
        on='order_id', how='inner'
    )
    
    df_merged['purchase_month'] = df_merged['order_purchase_timestamp'].dt.to_period('M')
    df_merged['purchase_date'] = df_merged['order_purchase_timestamp'].dt.date
    
    # 6. 상위/하위 셀러 분리 로직
    seller_sales = df_merged.groupby('seller_id')['price'].sum().sort_values(ascending=False)
    top_20_cutoff = int(len(seller_sales) * 0.2)
    bottom_20_cutoff = int(len(seller_sales) * 0.8)
    
    top_sellers = set(seller_sales.head(top_20_cutoff).index)
    bottom_sellers = set(seller_sales.tail(len(seller_sales) - bottom_20_cutoff).index)
    
    def classify_seller(sid):
        if sid in top_sellers: return 'Top 20%'
        if sid in bottom_sellers: return 'Bottom 20%'
        return 'Middle 60%'
        
    df_merged['seller_group'] = df_merged['seller_id'].apply(classify_seller)
    
    # 7. 코호트 계산
    first_purchase = df_merged.groupby('customer_unique_id')['order_purchase_timestamp'].min().dt.to_period('M').reset_index()
    first_purchase.columns = ['customer_unique_id', 'cohort_month']
    df_merged = df_merged.merge(first_purchase, on='customer_unique_id', how='left')
    
    df_merged['purchase_m_num'] = df_merged['purchase_month'].dt.year * 12 + df_merged['purchase_month'].dt.month
    df_merged['cohort_m_num'] = df_merged['cohort_month'].dt.year * 12 + df_merged['cohort_month'].dt.month
    df_merged['cohort_index'] = df_merged['purchase_m_num'] - df_merged['cohort_m_num']
    
    return orders, df_merged

def calculate_dau_mau(df_merged):
    dau = df_merged.groupby('purchase_date')['customer_unique_id'].nunique().reset_index()
    dau.columns = ['date', 'dau']
    mau = df_merged.groupby('purchase_month')['customer_unique_id'].nunique().reset_index()
    mau['purchase_month'] = mau['purchase_month'].astype(str)
    mau.columns = ['month', 'mau']
    return dau, mau

def calculate_arppu(df_merged):
    monthly_sales = df_merged.groupby('purchase_month')['payment_value'].sum()
    monthly_users = df_merged.groupby('purchase_month')['customer_unique_id'].nunique()
    arppu = (monthly_sales / monthly_users).reset_index()
    arppu.columns = ['month', 'arppu']
    arppu['month'] = arppu['month'].astype(str)
    return arppu

def calculate_retention(df_merged, metric='users', seller_group=None):
    df = df_merged.copy()
    if seller_group:
        df = df[df['seller_group'] == seller_group]
        
    if len(df) == 0:
        return None
        
    if metric == 'users':
        cohort_data = df.groupby(['cohort_month', 'cohort_index'])['customer_unique_id'].nunique().reset_index()
        cohort_data.rename(columns={'customer_unique_id': 'value'}, inplace=True)
    else:
        cohort_data = df.groupby(['cohort_month', 'cohort_index'])['payment_value'].sum().reset_index()
        cohort_data.rename(columns={'payment_value': 'value'}, inplace=True)
        
    cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='value')
    if 0 in cohort_pivot.columns:
        cohort_size = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    else:
        retention_matrix = cohort_pivot
        
    retention_matrix.index = retention_matrix.index.astype(str)
    return retention_matrix
