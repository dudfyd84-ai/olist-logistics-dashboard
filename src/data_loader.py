"""
이 파일은 Olist 이커머스 데이터를 로드하고 전처리하는 스크립트입니다.
여러 CSV 파일을 병합하고 파생 변수(배송 지연일, 상위/하위 셀러 분류 등)를 생성합니다.
"""
import pandas as pd
import numpy as np
import os
import streamlit as st

@st.cache_data(show_spinner="Olist 데이터를 로딩 및 전처리하는 중입니다. (최초 1회 약 1~2분 소요)")
def load_and_preprocess_data(data_dir="c:/Users/테오/Documents/icb10/Project1/data"):
    # 1. 파일 로드
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    translations = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv"))
    
    # 2. 날짜 컬럼 형변환
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])
        
    # 3. 배송 지연 파생 변수 생성 (가설 1 용도)
    # 실제 도착일 - 예상 배송일 (값이 양수면 지연)
    orders['delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.total_seconds() / (24*3600)
    orders['is_late'] = orders['delay_days'] > 0
    orders['delay_category'] = pd.cut(
        orders['delay_days'],
        bins=[-np.inf, -1, 0, 3, 7, np.inf],
        labels=['Early', 'On Time', 'Late(1-3 Days)', 'Late(4-7 Days)', 'Late(>7 Days)']
    )
    
    # 4. 리뷰 병합
    # 하나의 주문에 여러 리뷰가 있을 수 있으므로 첫번째 리뷰(혹은 평균)만 사용
    reviews_agg = reviews.groupby('order_id')['review_score'].mean().reset_index()
    orders = orders.merge(reviews_agg, on='order_id', how='left')
    orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')
    
    # 5. 아이템, 상품명 병합
    items_prod = order_items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    items_prod = items_prod.merge(translations, on='product_category_name', how='left')
    
    # 아이템에 주문 기본 정보 결합
    df_merged = items_prod.merge(
        orders[['order_id', 'customer_unique_id', 'order_purchase_timestamp', 'delay_days', 'delay_category', 'review_score']], 
        on='order_id', how='inner'
    )
    
    # 구매 년월 파생변수
    df_merged['purchase_month'] = df_merged['order_purchase_timestamp'].dt.to_period('M')
    df_merged['purchase_date'] = df_merged['order_purchase_timestamp'].dt.date
    
    # 6. 상위/하위 셀러 라벨링 (가설 2 용도)
    seller_sales = df_merged.groupby('seller_id')['price'].sum().sort_values(ascending=False)
    # 상위 20%, 하위 20%
    top_20_cutoff = int(len(seller_sales) * 0.2)
    bottom_20_cutoff = int(len(seller_sales) * 0.8)
    
    top_sellers = seller_sales.head(top_20_cutoff).index
    bottom_sellers = seller_sales.tail(len(seller_sales) - bottom_20_cutoff).index
    
    def classify_seller(sid):
        if sid in top_sellers: return 'Top 20%'
        if sid in bottom_sellers: return 'Bottom 20%'
        return 'Middle 60%'
        
    df_merged['seller_group'] = df_merged['seller_id'].apply(classify_seller)
    
    # 7. 고객 코호트(첫 구매월) 계산 (리텐션 분석 용도)
    first_purchase = df_merged.groupby('customer_unique_id')['order_purchase_timestamp'].min().dt.to_period('M').reset_index()
    first_purchase.columns = ['customer_unique_id', 'cohort_month']
    df_merged = df_merged.merge(first_purchase, on='customer_unique_id', how='left')
    
    # 코호트 인덱스 계산 (몇 개월 차 구매인지)
    def calculate_cohort_index(df):
        purchase_m = df['purchase_month'].dt.year * 12 + df['purchase_month'].dt.month
        cohort_m = df['cohort_month'].dt.year * 12 + df['cohort_month'].dt.month
        return purchase_m - cohort_m
        
    df_merged['cohort_index'] = calculate_cohort_index(df_merged)
    
    return orders, df_merged

def calculate_dau_mau(df_merged):
    # DAU
    dau = df_merged.groupby('purchase_date')['customer_unique_id'].nunique().reset_index()
    dau.columns = ['date', 'dau']
    # MAU
    mau = df_merged.groupby('purchase_month')['customer_unique_id'].nunique().reset_index()
    mau['purchase_month'] = mau['purchase_month'].astype(str)
    mau.columns = ['month', 'mau']
    return dau, mau

def calculate_arppu(df_merged):
    # 월별 총 매출 / 월별 구매 고객 수
    monthly_sales = df_merged.groupby('purchase_month')['price'].sum()
    monthly_users = df_merged.groupby('purchase_month')['customer_unique_id'].nunique()
    arppu = (monthly_sales / monthly_users).reset_index()
    arppu.columns = ['month', 'arppu']
    arppu['month'] = arppu['month'].astype(str)
    return arppu

def calculate_retention(df_merged, metric='users', seller_group=None):
    """
    metric: 'users' (고객 고유값 수) or 'sales' (고객 매출액)
    seller_group: 'Top 20%' or 'Bottom 20%'
    """
    df = df_merged.copy()
    if seller_group:
        df = df[df['seller_group'] == seller_group]
        
    if len(df) == 0:
        return None
        
    if metric == 'users':
        cohort_data = df.groupby(['cohort_month', 'cohort_index'])['customer_unique_id'].nunique().reset_index()
        cohort_data.rename(columns={'customer_unique_id': 'value'}, inplace=True)
    else:
        cohort_data = df.groupby(['cohort_month', 'cohort_index'])['price'].sum().reset_index()
        cohort_data.rename(columns={'price': 'value'}, inplace=True)
        
    cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='value')
    # 0개월차(첫 구매) 기준으로 퍼센티지 계산 (이탈률 및 리텐션)
    # 매출의 경우 첫달 매출 대비 몇 %가 유지되는지
    if 0 in cohort_pivot.columns:
        cohort_size = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    else:
        retention_matrix = cohort_pivot
        
    # 인덱스 문자열 변환
    retention_matrix.index = retention_matrix.index.astype(str)
    return retention_matrix
