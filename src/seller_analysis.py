"""
이 파일은 Olist 판매자 성과를 분석하는 스크립트입니다.
판매자를 매출 규모에 따라 상위 20%, 하위 20% 등으로 그룹화하고, 그룹별로 매출, 리뷰 점수, 배송 성과, 가격 구조 등을 비교 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os
import sys
from tabulate import tabulate

# 표준 출력 인코딩을 utf-8로 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 설정
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
image_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(image_dir, exist_ok=True)

def load_data():
    files = {
        'items': 'olist_order_items_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'translation': 'product_category_name_translation.csv'
    }
    dfs = {}
    for name, file in files.items():
        dfs[name] = pd.read_csv(os.path.join(data_dir, file))
    return dfs

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"seller_{name}.png"))
    plt.close()

def run_seller_analysis():
    dfs = load_data()
    
    # 1. 데이터 조인 및 전처리
    items = dfs['items']
    orders = dfs['orders']
    reviews = dfs['reviews']
    products = dfs['products']
    sellers = dfs['sellers']
    trans = dfs['translation']
    
    # 날짜 변환
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])
        
    # 상품 카테고리 영문명 매핑
    products = pd.merge(products, trans, on='product_category_name', how='left')
    products['category'] = products['product_category_name_english'].fillna(products['product_category_name'])
    
    # 통합 데이터프레임 (Items 중심)
    df = pd.merge(items, orders, on='order_id', how='left')
    df = pd.merge(df, reviews[['order_id', 'review_score']], on='order_id', how='left')
    df = pd.merge(df, products[['product_id', 'category']], on='product_id', how='left')
    
    # 2. 셀러별 요약 (seller_summary) 생성
    print("Generating seller_summary...")
    
    # 매출/규모 지표
    seller_sales = df.groupby('seller_id').agg(
        total_sales=('price', 'sum'),
        order_count=('order_id', 'nunique'),
        item_count=('order_id', 'count')
    )
    seller_sales['avg_order_value'] = seller_sales['total_sales'] / seller_sales['order_count']
    
    # 리뷰 지표
    seller_reviews = df.groupby('seller_id').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review_score', 'count')
    )
    
    # 리뷰 비율 (1~2점, 4~5점)
    df['is_low_review'] = df['review_score'].apply(lambda x: 1 if x in [1, 2] else 0)
    df['is_high_review'] = df['review_score'].apply(lambda x: 1 if x in [4, 5] else 0)
    
    review_rates = df.groupby('seller_id').agg(
        low_review_rate=('is_low_review', 'mean'),
        high_review_rate=('is_high_review', 'mean')
    )
    
    # 배송 지표 (delivered 기준)
    del_df = df[(df['order_status'] == 'delivered') & (df['order_delivered_customer_date'].notnull())].copy()
    del_df['is_on_time_delivery'] = del_df['order_delivered_customer_date'] <= del_df['order_estimated_delivery_date']
    del_df['delivery_days'] = (del_df['order_delivered_customer_date'] - del_df['order_purchase_timestamp']).dt.days
    del_df['delay_days'] = (del_df['order_delivered_customer_date'] - del_df['order_estimated_delivery_date']).dt.days
    del_df['delay_days_only'] = del_df['delay_days'].apply(lambda x: x if x > 0 else 0)
    del_df['is_delayed'] = del_df['delay_days'] > 0
    
    seller_delivery = del_df.groupby('seller_id').agg(
        on_time_rate=('is_on_time_delivery', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        avg_delay_days=('delay_days_only', 'mean'),
        delayed_order_rate=('is_delayed', 'mean')
    )
    
    # 판매 아이템 지표
    seller_items = df.groupby('seller_id').agg(
        category_count=('category', 'nunique')
    )
    
    # 메인 카테고리
    cat_sales = df.groupby(['seller_id', 'category'])['price'].sum().reset_index()
    main_cat = cat_sales.sort_values(['seller_id', 'price'], ascending=[True, False]).drop_duplicates('seller_id')
    main_cat = main_cat.rename(columns={'category': 'main_category', 'price': 'main_cat_sales'})
    
    # 가격/배송비 지표
    df['freight_ratio'] = np.where(df['price'] > 0, df['freight_value'] / df['price'], np.nan)
    seller_price = df.groupby('seller_id').agg(
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        avg_freight=('freight_value', 'mean'),
        avg_freight_ratio=('freight_ratio', 'mean')
    )
    
    # 최종 병합
    seller_summary = pd.concat([seller_sales, seller_reviews, review_rates, seller_delivery, seller_items, seller_price], axis=1)
    
    # main_cat 조인 전 인덱스 확인 및 리셋
    seller_summary.index.name = 'seller_id'
    seller_summary = seller_summary.reset_index()
    
    # 중복 제거 (만약의 경우 대비)
    seller_summary = seller_summary.drop_duplicates('seller_id')
    main_cat = main_cat.drop_duplicates('seller_id')
    
    seller_summary = pd.merge(seller_summary, main_cat[['seller_id', 'main_category', 'main_cat_sales']], on='seller_id', how='left')
    seller_summary['main_category_share'] = seller_summary['main_cat_sales'] / seller_summary['total_sales']
    
    # 인덱스 중복 제거 (seaborn 에러 방지)
    seller_summary = seller_summary.drop_duplicates('seller_id').reset_index(drop=True)
    
    # 3. 셀러 그룹 정의
    seller_summary = seller_summary.sort_values('total_sales', ascending=False)
    n = len(seller_summary)
    top_n = int(n * 0.2)
    
    seller_summary['seller_group'] = 'Middle'
    seller_summary.iloc[:top_n, seller_summary.columns.get_loc('seller_group')] = 'Top 20%'
    seller_summary.iloc[-top_n:, seller_summary.columns.get_loc('seller_group')] = 'Bottom 20%'
    
    # 보조 분석: order_count >= 5
    seller_summary_active = seller_summary[seller_summary['order_count'] >= 5].copy()
    n_act = len(seller_summary_active)
    top_n_act = int(n_act * 0.2)
    seller_summary_active['seller_group_act'] = 'Middle'
    seller_summary_active.iloc[:top_n_act, seller_summary_active.columns.get_loc('seller_group_act')] = 'Top 20% (Act)'
    seller_summary_active.iloc[-top_n_act:, seller_summary_active.columns.get_loc('seller_group_act')] = 'Bottom 20% (Act)'

    # 4. 시각화
    print("Generating plots...")
    
    group_order = ['Top 20%', 'Middle', 'Bottom 20%']
    palette = {'Top 20%': 'royalblue', 'Middle': 'lightgray', 'Bottom 20%': 'indianred'}
    
    # 1. seller_id별 total_sales 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(seller_summary['total_sales'], bins=50, kde=True, log_scale=(True, False))
    plt.title('Seller별 총 매출 분포 (Log Scale)')
    save_plot('1_sales_dist')
    
    # 2. seller_group별 셀러 수 비교
    plt.figure(figsize=(8, 6))
    seller_summary['seller_group'].value_counts().reindex(group_order).plot(kind='bar', color=[palette[g] for g in group_order])
    plt.title('Seller Group별 셀러 수')
    save_plot('2_group_count')
    
    # 3. seller_group별 total_sales 합계 및 기여도
    group_sales = seller_summary.groupby('seller_group')['total_sales'].sum().reindex(group_order)
    plt.figure(figsize=(8, 6))
    group_sales.plot(kind='pie', autopct='%1.1f%%', colors=[palette[g] for g in group_order])
    plt.title('Seller Group별 매출 기여도')
    save_plot('3_sales_contribution')
    
    # 4. 평균 order_count 비교
    plt.figure(figsize=(8, 6))
    sns.barplot(x='seller_group', y='order_count', data=seller_summary, order=group_order, palette=palette)
    plt.title('Seller Group별 평균 주문 수')
    save_plot('4_avg_order_count')
    
    # 5. 평균 item_count 비교
    plt.figure(figsize=(8, 6))
    sns.barplot(x='seller_group', y='item_count', data=seller_summary, order=group_order, palette=palette)
    plt.title('Seller Group별 평균 판매 아이템 수')
    save_plot('5_avg_item_count')
    
    # 6. avg_review_score 비교
    plt.figure(figsize=(8, 6))
    sns.barplot(x='seller_group', y='avg_review_score', data=seller_summary, order=group_order, palette=palette)
    plt.title('Seller Group별 평균 리뷰 점수')
    save_plot('6_avg_review_score')
    
    # 7. low_review_rate / high_review_rate 비교
    rates = seller_summary.groupby('seller_group')[['low_review_rate', 'high_review_rate']].mean().reindex(group_order)
    rates.plot(kind='bar', figsize=(10, 6), color=['orange', 'green'])
    plt.title('Seller Group별 부정/긍정 리뷰 비율')
    save_plot('7_review_rates')
    
    # 8. on_time_rate 비교
    plt.figure(figsize=(8, 6))
    sns.barplot(x='seller_group', y='on_time_rate', data=seller_summary, order=group_order, palette=palette)
    plt.title('Seller Group별 정시 도착률')
    save_plot('8_on_time_rate')
    
    # 9. avg_delivery_days / avg_delay_days 비교
    delivery = seller_summary.groupby('seller_group')[['avg_delivery_days', 'avg_delay_days']].mean().reindex(group_order)
    delivery.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
    plt.title('Seller Group별 배송 소요일 및 지연일')
    save_plot('9_delivery_stats')
    
    # 10. category_count 비교
    plt.figure(figsize=(8, 6))
    sns.barplot(x='seller_group', y='category_count', data=seller_summary, order=group_order, palette=palette)
    plt.title('Seller Group별 판매 카테고리 수')
    save_plot('10_category_count')
    
    # 11. Top 20%와 Bottom 20%의 main_category TOP 10 비교
    top_20_cats = seller_summary[seller_summary['seller_group'] == 'Top 20%']['main_category'].value_counts().head(10)
    bot_20_cats = seller_summary[seller_summary['seller_group'] == 'Bottom 20%']['main_category'].value_counts().head(10)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    top_20_cats.plot(kind='barh', ax=ax[0], color='royalblue')
    ax[0].set_title('Top 20% 셀러 주요 카테고리')
    bot_20_cats.plot(kind='barh', ax=ax[1], color='indianred')
    ax[1].set_title('Bottom 20% 셀러 주요 카테고리')
    save_plot('11_main_categories')
    
    # 12. avg_price / median_price 비교
    prices = seller_summary.groupby('seller_group')[['avg_price', 'median_price']].mean().reindex(group_order)
    prices.plot(kind='bar', figsize=(10, 6), color=['gold', 'darkgoldenrod'])
    plt.title('Seller Group별 평균 및 중앙 가격')
    save_plot('12_price_stats')
    
    # 13. avg_freight / avg_freight_ratio 비교
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    sns.barplot(x='seller_group', y='avg_freight', data=seller_summary, order=group_order, palette=palette, ax=ax1, alpha=0.6)
    sns.lineplot(x='seller_group', y='avg_freight_ratio', data=seller_summary, sort=False, ax=ax2, color='red', marker='o')
    ax1.set_ylabel('Avg Freight Value')
    ax2.set_ylabel('Avg Freight Ratio')
    plt.title('Seller Group별 배송비 및 배송비 비중')
    save_plot('13_freight_stats')
    
    # 14. on_time_rate와 avg_review_score 관계
    plt.figure(figsize=(10, 6))
    plt.scatter(seller_summary['on_time_rate'], seller_summary['avg_review_score'], alpha=0.1)
    plt.xlabel('정시 도착률')
    plt.ylabel('평균 리뷰 점수')
    plt.title('배송 성과와 리뷰 점수 관계')
    save_plot('14_ontime_vs_review')
    
    # 15. avg_freight_ratio와 avg_review_score 관계
    plt.figure(figsize=(10, 6))
    plt.scatter(seller_summary['avg_freight_ratio'], seller_summary['avg_review_score'], alpha=0.1)
    plt.xlabel('평균 배송비 비중')
    plt.ylabel('평균 리뷰 점수')
    plt.title('배송비 비중과 리뷰 점수 관계')
    save_plot('15_freight_vs_review')
    
    # 16. total_sales와 avg_review_score 관계
    plt.figure(figsize=(10, 6))
    plt.scatter(seller_summary['total_sales'], seller_summary['avg_review_score'], alpha=0.1)
    plt.xscale('log')
    plt.xlabel('총 매출 (Log Scale)')
    plt.ylabel('평균 리뷰 점수')
    plt.title('매출 규모와 리뷰 점수 관계')
    save_plot('16_sales_vs_review')

    # 5. 핵심 비교표 데이터 출력
    metrics = [
        'total_sales', 'order_count', 'item_count', 'avg_review_score', 
        'low_review_rate', 'high_review_rate', 'on_time_rate', 
        'avg_delivery_days', 'avg_delay_days', 'category_count', 
        'avg_price', 'median_price', 'avg_freight', 'avg_freight_ratio'
    ]
    
    summary_table = seller_summary.groupby('seller_group')[metrics].mean().reindex(group_order)
    summary_table['seller_count'] = seller_summary['seller_group'].value_counts()
    summary_table['total_group_sales'] = seller_summary.groupby('seller_group')['total_sales'].sum()
    summary_table['sales_contribution'] = summary_table['total_group_sales'] / seller_summary['total_sales'].sum()
    
    print("\n[Summary Table] Seller Group Comparison (All)")
    print(tabulate(summary_table.reset_index(), headers='keys', tablefmt='pipe'))
    
    # 보조 분석 표 (order_count >= 5)
    summary_table_act = seller_summary_active.groupby('seller_group_act')[metrics].mean().reindex(['Top 20% (Act)', 'Middle', 'Bottom 20% (Act)'])
    print("\n[Summary Table] Seller Group Comparison (Active: Order Count >= 5)")
    print(tabulate(summary_table_act.reset_index(), headers='keys', tablefmt='pipe'))

if __name__ == "__main__":
    run_seller_analysis()
