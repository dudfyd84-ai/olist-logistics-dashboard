"""
이 파일은 Olist 데이터셋의 탐색적 데이터 분석(EDA)을 수행하는 스크립트입니다.
주문 상태, 결제 수단, 상품 카테고리, 가격 및 배송비 상관관계, 시계열 추이 등 다양한 지표를 시각화하고 요약 통계를 생성합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate

import sys
import os

# 표준 출력 인코딩을 utf-8로 설정 (Windows cp949 에러 방지)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 설정
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
image_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(image_dir, exist_ok=True)

def load_data():
    files = {
        'customers': 'olist_customers_dataset.csv',
        'items': 'olist_order_items_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
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
    plt.savefig(os.path.join(image_dir, f"{name}.png"))
    plt.close()

def run_eda():
    dfs = load_data()
    
    # 데이터 병합 (분석용)
    orders = dfs['orders']
    items = dfs['items']
    products = dfs['products']
    payments = dfs['payments']
    customers = dfs['customers']
    reviews = dfs['reviews']
    
    # 날짜 변환
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    # 1. 일변량 분석: 주문 상태 분포
    plt.figure(figsize=(10, 6))
    order_status_counts = orders['order_status'].value_counts()
    order_status_counts.plot(kind='bar', color='skyblue')
    plt.title('주문 상태 분포')
    plt.xlabel('상태')
    plt.ylabel('주문 수')
    save_plot('1_order_status_dist')
    
    print("\n[Table 1] 주문 상태 빈도표")
    print(tabulate(order_status_counts.reset_index(), headers=['Status', 'Count'], tablefmt='pipe'))

    # 2. 일변량 분석: 결제 수단 분포
    plt.figure(figsize=(10, 6))
    payment_counts = payments['payment_type'].value_counts()
    payment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('결제 수단 비율')
    save_plot('2_payment_type_dist')
    
    print("\n[Table 2] 결제 수단 빈도표")
    print(tabulate(payment_counts.reset_index(), headers=['Payment Type', 'Count'], tablefmt='pipe'))

    # 3. 일변량 분석: 상품 카테고리 Top 30
    plt.figure(figsize=(12, 8))
    top_categories = products['product_category_name'].value_counts().head(30)
    top_categories.plot(kind='barh', color='salmon')
    plt.title('상위 30개 상품 카테고리')
    plt.gca().invert_yaxis()
    save_plot('3_top_categories')
    
    print("\n[Table 3] 상위 30개 카테고리 통계")
    print(tabulate(top_categories.reset_index(), headers=['Category', 'Count'], tablefmt='pipe'))

    # 4. 이변량 분석: 카테고리별 평균 가격 (Top 10)
    merged_items_products = pd.merge(items, products, on='product_id')
    cat_price = merged_items_products.groupby('product_category_name')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    cat_price['mean'].plot(kind='bar', color='gold')
    plt.title('카테고리별 평균 가격 (상위 10)')
    plt.ylabel('평균 가격')
    save_plot('4_cat_avg_price')
    
    print("\n[Table 4] 카테고리별 평균 가격 및 판매량")
    print(tabulate(cat_price.reset_index(), headers=['Category', 'Avg Price', 'Count'], tablefmt='pipe'))

    # 5. 이변량 분석: 결제 금액과 배송비 상관관계
    plt.figure(figsize=(10, 6))
    plt.scatter(items['price'], items['freight_value'], alpha=0.5, s=1)
    plt.title('상품 가격과 배송비 상관관계')
    plt.xlabel('가격')
    plt.ylabel('배송비')
    save_plot('5_price_vs_freight')
    
    print("\n[Table 5] 가격 및 배송비 상관계수")
    print(tabulate(items[['price', 'freight_value']].corr(), headers='keys', tablefmt='pipe'))

    # 6. 시계열 분석: 월별 주문 수 추이
    orders['month_year'] = orders['order_purchase_timestamp'].dt.to_period('M')
    monthly_orders = orders.groupby('month_year').size()
    
    plt.figure(figsize=(15, 6))
    monthly_orders.plot(marker='o')
    plt.title('월별 주문 수 추이')
    plt.grid(True)
    save_plot('6_monthly_orders')
    
    print("\n[Table 6] 월별 주문 통계")
    print(tabulate(monthly_orders.reset_index(), headers=['Month', 'Order Count'], tablefmt='pipe'))

    # 7. 다변량 분석: 결제 수단별 할부 횟수 및 금액 분포
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='payment_type', y='payment_value', data=payments[payments['payment_value'] < 1000])
    plt.title('결제 수단별 금액 분포 (1000 미만)')
    save_plot('7_payment_value_boxplot')
    
    print("\n[Table 7] 결제 수단별 금액 기술통계")
    print(tabulate(payments.groupby('payment_type')['payment_value'].describe(), headers='keys', tablefmt='pipe'))

    # 8. 이변량 분석: 리뷰 점수 분포
    plt.figure(figsize=(10, 6))
    review_scores = reviews['review_score'].value_counts().sort_index()
    review_scores.plot(kind='bar', color='lightgreen')
    plt.title('리뷰 점수 분포')
    save_plot('8_review_score_dist')
    
    print("\n[Table 8] 리뷰 점수 빈도표")
    print(tabulate(review_scores.reset_index(), headers=['Score', 'Count'], tablefmt='pipe'))

    # 9. 이변량 분석: 주(State)별 고객 수
    plt.figure(figsize=(12, 6))
    state_counts = customers['customer_state'].value_counts()
    state_counts.plot(kind='bar', color='mediumpurple')
    plt.title('주별 고객 분포')
    save_plot('9_state_dist')
    
    print("\n[Table 9] 주별 고객 수")
    print(tabulate(state_counts.reset_index(), headers=['State', 'Customer Count'], tablefmt='pipe'))

    # 10. 텍스트 분석: 리뷰 메시지 키워드 추출 (TF-IDF)
    # 결측치 제거 및 샘플링 (속도 조절)
    review_msgs = reviews['review_comment_message'].dropna().sample(min(5000, len(reviews['review_comment_message'].dropna())), random_state=42)
    
    tfidf = TfidfVectorizer(max_features=30, stop_words=None) # 포르투갈어 불용어 설정 생략 (기본 분석)
    tfidf_matrix = tfidf.fit_transform(review_msgs)
    words = tfidf.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    
    data = []
    for col, word in enumerate(words):
        data.append((word, sums[0, col]))
    
    ranking = pd.DataFrame(data, columns=['word', 'importance']).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='word', data=ranking, palette='viridis')
    plt.title('리뷰 메시지 주요 키워드 Top 30 (TF-IDF)')
    save_plot('10_review_keywords')
    
    print("\n[Table 10] 리뷰 키워드 중요도")
    print(tabulate(ranking, headers='keys', tablefmt='pipe', showindex=False))

    # 추가 시각화 (다변량): 결제 수단별 평균 할부 횟수
    plt.figure(figsize=(10, 6))
    avg_installments = payments.groupby('payment_type')['payment_installments'].mean().sort_values()
    avg_installments.plot(kind='barh', color='teal')
    plt.title('결제 수단별 평균 할부 횟수')
    save_plot('11_avg_installments')
    
    print("\n[Table 11] 결제 수단별 평균 할부 횟수")
    print(tabulate(avg_installments.reset_index(), headers=['Payment Type', 'Avg Installments'], tablefmt='pipe'))

if __name__ == "__main__":
    run_eda()
