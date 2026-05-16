"""
가설 해결방안 검증을 위한 데이터 시뮬레이션 및 시각화 이미지 생성 스크립트.
기술 통계, 피벗 테이블 텍스트 추출 및 matplotlib/seaborn 그래프를 이미지로 저장합니다.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as plt_sns
import matplotlib.font_manager as fm

# 한글 폰트 셋팅 (윈도우 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    data_dir = "c:/Users/테오/Documents/icb10/Project1/data"
    img_dir = "c:/Users/테오/Documents/icb10/Project1/images/solutions"
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Load Data
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    
    date_cols = ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])
        
    orders['delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.total_seconds() / (24*3600)
    orders['is_late'] = orders['delay_days'] > 0
    orders['delivery_time_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / (24*3600)
    
    df = order_items.merge(orders[['order_id', 'delay_days', 'is_late', 'delivery_time_days', 'customer_id', 'order_estimated_delivery_date', 'order_delivered_customer_date']], on='order_id', how='left')
    df = df.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')
    
    # 상하위 셀러 분리 (Analysis 1)
    seller_sales = df.groupby('seller_id')['price'].sum().sort_values(ascending=False)
    top_20 = set(seller_sales.head(int(len(seller_sales)*0.2)).index)
    bottom_20 = set(seller_sales.tail(int(len(seller_sales)*0.8)).index)
    
    def get_group(x):
        if x in top_20: return 'Top 20%'
        elif x in bottom_20: return 'Bottom 20%'
        else: return 'Middle'
    df['seller_group'] = df['seller_id'].apply(get_group)
    
    output_text = []

    # ==========================================
    # Solution 1: Olist-Prime 보조금 시뮬레이션
    # ==========================================
    df['freight_ratio_before'] = (df['freight_value'] / df['price']) * 100
    df.loc[df['freight_ratio_before'] == np.inf, 'freight_ratio_before'] = np.nan
    
    # 하위 셀러 대상 20% 배송비 보조
    df['freight_after'] = np.where(df['seller_group'] == 'Bottom 20%', df['freight_value'] * 0.8, df['freight_value'])
    df['freight_ratio_after'] = (df['freight_after'] / df['price']) * 100
    
    pivot_s1 = df[df['seller_group'].isin(['Top 20%', 'Bottom 20%'])].groupby('seller_group')[['freight_ratio_before', 'freight_ratio_after']].mean().round(2)
    output_text.append("=== Solution 1: Freight Subsidy Pivot Table ===")
    output_text.append(pivot_s1.to_markdown())
    output_text.append("\n")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_s1.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
    plt.title("하위 20% 셀러 배송비 보조 시뮬레이션 (운임 비중 감소)")
    plt.ylabel("평균 운임 비중 (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "sol1_freight_subsidy.png"))
    plt.close()

    # ==========================================
    # Solution 2: 예측 배송일 보수화 산정 (+3일 Buffer)
    # ==========================================
    # 북부/북동부 주: AM, RR, AP, PA, TO, RO, AC, MA, PI, CE, RN, PB, PE, AL, SE, BA
    north_states = ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC', 'MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'AL', 'SE', 'BA']
    df_north = df[df['customer_state'].isin(north_states)].copy()
    
    before_late_rate = df_north['is_late'].mean() * 100
    
    # 예측일에 3일을 더함
    df_north['new_estimated_date'] = df_north['order_estimated_delivery_date'] + pd.Timedelta(days=3)
    df_north['new_delay_days'] = (df_north['order_delivered_customer_date'] - df_north['new_estimated_date']).dt.total_seconds() / (24*3600)
    df_north['new_is_late'] = df_north['new_delay_days'] > 0
    
    after_late_rate = df_north['new_is_late'].mean() * 100
    
    desc_s2 = pd.DataFrame({
        "구분": ["도입 전 (현재 예상일 기준)", "도입 후 (+3일 Buffer 적용)"],
        "북부/북동부 지역 지연율 (%)": [round(before_late_rate, 2), round(after_late_rate, 2)],
        "지연 감소 효과 (%p)": ["-", f"{round(before_late_rate - after_late_rate, 2)}%p 감소"]
    })
    output_text.append("=== Solution 2: Buffer Date Descriptive Stats ===")
    output_text.append(desc_s2.to_markdown(index=False))
    output_text.append("\n")
    
    plt.figure(figsize=(7, 5))
    plt_sns.barplot(data=desc_s2, x="구분", y="북부/북동부 지역 지연율 (%)", hue="구분", palette=['#e74c3c', '#2ecc71'], legend=False)
    plt.title("북부/북동부 예상 배송일 보수화 적용에 따른 지연율 감소")
    plt.ylim(0, before_late_rate + 10)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "sol2_delay_reduction.png"))
    plt.close()

    # ==========================================
    # Solution 3: 물류 거점 분산 타당성 검토
    # ==========================================
    # SP(상파울루)와 주요 북부/북동부 거점(AM, CE, BA, PE) 배송 기간 비교
    target_states = ['SP', 'AM', 'CE', 'BA', 'PE']
    df_scm = df.dropna(subset=['delivery_time_days']).copy()
    df_scm = df_scm[df_scm['customer_state'].isin(target_states)]
    # 극단치 제외
    df_scm = df_scm[df_scm['delivery_time_days'] < 50]
    
    pivot_s3 = df_scm.groupby('customer_state')['delivery_time_days'].describe()[['count', 'mean', '50%', 'max']].round(1)
    output_text.append("=== Solution 3: Regional Delivery Time (Boxplot Base) ===")
    output_text.append(pivot_s3.to_markdown())
    output_text.append("\n")
    
    plt.figure(figsize=(9, 6))
    plt_sns.boxplot(data=df_scm, x='customer_state', y='delivery_time_days', order=target_states, palette='Set2')
    plt.title("SP 및 북동부 주요 권역별 실제 배송 소요 기간 (리드타임)")
    plt.xlabel("고객 거주 주 (State)")
    plt.ylabel("배송 소요 일수 (Days)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "sol3_regional_boxplot.png"))
    plt.close()

    # Write output to text
    with open("c:/Users/테오/Documents/icb10/Project1/report/solution_stats.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(output_text))
        
    print("성공적으로 시뮬레이션을 완료하고 이미지를 생성했습니다.")

if __name__ == "__main__":
    main()
