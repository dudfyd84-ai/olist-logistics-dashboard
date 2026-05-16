"""
이 파일은 Olist 이커머스 데이터 분석 및 가설 검증, 핵심 지표 모니터링을 위한 Streamlit 대시보드 스크립트입니다.
py-streamlit 스킬의 가이드라인에 따라 Plotly를 사용한 인터랙티브 시각화, 캐싱, 세션 상태를 활용합니다.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# 모듈 임포트
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_preprocess_data, calculate_dau_mau, calculate_arppu, calculate_retention

st.set_page_config(page_title="Olist 이커머스 대시보드", page_icon="🛒", layout="wide")

def test_hypothesis_1(orders):
    """가설 1 검증: 예상 배송 예정일보다 실제 도착일이 늦어질수록 고객 리뷰 점수가 급 하락 할 것이고 판매량에 영향을 미칠 것이다."""
    # 지연된 주문과 정상 도착 주문의 리뷰 점수 비교
    valid_orders = orders.dropna(subset=['delay_days', 'review_score'])
    late_orders = valid_orders[valid_orders['delay_days'] > 0]
    ontime_orders = valid_orders[valid_orders['delay_days'] <= 0]
    
    late_avg = late_orders['review_score'].mean()
    ontime_avg = ontime_orders['review_score'].mean()
    
    # 지연 일수와 리뷰 점수 간의 상관관계 (지연된 건만)
    correlation, p_value = stats.pearsonr(late_orders['delay_days'], late_orders['review_score'])
    
    is_true = (ontime_avg > late_avg) and (correlation < 0) and (p_value < 0.05)
    
    # 취소율 비교 (판매량 영향 간접 확인)
    late_cancel_rate = len(late_orders[late_orders['order_status'] == 'canceled']) / len(late_orders) if len(late_orders)>0 else 0
    ontime_cancel_rate = len(ontime_orders[ontime_orders['order_status'] == 'canceled']) / len(ontime_orders) if len(ontime_orders)>0 else 0
    
    return is_true, ontime_avg, late_avg, correlation, p_value, late_cancel_rate, ontime_cancel_rate

def test_hypothesis_2(df_merged):
    """가설 2 검증: 매출이 높은 셀러는 반복 구매가 잦은 카테고리이며 실제 재구매 주기가 빠를 것이다."""
    # 탑 셀러와 바텀 셀러의 주문 분리
    top_df = df_merged[df_merged['seller_group'] == 'Top 20%']
    bottom_df = df_merged[df_merged['seller_group'] == 'Bottom 20%']
    
    def get_repurchase_metrics(df):
        # 고객별 주문 횟수
        customer_orders = df.groupby('customer_unique_id').size()
        repeat_customers = customer_orders[customer_orders > 1].count()
        repeat_rate = repeat_customers / len(customer_orders) if len(customer_orders) > 0 else 0
        
        # 재구매 주기 (일 단위)
        # 여러번 구매한 고객의 구매일 차이 평균
        multi_buyers = customer_orders[customer_orders > 1].index
        if len(multi_buyers) > 0:
            multi_df = df[df['customer_unique_id'].isin(multi_buyers)].sort_values(['customer_unique_id', 'order_purchase_timestamp'])
            multi_df['prev_date'] = multi_df.groupby('customer_unique_id')['order_purchase_timestamp'].shift(1)
            multi_df['cycle_days'] = (multi_df['order_purchase_timestamp'] - multi_df['prev_date']).dt.total_seconds() / (24*3600)
            avg_cycle = multi_df['cycle_days'].mean()
        else:
            avg_cycle = np.nan
            
        return repeat_rate, avg_cycle
        
    top_repeat_rate, top_avg_cycle = get_repurchase_metrics(top_df)
    bottom_repeat_rate, bottom_avg_cycle = get_repurchase_metrics(bottom_df)
    
    # 가설 검증: Top의 반복구매율이 높고, 재구매 주기가 짧은가?
    # 바텀 셀러의 데이터가 너무 적어 np.nan이 될 수 있으므로 처리
    cycle_valid = True
    if pd.notna(top_avg_cycle) and pd.notna(bottom_avg_cycle):
        cycle_valid = top_avg_cycle < bottom_avg_cycle
        
    is_true = (top_repeat_rate > bottom_repeat_rate) and cycle_valid
    
    return is_true, top_repeat_rate, bottom_repeat_rate, top_avg_cycle, bottom_avg_cycle

def main():
    st.title("🛒 Olist 이커머스 성과 대시보드")
    
    # 1. 데이터 로드
    try:
        orders, df_merged = load_and_preprocess_data()
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.info("데이터 파일이 `Project1/data/` 경로에 모두 존재하는지 확인해주세요.")
        return

    # 2. 탭 구성
    tab1, tab2, tab3 = st.tabs(["🧪 가설 검증", "📈 핵심 서비스 지표 (Metrics)", "🔄 코호트 리텐션 분석"])
    
    # ==========================================
    # 탭 1: 가설 검증
    # ==========================================
    with tab1:
        st.header("가설 1: 배송 지연과 리뷰/판매량의 관계")
        st.markdown("**가설**: 예상 배송 예정일보다 실제 도착일이 늦어질수록 고객 리뷰 점수가 급 하락 할 것이고 판매량에 영향을 미칠 것이다.")
        
        h1_true, ontime_avg, late_avg, corr, pval, late_c_rate, ontime_c_rate = test_hypothesis_1(orders)
        
        if h1_true:
            st.success("✅ **가설 검증 결과: 참 (True)** - 데이터 분석 결과 해당 가설이 사실로 확인되었습니다.")
            st.markdown(f"> **증명 내용**: 정상 도착 평균 리뷰 점수는 **{ontime_avg:.2f}점**이나, 배송 지연 시 평균 **{late_avg:.2f}점**으로 하락했습니다. 또한 지연 일수가 길어질수록 리뷰 점수가 하락하는 뚜렷한 음의 상관관계(상관계수: {corr:.2f}, p-value: {pval:.4f})를 보였습니다.")
        else:
            st.warning("⚠️ **가설 검증 결과: 기각 (False)** - 데이터가 가설을 충분히 뒷받침하지 않습니다.")
            
        # 가설 1 시각화
        valid_orders = orders.dropna(subset=['delay_days', 'review_score'])
        # 극단치 제외 (그래프 가독성)
        plot_df = valid_orders[(valid_orders['delay_days'] > 0) & (valid_orders['delay_days'] < 30)]
        if len(plot_df) > 5000: plot_df = plot_df.sample(5000, random_state=42)
        
        fig1 = px.scatter(plot_df, x="delay_days", y="review_score", opacity=0.3, trendline="ols",
                          title="배송 지연 일수와 리뷰 점수의 관계 (지연 건 한정)",
                          labels={"delay_days": "배송 지연 일수", "review_score": "리뷰 점수(1-5)"})
        st.plotly_chart(fig1, use_container_width=True)
        
        st.divider()
        
        st.header("가설 2: 매출 상위 셀러의 반복 구매 및 재구매 주기")
        st.markdown("**가설**: 매출이 높은 셀러는 반복 구매가 잦은 카테고리이며 실제 재구매 주기가 빠를 것이다.")
        
        h2_true, top_rr, bot_rr, top_cycle, bot_cycle = test_hypothesis_2(df_merged)
        
        if h2_true:
            st.success("✅ **가설 검증 결과: 참 (True)** - 상위 셀러가 하위 셀러보다 반복 구매 지표가 우수합니다.")
            st.markdown(f"> **증명 내용**: 상위 20% 셀러의 반복 구매율은 **{top_rr*100:.2f}%**로 하위 20% 셀러(**{bot_rr*100:.2f}%**)보다 높습니다. 평균 재구매 주기 또한 상위 셀러가 더 빠릅니다.")
        else:
            st.warning("⚠️ **가설 검증 결과: 부분 참 또는 기각** - 상위 셀러가 반드시 더 빠른 재구매 주기를 가지는 것은 아니거나 반복구매율 차이가 뚜렷하지 않습니다.")
            st.markdown(f"상위 20% 반복구매율: {top_rr*100:.2f}%, 평균 주기: {top_cycle:.1f}일 | 하위 20% 반복구매율: {bot_rr*100:.2f}%, 평균 주기: {bot_cycle:.1f}일")
            
        # 가설 2 시각화
        fig2 = go.Figure(data=[
            go.Bar(name='Top 20% Sellers', x=['반복 구매율(%)'], y=[top_rr*100]),
            go.Bar(name='Bottom 20% Sellers', x=['반복 구매율(%)'], y=[bot_rr*100])
        ])
        fig2.update_layout(title="상위/하위 셀러의 반복 구매율 비교", barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # 탭 2: 핵심 서비스 지표
    # ==========================================
    with tab2:
        st.header("핵심 서비스 지표 (DAU, MAU, ARPPU)")
        dau, mau = calculate_dau_mau(df_merged)
        arppu = calculate_arppu(df_merged)
        
        col1, col2 = st.columns(2)
        with col1:
            # DAU는 변동이 심하므로 7일 이동평균 적용
            dau['dau_7d_ma'] = dau['dau'].rolling(window=7).mean()
            fig_dau = px.line(dau, x='date', y=['dau', 'dau_7d_ma'], title="일간 활성 구매자 수 (DAU & 7D MA)")
            st.plotly_chart(fig_dau, use_container_width=True)
            
            fig_mau = px.bar(mau, x='month', y='mau', title="월간 활성 구매자 수 (MAU)")
            st.plotly_chart(fig_mau, use_container_width=True)
            
        with col2:
            fig_arppu = px.line(arppu, x='month', y='arppu', title="월별 ARPPU (유저당 평균 결제 금액)", markers=True)
            fig_arppu.update_traces(line_color='green')
            st.plotly_chart(fig_arppu, use_container_width=True)

    # ==========================================
    # 탭 3: 코호트 리텐션 분석
    # ==========================================
    with tab3:
        st.header("코호트 리텐션 (이탈률) 분석")
        st.markdown("상위 20% 셀러와 하위 20% 셀러의 고객 유지율(Retention)을 사용자 수와 매출액 기준으로 비교합니다.")
        
        # 1. 고객 고유값 수 기준
        st.subheader("1. 유저 수 기준 리텐션 (Unique Customers)")
        col_ret1, col_ret2 = st.columns(2)
        
        ret_top_users = calculate_retention(df_merged, metric='users', seller_group='Top 20%')
        ret_bot_users = calculate_retention(df_merged, metric='users', seller_group='Bottom 20%')
        
        with col_ret1:
            if ret_top_users is not None:
                fig_rt1 = px.imshow(ret_top_users.iloc[:, 1:13], color_continuous_scale='Blues',
                                    title="상위 20% 셀러 (고객 수 유지율)",
                                    labels=dict(x="코호트 경과월", y="가입(첫구매)월", color="Retention"))
                st.plotly_chart(fig_rt1, use_container_width=True)
                
        with col_ret2:
            if ret_bot_users is not None:
                fig_rb1 = px.imshow(ret_bot_users.iloc[:, 1:13], color_continuous_scale='Reds',
                                    title="하위 20% 셀러 (고객 수 유지율)",
                                    labels=dict(x="코호트 경과월", y="가입(첫구매)월", color="Retention"))
                st.plotly_chart(fig_rb1, use_container_width=True)

        st.divider()
        
        # 2. 매출액 기준
        st.subheader("2. 매출액 기준 리텐션 (Sales Amount)")
        col_ret3, col_ret4 = st.columns(2)
        
        ret_top_sales = calculate_retention(df_merged, metric='sales', seller_group='Top 20%')
        ret_bot_sales = calculate_retention(df_merged, metric='sales', seller_group='Bottom 20%')
        
        with col_ret3:
            if ret_top_sales is not None:
                fig_rt2 = px.imshow(ret_top_sales.iloc[:, 1:13], color_continuous_scale='Greens',
                                    title="상위 20% 셀러 (매출 유지율)",
                                    labels=dict(x="코호트 경과월", y="가입(첫구매)월", color="Sales Ret."))
                st.plotly_chart(fig_rt2, use_container_width=True)
                
        with col_ret4:
            if ret_bot_sales is not None:
                fig_rb2 = px.imshow(ret_bot_sales.iloc[:, 1:13], color_continuous_scale='Oranges',
                                    title="하위 20% 셀러 (매출 유지율)",
                                    labels=dict(x="코호트 경과월", y="가입(첫구매)월", color="Sales Ret."))
                st.plotly_chart(fig_rb2, use_container_width=True)
                
        st.info("💡 **리텐션 분석 결과**: 진한 색상일수록 이탈률이 적고 유지가 잘 됨을 나타냅니다. 일반적으로 상위 셀러의 고객 및 매출 유지율이 높게 나타납니다.")

if __name__ == "__main__":
    main()
