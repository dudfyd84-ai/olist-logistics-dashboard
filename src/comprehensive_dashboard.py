"""
이 파일은 물류 최적화 및 고객 이탈 방지 전략을 모니터링하기 위한 종합 비즈니스 대시보드 스크립트입니다.
PM, DA, SCM, 마케팅 파트의 6명 팀원 분석 결과를 4개의 탭에 시각적으로 통합 제공합니다.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader_v2 import load_and_preprocess_data, calculate_dau_mau, calculate_arppu, calculate_retention

st.set_page_config(page_title="Olist 물류 최적화 대시보드", page_icon="🚚", layout="wide")

def main():
    st.title("🚚 물류 최적화 및 고객 이탈 방지 전략 통합 대시보드")
    st.markdown("본 대시보드는 **Olist 하위 셀러의 물류 병목 원인 진단과 수익성(객단가/리텐션) 방어 전략**을 실시간으로 모니터링합니다.")
    
    try:
        orders, df_merged = load_and_preprocess_data()
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 종합 상황판 (PM 전략)", 
        "🧪 가설 검증 (배송 지연 vs 평점)", 
        "🗺️ 물류 SCM 현황 (지도)", 
        "💰 수익성 및 마케팅 분석",
        "💡 해결방안 데이터 시뮬레이션"
    ])
    
    # ==================================================
    # 탭 1: 종합 상황판
    # ==================================================
    with tab1:
        st.header("[PM] 비즈니스 전략 핵심 KPI")
        st.markdown("**목표**: Olist-Prime 물류 보조금 도입 및 거점 분산을 통한 고객 이탈 방어")
        
        col1, col2, col3, col4 = st.columns(4)
        total_orders = len(df_merged)
        late_ratio = len(orders[orders['delay_days'] > 0]) / len(orders.dropna(subset=['delay_days'])) * 100
        avg_freight_bot = df_merged[df_merged['seller_group'] == 'Bottom 20%']['freight_value'].mean()
        avg_freight_top = df_merged[df_merged['seller_group'] == 'Top 20%']['freight_value'].mean()
        
        with col1: st.metric("총 결제 건수", f"{total_orders:,.0f}건")
        with col2: st.metric("전체 배송 지연율", f"{late_ratio:.1f}%")
        with col3: st.metric("상위 20% 평균 운임", f"R$ {avg_freight_top:.2f}")
        with col4: st.metric("하위 20% 평균 운임", f"R$ {avg_freight_bot:.2f}", delta=f"+R$ {avg_freight_bot - avg_freight_top:.2f}", delta_color="inverse")
            
        dau, mau = calculate_dau_mau(df_merged)
        dau['dau_7d_ma'] = dau['dau'].rolling(window=7).mean()
        
        st.subheader("일간 활성 결제 유저 수 (DAU)")
        fig_dau = px.line(dau, x='date', y=['dau', 'dau_7d_ma'], title="DAU 추이 및 7일 이동평균", color_discrete_sequence=['#95a5a6', '#e74c3c'])
        st.plotly_chart(fig_dau, use_container_width=True)

    # ==================================================
    # 탭 2: 가설 검증 통계 (DA)
    # ==================================================
    with tab2:
        st.header("[DA] 가설 검증: 배송 지연이 평점에 미치는 파괴적 영향")
        
        valid_orders = orders.dropna(subset=['delay_days', 'review_score'])
        late_orders = valid_orders[valid_orders['delay_days'] > 0]
        ontime_orders = valid_orders[valid_orders['delay_days'] <= 0]
        
        late_avg = late_orders['review_score'].mean()
        ontime_avg = ontime_orders['review_score'].mean()
        correlation, p_value = stats.pearsonr(late_orders['delay_days'], late_orders['review_score'])
        
        if (ontime_avg > late_avg) and (correlation < 0) and (p_value < 0.05):
            st.success("✅ **검증 결과: 참 (True)** - 배송이 지연될수록 고객 리뷰 점수가 선형적으로 하락하는 것이 통계적으로 유의미하게 입증되었습니다.")
            st.markdown(f"**정상 배송 평균 평점**: {ontime_avg:.2f}점 🆚 **지연 배송 평균 평점**: {late_avg:.2f}점 (상관계수: {correlation:.3f})")
        
        plot_df = valid_orders[(valid_orders['delay_days'] > 0) & (valid_orders['delay_days'] < 20)].sample(min(5000, len(late_orders)), random_state=42)
        fig_hyp = px.scatter(plot_df, x="delay_days", y="review_score", opacity=0.2, trendline="ols",
                             title="배송 지연 일수에 따른 리뷰 평점 하락 추세선", trendline_color_override="red")
        st.plotly_chart(fig_hyp, use_container_width=True)

    # ==================================================
    # 탭 3: 물류 SCM 현황 지도 (SCM)
    # ==================================================
    with tab3:
        st.header("[SCM] 지역별 물류 병목 현황 및 지연 분포")
        st.markdown("남동부에 집중된 셀러 인프라로 인해 발생한 타 권역(북부, 북동부)의 배송 지연 밀집도를 지도에 시각화합니다.")
        
        # 맵 렌더링 최적화를 위해 일부 지연 건만 샘플링
        geo_df = df_merged[df_merged['delay_days'] > 3].dropna(subset=['customer_lat', 'customer_lng'])
        if len(geo_df) > 3000:
            geo_df = geo_df.sample(3000, random_state=42)
            
        if not geo_df.empty:
            fig_map = px.scatter_mapbox(
                geo_df, lat="customer_lat", lon="customer_lng", color="delay_days", size="delay_days",
                color_continuous_scale=px.colors.sequential.YlOrRd, size_max=15, zoom=3,
                mapbox_style="carto-positron", title="3일 이상 배송 지연 발생 고객의 지리적 분포 (샘플링)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("시각화할 지연 배송 위경도 데이터가 부족합니다.")

    # ==================================================
    # 탭 4: 수익성 및 마케팅 분석
    # ==================================================
    with tab4:
        st.header("[Marketing] 상위/하위 셀러의 락인 효과 및 리텐션")
        st.markdown("수익성(AOV) 유지 및 이탈 방지를 위해 상위 셀러와 하위 셀러의 고객 유지율(Retention)을 매출 기준으로 비교합니다.")
        
        col_ret1, col_ret2 = st.columns(2)
        ret_top_sales = calculate_retention(df_merged, metric='sales', seller_group='Top 20%')
        ret_bot_sales = calculate_retention(df_merged, metric='sales', seller_group='Bottom 20%')
        
        with col_ret1:
            if ret_top_sales is not None:
                fig_rt2 = px.imshow(ret_top_sales.iloc[:, 1:13], color_continuous_scale='Greens', title="[Top 20% 셀러] 월간 매출 코호트 유지율")
                st.plotly_chart(fig_rt2, use_container_width=True)
                
        with col_ret2:
            if ret_bot_sales is not None:
                fig_rb2 = px.imshow(ret_bot_sales.iloc[:, 1:13], color_continuous_scale='Oranges', title="[Bottom 20% 셀러] 월간 매출 코호트 유지율")
                st.plotly_chart(fig_rb2, use_container_width=True)
        
        st.divider()
        st.subheader("객단가 (ARPPU) 추이")
        arppu = calculate_arppu(df_merged)
        fig_arppu = px.line(arppu, x='month', y='arppu', title="월별 평균 결제액(ARPPU) 추이", markers=True)
        st.plotly_chart(fig_arppu, use_container_width=True)

    # ==================================================
    # 탭 5: 해결방안 시뮬레이션
    # ==================================================
    with tab5:
        st.header("💡 데이터 기반 해결방안 및 시뮬레이션 검증")
        st.markdown("가설 검증 결과를 바탕으로 도출된 3가지 해결방안의 실제 효과를 Olist 데이터를 통해 검증했습니다.")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, "..", "images", "solutions")
        
        col_sol1, col_sol2 = st.columns(2)
        
        with col_sol1:
            st.subheader("1. Olist-Prime 운임 보조금 효과")
            st.markdown("**목적**: 운임 비중이 과도한 하위 20% 셀러를 위한 배송비 20% 플랫폼 보조")
            st.markdown("- **Top 20% 운임 비중**: 29.20%\n- **Bottom 20% 운임 비중(전)**: 41.59%\n- **Bottom 20% 운임 비중(후)**: **33.27%**")
            st.image(os.path.join(img_dir, "sol1_freight_subsidy.png"), use_container_width=True)
            
        with col_sol2:
            st.subheader("2. 예상 배송일(Estimated Date) 보수화 산정")
            st.markdown("**목적**: 북부/북동부 지역 예측 배송일에 +3일 여유(Buffer) 부여 시 지연율 감소")
            st.markdown("- **기존 지연율**: 13.04%\n- **Buffer 적용 후**: **9.80% (3.25%p 즉각 감소)**")
            st.image(os.path.join(img_dir, "sol2_delay_reduction.png"), use_container_width=True)
            
        st.divider()
        st.subheader("3. 물류 거점 분산(Fulfillment) 인프라 타당성 검토")
        st.markdown("**분석**: 상파울루(SP) 내부 배송 리드타임 대비 주요 북동부 지역의 실제 소요 시간 비교")
        st.markdown("- **SP 평균 리드타임**: 8.6일 (중앙값 7.2일)\n- **AM/CE 등 외곽 평균 리드타임**: 18~25일 (최대 3.6배 격차)")
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(os.path.join(img_dir, "sol3_regional_boxplot.png"), use_container_width=True)

if __name__ == "__main__":
    main()
