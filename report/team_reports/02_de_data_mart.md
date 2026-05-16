# [DE] 데이터 전처리 및 데이터 마트 구축 리포트

**작성자**: 데이터 엔지니어 (DE)
**주제**: 9개 분산 테이블의 효율적 병합 및 가설 검증용 파생 변수 구축

## 1. ERD (개체-관계 다이어그램) 분석 전략
Olist 데이터셋은 관계형 데이터베이스 구조로 되어 있으며, 이를 효과적으로 다루기 위한 마스터 마트를 설계합니다.
- `orders` 테이블을 중심으로 `order_items`를 1:N으로 Join하여 상품 및 가격/배송비 정보 확보.
- `order_reviews`를 조인하여 주문별 평점 매핑.
- `customers`와 `sellers`를 조인하여 구매자와 판매자의 지리적(Geolocation) 위경도 좌표 및 주(State) 데이터 결합.

## 2. 핵심 파생 변수 (Derived Features) 정의
가설 검증과 도메인 분석을 지원하기 위해 다음의 파생 변수를 마트 내에 계산하여 적재합니다.

1. **배송 지연 일수 (`delay_days`)**
   - 수식: `order_delivered_customer_date` - `order_estimated_delivery_date`
   - 양수(+)일 경우 지연, 음수(-)일 경우 조기 도착으로 정의.
2. **총 결제 금액 (`total_order_value`)**
   - 수식: `price` + `freight_value` (또는 `order_payments` 테이블의 `payment_value` 합산)
3. **셀러 매출 그룹 (`seller_tier`)**
   - 누적 매출(price 합산) 기준 정렬 후 상위 20% (Top), 중위 60% (Mid), 하위 20% (Bottom) 라벨링.
4. **리텐션 코호트 (`cohort_month`)**
   - `customer_unique_id` 기준 최초 구매가 발생한 년/월.

## 3. 결측치 및 이상치 처리 규칙
- `order_delivered_customer_date`가 NULL인 건은 배송 중이거나 유실된 건으로 간주하여 지연 검증 대상에서 제외(Drop).
- `review_score`가 없는 건은 가설 검증 단계에서 노이즈가 될 수 있으므로 제외 처리.
- 배송 지연일이 비정상적으로 긴 건(예: 100일 초과)은 시스템 오류나 특별한 예외 케이스이므로 상위 1% 극단치 윈저라이징(Winsorizing) 또는 필터링 적용.
