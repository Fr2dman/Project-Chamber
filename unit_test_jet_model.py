from simulator.physics import JetModel

def test_jet_model():
    """외부 슬롯 각도에 따른 순환 패턴 테스트"""
    
    model = JetModel(num_zones=4)
    
    # 테스트 케이스 1: 수평 토출 (short-circuit)
    print("=" * 50)
    print("테스트 1: 수평 토출 (0도)")
    Q1, info1 = model.get_flow_matrix(
        fan_rpms_S=[3000, 3000, 3000, 3000],
        fan_rpms_L=2000,
        theta_int=[30, 30, 30, 30],  # 내부 슬롯 열림
        theta_ext=[0, 0, 0, 0]       # 수평 토출
    )
    print(f"재순환 비율: {info1['recirculation_ratio']}")
    print(f"혼합 계수: {info1['mixing_factor']}")
    print(f"Q_recirc:\n{info1['Q_recirc']}")
    
    # 테스트 케이스 2: 수직 토출 (전체 혼합)
    print("\n" + "=" * 50)
    print("테스트 2: 수직 토출 (80도)")
    Q2, info2 = model.get_flow_matrix(
        fan_rpms_S=[3000, 3000, 3000, 3000],
        fan_rpms_L=2000,
        theta_int=[30, 30, 30, 30],  # 내부 슬롯 열림
        theta_ext=[80, 80, 80, 80]   # 수직 토출
    )
    print(f"재순환 비율: {info2['recirculation_ratio']}")
    print(f"혼합 계수: {info2['mixing_factor']}")
    print(f"Q_mix:\n{info2['Q_mix']}")
    
    # 테스트 케이스 3: 혼합 각도
    print("\n" + "=" * 50)
    print("테스트 3: 다양한 각도")
    Q3, info3 = model.get_flow_matrix(
        fan_rpms_S=[3000, 3000, 3000, 3000],
        fan_rpms_L=2000,
        theta_int=[45, 30, 30, 45],  # 내부 슬롯 차등
        theta_ext=[0, 30, 60, 80]    # 외부 슬롯 차등
    )
    print(f"재순환 비율: {info3['recirculation_ratio']}")
    print(f"혼합 계수: {info3['mixing_factor']}")
    print(f"최종 유량 행렬:\n{Q3}")

if __name__ == "__main__":
    test_jet_model()