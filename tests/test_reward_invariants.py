from simulator.environment import AdvancedSmartACSimulator
def test_progress_sign_cold():
    sim = AdvancedSmartACSimulator(num_zones=4)
    sim.reset()
    sim.update_tsv([-2,-2,-2,-2])
    # 냉각 약화(예시): 펠티어/팬 -1
    a = [-1]+[0]*8+[-1,-1,-1,-1]+[-1]
    _, r1, _, _ = sim.step(a)
    # 냉각 강화: +1
    b = [+1]+[0]*8+[+1,+1,+1,+1]+[+1]
    _, r2, _, _ = sim.step(b)
    assert r1 > r2  # 추울 때 약화가 더 높은 보상이어야 함
