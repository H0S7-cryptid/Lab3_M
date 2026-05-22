"""
Тест конвергенции модели по шагу dt.

Если модель правильно дискретизирована, то при уменьшении dt
результаты должны сходиться к пределу.
"""

from production_simulator import ProductionSimulator
import numpy as np

def test_convergence():
    """Проверить конвергенцию по dt."""
    
    print("=" * 70)
    print("ТЕСТ КОНВЕРГЕНЦИИ ПО ШАГУ ИНТЕГРИРОВАНИЯ dt")
    print("=" * 70)
    print()
    
    # Фиксированное физическое время моделирования
    T_physical = 100  # физические единицы (скажем, дни)
    
    results = {}
    
    for dt in [1.0, 0.5, 0.25, 0.1]:
        # При каждом dt считаем шаги так, чтобы пройти одинаковое время
        steps = int(T_physical / dt)
        
        sim = ProductionSimulator()
        result = sim.simulate(T=T_physical, dt=dt, seed=42)
        
        # Извлечём ключевые метрики
        G_final = result['G'][-1]
        y_A_mean = np.mean(result['y_A'])
        y_B_mean = np.mean(result['y_B'])
        downtime_A = np.sum(result['fA'] == 0) / len(result['fA']) if len(result['fA']) > 0 else 0
        
        results[dt] = {
            'steps': steps,
            'G_final': G_final,
            'y_A_mean': y_A_mean,
            'y_B_mean': y_B_mean,
            'downtime_A_frac': downtime_A
        }
        
        print(f"dt = {dt:4.2f} | steps = {steps:5d} | " +
              f"G_final = {G_final:8.2f} | " +
              f"<Y_A> = {y_A_mean:7.2f} | " +
              f"downtime_A% = {downtime_A*100:5.1f}%")
    
    print()
    print("АНАЛИЗ КОНВЕРГЕНЦИИ:")
    print("-" * 70)
    
    # Проверить, сходятся ли результаты
    dt_vals = sorted(results.keys())
    G_vals = [results[dt]['G_final'] for dt in dt_vals]
    
    # Вычислить относительное изменение
    print("\nОтносительное изменение G_final при уменьшении dt:")
    for i in range(len(dt_vals) - 1):
        dt1, dt2 = dt_vals[i], dt_vals[i+1]
        G1, G2 = G_vals[i], G_vals[i+1]
        rel_change = abs(G2 - G1) / G1 * 100
        print(f"  dt {dt1} → {dt2}: ΔG = {rel_change:6.2f}%")
    
    # Проверить скорость конвергенции
    print("\nФактор конвергенции (наклон log-log):")
    log_dt = np.log([dt_vals[i] / dt_vals[i+1] for i in range(len(dt_vals)-1)])
    log_G = np.log([abs(G_vals[i+1] - G_vals[i]) / G_vals[i] 
                    for i in range(len(G_vals)-1)])
    
    if len(log_dt) > 0:
        slope = np.mean(log_G / log_dt)
        print(f"  Средний наклон: {slope:.2f}")
        print(f"  (1.0 = линейная конвергенция, 2.0 = квадратичная)")
    
    # ДИАГНОСТИКА
    print()
    print("=" * 70)
    print("ДИАГНОСТИКА:")
    print("=" * 70)
    
    if G_vals[-1] < G_vals[0]:
        print("⚠ СТРАННО: G_final УМЕНЬШАЕТСЯ при уменьшении dt")
        print("  Это может означать нестабильность или ошибку в модели!")
    else:
        rel_change_total = abs(G_vals[-1] - G_vals[0]) / G_vals[0] * 100
        if rel_change_total < 5:
            print("✓ ХОРОШО: Модель быстро сходится (<5% изменения)")
            print("  При dt < 0.1 результаты стабильны")
        elif rel_change_total < 20:
            print("⚠ ОСТОРОЖНО: Конвергенция медленная (5-20% изменения)")
            print("  Рекомендуется использовать dt ≤ 0.1")
        else:
            print("✗ ПРОБЛЕМА: Модель не сходится (>20% изменения)")
            print("  Может быть ошибка в дискретизации или методе Эйлера неподходящ")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    test_convergence()
