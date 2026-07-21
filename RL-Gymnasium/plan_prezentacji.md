# Plan prezentacji – Projekt 3 (RL, Gymnasium MountainCar-v0)
Rola: Osoba E – metryki, integracja wyników A–D, wykresy/tabele, wnioski, prezentacja końcowa

## Uwagi przed edycją
- PPO (Osoba D) – brak logów/wykresów w obecnym eksporcie; slajd 7 to placeholder do uzupełnienia.

- Tabela zbiorcza i heatmapa są wymagane w podziale ról, ale jeszcze nie istnieją jako pliki – do przygotowania.
- `plots/` i `wyniki/plots/` mają te same 2 pliki shaping – w planie zostaje `wyniki/plots/` jako jedyne źródło.
- `wyniki/plots/gamma|epsilon_decay/Eksperyment_*` oraz `wpływ hiperparametrów/*_comparison_variant_*` to dwa cięcia tych samych danych (po eksperymencie vs po wariancie) – wybrać jedno
- W `Eksperyment_1` i `Eksperyment_2` (epsilon_decay) są po 2 pliki (z i bez sufiksu `e_d_xxxx`) – sprawdzić, który jest aktualny.

## 1. Slajd tytułowy
- Tytuł projektu, środowisko MountainCar

## 2. Cel i pytania badawcze
- Cel: poprawa agenta RL względem baseline
- Pytania (z prezentacji wstępnej – do potwierdzenia, że nadal aktualne):
  - Który element daje największą poprawę jakości uczenia?
  - Czy bardziej złożony algorytm zawsze wygrywa?
  - Czy tuning jest ważniejszy niż wybór metody?

## 3. Metodologia oceny
- Nagroda treningowa (kolumna `reward` w CSV) jest **shaped** – zawiera bonus z `reward_wrappers.py`, inny wzór dla każdego wariantu → wartości shaped nieporównywalne między wariantami (potwierdzone w danych, zob. pkt 9).
- Nagroda ewaluacyjna (`evaluate()` na nieowiniętym env, bez eksploracji) jest surowa i porównywalna – na niej oparte są wykresy `eval_mean_all_variants_vs_*`.
- Metryki:
  - średnia nagroda ewaluacyjna – porównanie między wariantami/metodami
  - stabilność = odchylenie std nagrody (ostatnie 100 epizodów treningu)
  - szybkość uczenia = epizod, w którym `avg_100` osiąga maksimum
  - czas treningu – nie jest zapisywany w `utils/training.py`/logach, do osobnego pomiaru

## 4. Baseline 
`wyniki/plots/baseline/`
- DQN_MountainCar-v0.png
- REINFORCE_MountainCar-v0.png

## 5. Tuning hiperparametrów
`wyniki/plots/`
- eval_mean_all_variants_vs_gamma.png
- eval_mean_all_variants_vs_epsilon_decay.png

`wyniki/plots/gamma/`
- Eksperyment_1
- Eksperyment_2
- Eksperyment_3
- Eksperyment_4
- Eksperyment_5

`wyniki/plots/epsilon_decay/`
- Eksperyment_0
- Eksperyment_1
- Eksperyment_2
- Eksperyment_3
- Eksperyment_4
- Eksperyment_5

`wpływ hiperparametrów/`
- gamma_comparison_MountainCar-v0_variant_A
- gamma_comparison_MountainCar-v0_variant_B
- gamma_comparison_MountainCar-v0_variant_C
- gamma_comparison_MountainCar-v0_variant_D
- gamma_comparison_MountainCar-v0_variant_E
- epsilon_decay_comparison_MountainCar-v0_variant_A
- epsilon_decay_comparison_MountainCar-v0_variant_B
- epsilon_decay_comparison_MountainCar-v0_variant_C
- epsilon_decay_comparison_MountainCar-v0_variant_D
- epsilon_decay_comparison_MountainCar-v0_variant_E

## 6. Reward shaping 
`wyniki/plots/`
- DQN_variant_A_..._E_MountainCar-v0_DQN_shaping.png
- REINFORCE_variant_A_..._E_MountainCar-v0_REINFORCE_shaping.png

`filmiki reward shaping i baseline/`
- all_policies.png
- best_variants_comparison.png
- dqn_baseline.mp4
- mountaincar_variant_dqn_A_best.mp4
- mountaincar_variant_dqn_B_best.mp4
- mountaincar_variant_dqn_C_best.mp4
- mountaincar_dqn_variant_D_best.mp4
- mountaincar_dqn_variant_D_bad.mp4
- mountaincar_dqn_variant_E_best.mp4
- mountaincar_dqn_variant_E_worst.mp4

## 7. Zaawansowany algorytm (PPO) – Osoba D
- [placeholder – uzupełnić po otrzymaniu wyników]

## 8. Demonstracja – animacje
`animacje/`
- 02_dqn_reward_shaping_race.gif
- 03_reinforce_reward_shaping_race.gif
- 04_dqn_variant_D_noise_vs_trend.gif
- 05_car_on_hill_progression.gif
- 06_car_before_after.gif

## 9. Tabela zbiorcza i heatmapa

**Tabela – reward shaping, metryki treningowe (shaped, z `results/logs/*.csv`)**

| Seria | Śr. nagroda last-100 | Std last-100 | Najlepsze avg_100 | Epizod |
|---|---|---|---|---|
| DQN baseline | -199.8 | 2.1 | -197.2 | 411 |
| DQN wariant A | 42.0 | 86.1 | 42.8 | 498 |
| DQN wariant B | -105.4 | 13.1 | -105.4 | 500 |
| DQN wariant C | -109.1 | 18.0 | -108.7 | 490 |
| DQN wariant D | 32.5 | 59.3 | 113.6 | 358 |
| DQN wariant E | -160.3 | 4.7 | -156.4 | 330 |
| REINFORCE baseline | -200.0 | 0.0 | -200.0 | 1 |
| REINFORCE wariant A | -52.2 | 16.3 | -50.4 | 437 |
| REINFORCE wariant B | -189.0 | 4.6 | -177.8 | 1 |
| REINFORCE wariant C | -182.5 | 4.2 | -181.6 | 486 |
| REINFORCE wariant D | -5.4 | 6.2 | -0.3 | 14 |
| REINFORCE wariant E | -155.9 | 3.0 | -155.8 | 493 |

Wartości shaped nieporównywalne między wariantami (różne wzory bonusu, np. D ma skalę rzędu setek) – kolumny służą tylko do oceny przebiegu uczenia w obrębie wariantu, nie do rankingu.

**Heatmapa – DQN, nagroda ewaluacyjna vs gamma** (odczyt z `eval_mean_all_variants_vs_gamma.png`, wartości przybliżone)

| Wariant | γ=0.90 | γ=0.95 | γ=0.99 | γ=0.995 | γ=0.999 |
|---|---|---|---|---|---|
| A | -200 | -200 | -200 | -197 | -200 |
| B | -115 | -116 | -124 | -117 | -120 |
| C | -184 | -145 | -168 | -110 | -108 |
| D | -152 | -106 | -189 | -189 | -196 |
| E | -200 | -200 | -200 | -200 | -200 |

**Heatmapa – DQN, nagroda ewaluacyjna vs epsilon_decay** (odczyt z `eval_mean_all_variants_vs_epsilon_decay.png`, wartości przybliżone)

| Wariant | 0.990 | 0.995 | 0.996 | 0.997 | 0.998 | 0.999 |
|---|---|---|---|---|---|---|
| A | -200 | -200 | -197 | -197 | -200 | -138 |
| B | -143 | -200 | -104 | -137 | -130 | -135 |
| C | -132 | -123 | -115 | -136 | -123 | -137 |
| D | -178 | -200 | -195 | -183 | -181 | -200 |
| E | -200 | -200 | -200 | -200 | -200 | -200 |

Tylko DQN (REINFORCE nie miał tego sweepu). Dokładne liczby wymagają danych źródłowych od Osoby B – obecnie odczytane z wykresu.

## 10. Wnioski
- Odpowiedzi na pytania ze slajdu 2
- Ranking wpływu: baseline vs tuning vs reward shaping vs PPO

## 11. Podsumowanie
- Najważniejsze wyniki
