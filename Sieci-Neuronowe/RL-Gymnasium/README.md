# Optymalizacja agenta RL w Gymnasium (MountainCar-v0)

*Projekt grupowy: baseline, tuning hiperparametrów i reward shaping dla DQN/REINFORCE/PPO w środowisku MountainCar-v0.*

## 📖 Opis

Cel projektu: poprawić jakość uczenia agenta RL względem baseline w trudnym, rzadko-nagradzanym środowisku MountainCar-v0. Praca podzielona na role zespołowe (`podzial_rol.pdf`):

- **Baseline** — DQN i REINFORCE bez modyfikacji + teoria
- **Tuning hiperparametrów** — sweep `gamma` i `epsilon_decay` + automatyzacja eksperymentów
- **Reward shaping** — 5 wariantów (`reward_wrappers.py`): A) bonus za przesunięcie od środka, B) bonus za prędkość, C) hybryda A+B, D) potential-based shaping (PBRS), E) bonus za nowy rekord wysokości
- **PPO** — zaawansowany algorytm (`agents/ppo_agent.py`)
- **Integracja wyników i raport końcowy** — metryki, wykresy, wnioski, prezentacja

## 📈 Kluczowe wnioski

- Baseline DQN i REINFORCE praktycznie nie uczą się w MountainCar bez reward shapingu w 500 epizodach (REINFORCE zostaje płasko na -200 od pierwszego epizodu)
- Wariant E (bonus za rekord wysokości) jest nieskuteczny dla DQN — płaski wynik -200 w obu sweepach hiperparametrów
- Wariant D (PBRS) zależy nieliniowo od `gamma`, bo gamma wchodzi bezpośrednio do wzoru bonusu shapingowego — myli interpretację sweepa
- Nagroda treningowa w logach jest *shaped* (nieporównywalna między wariantami — inny wzór bonusu); do porównań między wariantami/metodami służy nagroda **ewaluacyjna** (`evaluate()` na nieowiniętym środowisku)

## 📂 Struktura

| Folder | Zawartość |
|---|---|
| `agents/` | Implementacje DQN, REINFORCE, PPO |
| `reward_wrappers.py` | 5 wariantów reward shapingu (A–E) |
| `run_baseline.py`, `run_shaping_experiments.py` | Skrypty uruchamiające eksperymenty |
| `utils/` | Logger, trening, wykresy |
| `results/logs/` | Logi CSV (episode/reward/avg_100) per wariant i metoda — surowe artefakty treningu |
| `results/models/` | Wytrenowane modele (DQN, REINFORCE) — surowe artefakty treningu |
| `plots/` | Surowy, nadpisywany co uruchomienie output `utils/plotting.py` (aktualnie 2 pliki reward shapingu) |
| `wyniki/plots/` | Wyselekcjonowana kopia wykresów użyta w prezentacji końcowej: baseline, tuning (`gamma/`, `epsilon_decay/`) i reward shaping |
| `wplyw_hiperparametrow/` | Wykresy porównawcze wpływu `gamma` i `epsilon_decay` per wariant (inne cięcie tych samych danych niż `wyniki/plots/gamma\|epsilon_decay/`) |
| `nagrania_reward_shaping_baseline/` | Nagrania polityk (mp4) + zestawienia najlepszych wariantów |
| `animacje/` | Animowane porównania (gif) — wyścig baseline, reward shaping, trajektorie samochodu |
| `plan_prezentacji.md` | Plan slajdów prezentacji końcowej |
| `podzial_rol.pdf` | Podział ról zespołu |
| `prezentacja_wstepna_rl_gymnasium.pptx` | Prezentacja wstępna (środowisko, kierunki poprawy, pytania badawcze) |
| `prezentacja_koncowa_rl_gymnasium.pptx` | Prezentacja końcowa (18 slajdów, wideo i gify wbudowane) |
| `build_final_presentation.js` | Skrypt budujący prezentację końcową (pptxgenjs) |

## ⚠️ Uwagi

- `results/` (ang.) i `wyniki/` (pol.) to nie duplikaty — `results/` to surowe artefakty treningu (logi CSV, checkpointy modeli), `wyniki/` to wyselekcjonowane wykresy do prezentacji końcowej. Dwujęzyczna nazwa jest nieidealna, ale zmiana wymagałaby edycji ~8 miejsc w `build_final_presentation.js` i `plan_prezentacji.md` bez realnej korzyści.
- `plots/` (surowy, nadpisywany output skryptów) i `wyniki/plots/` (kopia użyta w prezentacji) mają 2 wspólne pliki — to zamierzone, nie przypadkowy duplikat.

## 🛠️ Technologie

| Technologia | Szczegóły |
|---|---|
| Python | Gymnasium (MountainCar-v0), PyTorch |
| Infrastruktura | Docker + GPU |

## 👤 Autor

Michał Frąckowiak (rola: metryki ewaluacyjne, integracja wyników zespołu, wykresy/tabele/heatmapy, wnioski, raport i prezentacja końcowa) + zespół
