# Optymalizator układnia paczek na palecie

Skrypt układa paczki na palecie, optymalizując miejsce (Fill Rate) i stabilność (Środek Ciężkości). Generuje wizualizację wyniku.

## Co to robi?
1. **Upchanie:** Algorytm szuka najlepszego ułożenia elementów (możliwy obrót).
2. **Fizyka:** Oblicza środek ciężkości ładunku (CoG), żeby paleta się nie przewróciła.
3. **Raport:** Zapisuje obrazek `result_final.png` z wizualizacją ułożenia.

## Wymagania
Musisz mieć Pythona i te biblioteki:

```bash
pip install matplotlib numpy
