# SK.ALHE.12

Dla sieci o nazwie india35 ze strony http://sndlib.zib.de/home.action zastosować algorytm mrówkowy(Ant Colony) do znalezienia najlepszych (wg. ustalonej metryki) ścieżek w danej sieci. Porównanie z innym algorytmem będzie dodatkowym atutem.

## Wymagania

Wszystkie potrzebne paczki są widoczne w ```environment.yml```. 

Stworzenie środowiska poprzez ```conda```:

```bash
conda env create -f environment.yml
```

## Użycie

Uruchomienie głównego skryptu:

```bash
python -m antcolony PATH_TO_GRAPH START_NODE END_NODE
```

Opis opcjonalnych parametrów można uzyskać poprzez:

```bash
python -m antcolony --help
```

W katalogu głównym znajduje się plik z definicją grafu.

Do wyświetlenia grafu można posłużyć się pomocniczym skryptem:

```bash
python -m antcolony.scripts.plot PATH_TO_GRAPH
```