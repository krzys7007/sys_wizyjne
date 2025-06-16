
# **Sprawozdanie z projektu - Systemy wizyjne**
**Nowakowski Krzysztof 151246**

## 1. **Cel projektu**

Celem projektu jest przetworzenie zadanych sekwencji wideo w celu wyznaczenie liczby samochodów osobowych, ciężarowych / autobusów, tramwajów, które poruszają się po drodze dwujezdniowej oraz pieszych i rowerzystów poruszających się po drodze pieszo-rowerowej.

----------

## 2. **Opis działania kodu**

### 2.1. **Koncept działania**
W projekcie wykorzystano podejście bardzo prostej klasyfikacji obiektów na podstawie rozmiaru pikseli danego obszaru. Inne podejścia opisane w sekcji **3.** nie dawały zadawalających rezultatów. Najbardziej efektywnym rozwiązaniem okazało się najprostsze - polegające na podzieleniu nagrania na 4 obszary (chodnik, bliższa ulicę, torowisko i dalszą ulicę) i zliczaniu obiektów na tych obszarach oraz wykryciu ich kierunku.


### 2.2. **Wykorzystane przekształcenia oraz funkcje**

#### 2.2.1. **Przetwarzanie wideo**

-   Wczytywanie klatek z kamery (`cv2.VideoCapture`),
    
-   Tworzenie maski tła przez `cv2.createBackgroundSubtractorMOG2`,
    
-   Operacje morfologiczne i filtracja konturów (`morphologyEx`, `dilate`).
    

#### 2.2.2. **Usuwanie zasłaniających obiektów**

-   Funkcja `remove_multiple_crooked_lines` służy do usuwania przeszkód (dwóch słupów i dwóch drzew)  z obrazu poprzez inpainting (metoda TELEA).
    

#### 2.2.3. **Segmentacja i klasyfikacja obiektów**

-   Bounding Boxy są grupowane i filtrowane za pomocą funkcji `merge_boxes`, `remove_inner_boxes`,

-   Kontury są wykrywane (`cv2.findContours`) i potem klasyfikowane w funkcji (`classify`) na postawie położenia (obszaru) oraz rozmiaru.

#### 2.2.4. **Śledzenie obiektów i kierunek ruchu**
    
-   Przypisuje się ID i zapamiętuje trajektorię dzieki funkcjom (`track_paths`, `object_types`),
    
-   Funkcja `get_direction()` analizuje trajektorie w odniesieniu do linii referencyjnej,
    
-   Po jednokrotnym przekroczeniu linii, obiekt jest zliczany i oznaczany jako przetworzony.

----------

## 3. **Trzy wcześniejsze podejścia i finalne:**
### **3.1. Pierwsze podejście**
Pierwszym pomysłem było wykorzystanie prostej klasyfikacji na podstawie rozmiaru oraz stosunku wysokości do szerokości bounding boxa. W ostatecznej wersji zrezygnowano z tego podejścia i pozostawiono tylko sam rozmiar, ponieważ słupy oraz pojawiające sie artefakty po odejmowaniu tła uniemożliwiały utworzenie jednolitych boundingboxów, co skutkowało np. pojawieniem się dwóch boxów na jednym obiekcie. Dodatkowo sam stosunek wysokości do długości był zbyt podobny przy rozróżnianiu samochodów i pojazdów ciężarowych. Pozostawiono rozróżnienie tylko dla rowerów i ludzi.

### **3.2. Drugie podejście**
W drugiej iteracji zastosowano algorytm uczenia maszynowego PCA. Zebrano próbne bounding boxy z całego nagrania, wpisano ich cechy do pliku CSV i przyporzadkowano im etykiety (n- nothing, a - car, c - ciężarówka, p - pieszy, t - tramwaj, b - rower). Nastepnie wytrenowano algorytm do rozpoznawania. Niestety efekt był niezadowalający i w porównaniu z pierwszą metoda wypadał gorzej.

### **3.3. Trzecie podejście**
W trzecim podejściu zdecydowano się wykorzystać znaczniki TrackerCSRT, jednak po kilku próbach zrezygnowano z powodu zbyt wolnego przetwarzania znaczników oraz zbyt małej wiedzy autora o bibliotece.

### **3.4. Ostateczne rozwiązanie**
Na podstawie trzech prób wybrano wersję pierwszą, okrojoną o wyliczanie stosunku wysokości do szerokości, ale wprowadzono dodatkowo podział na cztery obszary. Zrezygnowano równiez ze zliczania obiektów, jesli ich bounding box wyjechał poza obraz, w zamian za to dany bounding box zliczany jest tylko jeśli zniknie (np. za przeszkodą). Obiektom przypisuje się klasyfikacje dopiero przy przejechaniu przez wyznaczoną linie na obrazie, gdyz są wtedy najlepiej widoczne.

----------

## 4. **Możliwości rozwoju kodu**
Pierwszym usprawnieniem w szybkości przetwarzania kodu byłoby wykorzystanie obszarów (ROI) do narzucenia kierunku ruchu, gdyż aktualnie ruch wyliczany jest z pozycji obiektów na poprzednich klatkach. Zajmuje to pamięć i czas, jednak dla danego podejścia dawało to lepsze efekty.
Drugą opcją rozwoju jest wytrenowanie lepszego algorytmu do rozpoznawania obiektów na obrazie. Możnaby wykorzystać charakterystyczne elementy jak większa ilość kół pojazdów ciężarowych. Implementacja tego niestety przerosła autora.


