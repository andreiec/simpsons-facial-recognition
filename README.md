## Problema supusă
    
   În cadrul celei de-a doua teme, am avut, pentru taskul 1, de identificat toate fețele personajelor din Simpsons dintr-o imagine, iar, pentru taskul 2, de clasificat aceste fețe în funcție de personaj (Bart, Homer, Lisa sau Marge).

## Setul de date

Pentru setul de date, am avut la dispoziție un total de 4404 poze (câte 1101 pentru fiecare dintre cele 4 caractere). Pozele conțin fie unul sau mai multe caractere (fie el cunoscut sau necunoscut). Pentru setul de label-uri, am avut câte un document atașat fiecărui folder cu imagini ce conține datele despre fiecare față detectată. Datele se află sub forma (nume_imagine - x_min, y_min - x_max y_max - nume caracter).

## Metoda de rezolvare (Task 1)

Metoda de rezolvare constă în aplicarea unei metode clasice pentru detectarea facială - sliding window, HOG (histogram of oriented gradients) și SVM.

### SVM

Pentru realizarea unei detectări eficiente am folosit un LinearSVC, bazat pe un set de date de 5,454 * 2 exemple pozitive și 6,853 * 2 exemple negative. Fiecare valoare este înmulțită cu 2, întrucât fiecărei imagini îi dau flip astfel încât să realizez o oarecare augmentare.

### Generarea exemplelor negative

Setul de date primit nu conține exemple negative, astfel, am fost nevoiți să le generăm singuri. Pentru a face acest lucru, am luat fiecare imagine din setul de date, am selectat 2 coordonate alese aleator (una x, și una y) și am generat un pătrat, începând cu acele coordonate, de dimensiuni fixe (36 x 36 pixeli). Am verificat apoi dacă acest pătrat intersectează vreun label, dacă nu, înseamnă că este valid și îl salvez pentru folosirea ulterioară. Pentru a asigura o acuratețe și mai bună, am selectat și patrate de dimensiuni mai mari (64 x 64, 96 x 96 etc.) pe care le-am redimensionat ulterior la 36 x 36 de pixeli.

Pentru a selecta exemple negative mai bune, am realizat o filtrare pe galben, astfel încât să am predominant chestii precum mâini, picioare, etc.

Odată generat tot setul de date, trebuie procesat și apoi antrenat Linear SVC-ul. Pentru procesare am transformat imaginile în grayscale și apoi am aplicat HOG pe ele.

### Sliding Window

Sliding window-ul reprezintă un ‘kernel’ ce se plimbă pe imagine, coloană pe coloană, rând pe rând pentru a realiza detectarea.

Implementarea pentru HOG necesită un sliding window pe fiecare celulă a histogramei, astfel, analizând celule întregi și reducând timpul de execuție.

Pentru a spori gradul de detecție, am luat în considerare doar window-uri ce conțin, în mare, galben, din imaginea originală. Un alt aspect important al sliding window-ului, este acela că este realizat pe diverse dimensiuni ale imaginii, astfel încât să detecteze de la cele mai mici la cele mai mari fețe.
    
Atunci când este realizată o detecție, salvez coordonatele și scorul acesteia (scorul fiind dat de către Linear SVM), astfel încât să aplic pe mai multe detecții overlapped funcția de non_maximal_suppression care unește mai multe detecții, după scorul cel mai mare.
    
În final, salvez detecțiile, path-urile și scorurile finale în fișiere .npy

## Metoda de rezolvare (Task 2)

Metoda de rezolvare pentru al doilea task reprezinta o extensie adusă primului task, întrucât suntem nevoiți doar să antrenăm un clasificator pe cele 4 personaje și să facem câte o predicție la fiecare detectare facială ca să vedem ce personaj este.

Clasificatorul folosit este un SVC, antrenat pe 1,146 * 2 poze cu Bart, 1,157 * 2 poze cu Homer, 1,132 * 2 poze cu Lisa și 1,155 poze cu Marge. (Am înmulțit fiecare valoare cu 2 întrucât la fiecare față am făcut flip).
