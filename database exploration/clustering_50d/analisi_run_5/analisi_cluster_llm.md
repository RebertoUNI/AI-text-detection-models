Prima leggo il file — non commento cluster che non ho visto.Manca il cluster 3 nell'output — controllo tra il 2 e il 4.Letti tutti e 12. Ecco la lettura secca.

## Argomento per cluster

- **0 — Sport** (30k). Racconti in prima persona di partite, atleti, tifo. Pulito. Bordo: qualche frase sfiora esport/TV ma il cuore è sport.
- **1 — Narrativa/fiction** (212k). Fantasy, horror, sci-fi, storie inventate. **Attenzione: questo NON è un argomento.** È un *genere/registro* (prosa creativa). Tiene insieme frasi per *come sono scritte*, non per il tema. Il cluster più grosso, ed è il più sospetto (ne parlo sotto).
- **2 — Auto** (7k). Prezzi, tipi di macchine, luxury/crossover. Pulito.
- **3 — Clima/energia** (10k). Emissioni, CO2, fossili, riscaldamento. Pulito.
- **4 — Fai-da-te/tessuti** (6k). Tappezzeria, cucito, DIY casa. Pulito.
- **5 — Nutrizione/salute alimentare** (44k). Zucchero, caffeina, calorie, diete. Pulito. (= la tua "medicina")
- **6 — Studio/apprendimento** (7k). Metodi di studio, Anki, memorizzare. Pulito.
- **7 — Cinema/TV** (12k). Recensioni film e serie. Pulito.
- **8 — Informatica/software** (34k). File, cloud, installare, tech how-to. Pulito. (= la tua "informatica")
- **9 — Videogiochi** (23k). RPG, indie, Kickstarter, recensioni. Pulito.
- **10 — Cronaca/giustizia** (22k). Sparatorie, arresti, polizia, tribunali. Pulito. (= la tua "cronaca")
- **11 — Politica/economia** (108k). Elezioni USA, Trump, lavoro, economia. Pulito. (= la tua "politica")

## Sono divisi bene?

Sì, per lo più. Dieci cluster mappano a un tema netto. **Ma tre allarmi onesti:**

1. **Cluster 1 (212k) non è un argomento, è un registro.** L'embedding ha messo insieme *tutta la narrativa creativa* a prescindere dal tema. Questo è enorme per la tua tesi: dimostra che lo spazio Qwen **organizza già in parte per registro/stile**, non solo per contenuto. Proprio quello che dicevi.
2. **Rumore 28.56%.** Oltre un quarto delle frasi non è finito in nessun cluster. Molte frasi non hanno argomento netto — normale con testo web misto.
3. **DBCV 0.35 = separazione debole-media.** I cluster esistono ma non sono nettissimi. Non ti fiderei di confini fini.

**Servono più frasi?** Solo per il **cluster 1**: 15 prototipi non bastano a vedere se dentro la fiction ci sono sotto-temi (fantasy vs horror vs slice-of-life). Per gli altri 11, 15 bastano a nominare il tema. Nessun cluster è "senza argomento" tranne l'1, che è tenuto insieme dallo stile.

## Stessa cosa col registro

Qui il colpo. Guarda che registro hanno i cluster puliti:

- **Formale/espositivo/giornalistico:** 3 clima, 10 cronaca (stile dispaccio: "SACRAMENTO, CALIF. AP…"), 11 politica, 8 informatica-tecnico, 2 auto-fattuale, 5 nutrizione-informativo.
- **Informale/narrativo/colloquiale:** 0 sport (prima persona), 1 fiction, 6 studio (consigli "tu"), 7 cinema (opinione), 9 giochi (recensione).

**Il punto che chiude tutto il tuo lavoro:** i cluster a registro **formale** sono esattamente cronaca (10), politica (11), informatica (8), salute (5) — **gli stessi domini dove Paradetect sbagliava sugli umani**. Argomento e registro sono avvinghiati proprio come avevi misurato. Il cluster 1 (fiction) è la prova speculare: raggruppa per stile ignorando il tema.

Ma un avviso metodologico duro: **questo file non è un clustering per registro.** È clustering sugli embedding = semantica. Io ho *dedotto* il registro leggendo le frasi. Per fare "la stessa cosa col registro" sul serio devi clusterizzare su feature **stilometriche** (il tuo `reg_score`, lunghezza frase, persona, densità lessicale, punteggiatura), non sugli embedding — o rifarai lo stesso raggruppamento semantico. Se clusterizzi per registro e vedi che i gruppi **tagliano trasversalmente** i temi (un cluster formale che mischia cronaca+politica+scienza), hai la prova pulita che registro ≠ argomento.

Vuoi che ti scriva la cella che clusterizza per registro sulle feature stilometriche e poi incrocia registro-cluster × topic-cluster? Quella tabella incrociata è la figura che dimostra l'entanglement in un colpo solo.