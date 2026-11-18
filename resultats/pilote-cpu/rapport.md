# Quantifier les poids d’un modèle de langage

**Essai pilote** — Le modèle devait écrire une lettre par lui-même, avec un budget de quatre tokens. Les réponses non conformes ont été comptées comme des erreurs. Le rapport principal mesure aussi le choix entre quatre lettres avec un décodage contraint.

**Question étudiée** — Combien de mémoire gagne-t-on en codant les poids sur 4 bits, et quelles réponses changent ?

Modèle **Qwen/Qwen2.5-1.5B-Instruct**, révision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Exécution **Transformers / CPU**.

## Mémoire et qualité

| Mesure | Référence FP16 | Poids INT4 |
| --- | ---: | ---: |
| Tenseurs stockés, échelles et zéros inclus | 2.875 Gio | 1.069 Gio |
| Checkpoint Safetensors sur disque | 2.875 Gio | 1.069 Gio |
| Bonnes réponses au QCM | 18/40 | 13/40 |
| Réponses avec un format invalide | 0 | 8 |

Le stockage des tenseurs diminue de **62.8%**. Le score du QCM varie de **-12.5 points**. La même lettre valide est donnée à 20/40 questions. L’INT4 gagne 5 question(s) et en perd 10.

Sur CPU, les deux variantes calculent en FP32 avec des poids arrondis en FP16. Les poids INT4 sont décompressés pour calculer. La table donne la taille du stockage. La consommation de VRAM sera mesurée par le parcours vLLM/ROCm.

## Erreur numérique

Les 196 projections linéaires sont quantifiées par groupes de 128 coefficients dans chaque ligne. Pour un groupe, on choisit une échelle s à partir de l’intervalle des poids (étendu à zéro) et un point zéro entier z. On stocke q = clip(arrondi(w/s) + z, 0, 15) et on reconstruit ŵ = s(q − z). Chaque groupe stocke aussi son échelle en FP16 et son point zéro sur 4 bits.

En relisant les poids compactés, l’erreur relative de Frobenius globale ‖Ŵ − W‖ / ‖W‖ vaut **10.52%**. L’écart quadratique moyen par coefficient (racine comprise) vaut 0.0029764, et l’erreur absolue maximale vaut 0.11719. Le rapport de normes et le score du QCM mesurent deux effets différents de la même approximation.

## Réponses qui diffèrent

- Combien vaut 3/8 + 1/6 ? Attendu **D**, FP16 **B**, INT4 **D**.
- Un prix de 80 € augmente de 25 %, puis diminue de 20 %. Quel est le prix final ? Attendu **B**, FP16 **B**, INT4 **C**.
- Quel est le plus petit multiple commun de 12 et 18 ? Attendu **A**, FP16 **D**, INT4 **A**.
- Que vaut 2 puissance 10 ? Attendu **B**, FP16 **A**, INT4 **C**.
- Combien de secondes y a-t-il dans 2 h 15 min ? Attendu **D**, FP16 **B**, INT4 **invalide**.
- Quelle est la moyenne de 7, 9, 12 et 16 ? Attendu **B**, FP16 **B**, INT4 **A**.
- Quel est le déterminant de la matrice de lignes (2, 3) et (5, 7) ? Attendu **C**, FP16 **B**, INT4 **invalide**.
- Dans le plan, quelle est la distance entre (1, 2) et (4, 6) ? Attendu **A**, FP16 **A**, INT4 **B**.
- On tire une boule uniformément parmi 4 rouges et 6 bleues. Quelle est la probabilité de tirer une rouge ? Attendu **C**, FP16 **A**, INT4 **C**.
- Camille arrive avant Zoé. Nour arrive après Camille et avant Zoé. Qui arrive en deuxième parmi les trois ? Attendu **C**, FP16 **B**, INT4 **C**.
- Une carte n’est acceptée que si elle est signée. La carte est acceptée. Que sait-on ? Attendu **B**, FP16 **B**, INT4 **invalide**.
- Combien de paires distinctes de personnes peut-on former parmi Anne, Bilal et Chloé, en ignorant l’ordre ? Attendu **A**, FP16 **A**, INT4 **D**.
- Le train quitte Tours à 9 h 12 et arrive à Poitiers à 10 h 01. Quelle est la durée du trajet ? Attendu **A**, FP16 **A**, INT4 **B**.
- Le musée ferme le mardi. Le lundi, l’entrée est gratuite. Quel jour est gratuit d’après le texte ? Attendu **D**, FP16 **D**, INT4 **invalide**.
- Lina a rangé le carnet dans le tiroir et le stylo dans le sac. Où se trouve le carnet ? Attendu **A**, FP16 **A**, INT4 **invalide**.
- Le colis de Max est arrivé vendredi, deux jours après son envoi. Quel jour a-t-il été envoyé ? Attendu **B**, FP16 **B**, INT4 **C**.
- Le refuge est accessible par le pont, fermé après 20 h, ou par le sentier, ouvert toute la nuit. Quel accès est indiqué à 22 h ? Attendu **D**, FP16 **D**, INT4 **invalide**.
- Jules prête trois BD à Inès et une à Mei. Le lendemain, Inès en rend deux. Combien de ses BD sont encore empruntées ? Attendu **C**, FP16 **B**, INT4 **C**.
- Le rapport doit être remis avant jeudi. Nora l’a envoyé mercredi. Le rapport est-il arrivé dans les délais si l’envoi est reçu immédiatement ? Attendu **A**, FP16 **B**, INT4 **invalide**.
- L’équipe a prévu de courir si la piste est sèche et de nager sinon. La piste est mouillée. Quelle activité est prévue ? Attendu **B**, FP16 **D**, INT4 **invalide**.

## Deux réponses à lire

### Explique en deux phrases pourquoi le produit de deux nombres impairs est impair.

Critère de lecture — La justification fait apparaître que le produit a la forme 2k + 1.

**FP16**

> Le produit de deux nombres impairs est impair car l'impair est défini comme un nombre entier qui ne peut être divisé par deux sans avoir un reste. Lorsque vous multipliez deux nombres impairs, vous obtenez un produit qui est également un nombre entier. Par conséquent, ce produit ne peut pas être divisé par deux sans avoir un reste, ce qui signifie qu'il est impair.

**INT4**

> Le produit de deux nombres impairs est impair car l'impair est défini comme étant divisible par 2 et ne pouvant donc pas être divisé par 2 sans rester entier. Lorsque vous multipliez deux nombres impairs, vous devez diviser par 2 et obtenir un reste, ce qui signifie que le résultat est également impair.

### Paul affirme que toute matrice carrée non nulle est inversible. Donne un contre-exemple 2 × 2 et explique-le brièvement.

Critère de lecture — La réponse donne une matrice non nulle avec un déterminant nul ou des lignes dépendantes.

**FP16**

> Paul est incorrect. Une matrice carrée non nulle n'est pas nécessairement inversible. Un contre-exemple 2 × 2 est la matrice identité 2 × 2, qui est de la forme:
> 
> \[
> \begin{pmatrix}
> 1 & 0 \\
> 0 & 1
> \end{pmatrix}
> \]
> 
> Cette matrice est inversible car elle est égale à sa propre inverse. Pour vérifier, on peut calculer la multiplication de la matrice par sa propre inverse :
> 
> \[
> \begin{pmatrix}
> 1 & 0 \\
> 0 & 1
> \

**INT4**

> Paul est incorrect. Il existe des matrices carrées non nulles qui ne sont pas inversibles. Un contre-exemple 2 × 2 est la matrice suivante :
> 
> \[
> \begin{pmatrix}
> 1 & 0 \\
> 0 & 0
> \end{pmatrix}
> \]
> 
> Cette matrice est carrée, non nulle et n'est pas inversible. Pour voir pourquoi, rappelle-toi que la matrice est inversible si et seulement si son déterminant est non nul. Le déterminant de cette matrice est \(1 \times 0 - 0 \times

Le QCM interne de 40 questions teste surtout le calcul, la logique et la compréhension de consignes courtes en français. Les réponses libres sont à examiner avec leur critère. Le résultat décrit ce modèle, ce réglage et ces questions.
