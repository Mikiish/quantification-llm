# Quantification des poids d’un modèle de langage

**Question pour la présentation** — Combien de mémoire récupère-t-on en remplaçant les poids d’un modèle par des codes sur 4 bits, et que devient la qualité de ses réponses ?

Le [rapport GPU](resultats/rocm/rapport.md) présente les mesures effectuées sur la RX 7900 XTX sous WSL2. La hausse de VRAM quand le serveur est prêt passe de 3,73 à 1,87 Gio avec le même cache KV. Les deux variantes totalisent 18 bonnes réponses sur 40 et changent de lettre sur 13 questions.

On compare deux variantes du même [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Le programme garde les mêmes embeddings, biais et normalisations en FP16. Il quantifie les 196 projections de l’attention et du MLP, sous la forme de groupes de 128 coefficients par ligne.

## Ce qu’on mesure

- **Taille des poids.** Chaque groupe de 128 coefficients occupe 256 octets en FP16. Après quantification, il occupe 64 octets de codes, 2 octets d’échelle et un demi-octet de point zéro. Les tenseurs restés en FP16 expliquent pourquoi le gain total est inférieur au gain par groupe.
- **Erreur numérique.** Pour les bornes `a ≤ 0 ≤ b` d’un groupe, on prend `s ≈ (b − a) / 15` et `z = clip(arrondi(−a/s), 0, 15)`. On stocke `q = clip(arrondi(w/s) + z, 0, 15)` et on reconstruit `ŵ = s(q − z)`. Le programme relit réellement les mots compactés pour mesurer la norme de l’erreur et son maximum. Le fichier utilise le format **GEMM AWQ** pour l’inférence vLLM. La quantification elle-même applique une règle uniforme min-max sans calibration sur des activations.
- **Comportement du modèle.** On soumet les mêmes 40 QCM à chaque modèle. Chacun choisit la lettre de plus forte probabilité parmi A, B, C et D. Les deux réponses libres permettent aussi de comparer la génération. Le rapport affiche le nombre de réponses correctes, les questions gagnées et perdues et les réponses brutes.
- **VRAM.** Les deux instances vLLM démarrent successivement sur la même carte, en mode eager et avec le même cache KV de 128 Mio. Le programme relève la VRAM de la carte avant le lancement, une fois le serveur prêt et après les requêtes. La mesure au chargement du modèle issue du journal vLLM complète ces relevés.

La référence est chargée en FP16 depuis le checkpoint BF16 officiel. Les deux formats occupent le même nombre d’octets par poids.

## Lancer la comparaison sur cette machine

Depuis Ubuntu WSL2, les checkpoints de l’expérience sont sous `.work/` et le runtime ROCm est dans ton dossier utilisateur.

```bash
cd /mnt/c/Users/Mikihisa/Desktop/CoursMaths/Workspace/quantification-llm
source ~/.cache/quantification-llm/rocm/bin/activate
python experience.py preparer
python experience.py comparer --backend rocm
```

`preparer` télécharge la révision épinglée du modèle et crée le fichier INT4. Les exécutions suivantes réutilisent les checkpoints préparés. Les journaux vLLM, le rapport Markdown et les données JSON vont dans `resultats/rocm/`. Compter environ 4 Gio pour les deux checkpoints et plusieurs Gio pour l’environnement logiciel.

## Faire la démonstration avec un client distant

Sur le PC qui détient la carte, depuis l’environnement Linux ci-dessus, lancer une variante sur l’interface locale. Arrêter le serveur avec `Ctrl+C` avant de lancer le benchmark ou de changer de variante.

```bash
python experience.py servir int4 --port 8000
```

Un client PowerShell sous Windows a envoyé une requête à ce serveur WSL2 sur `127.0.0.1:8000`. Depuis une autre machine, utiliser un serveur SSH déjà accessible sur le PC qui peut joindre ce port local. Le tunnel transmet les requêtes au serveur vLLM. Le calcul reste sur la Radeon.

```bash
ssh -N -L 8000:127.0.0.1:8000 utilisateur@adresse-du-pc
```

Dans un deuxième terminal client, sous Bash :

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"int4","messages":[{"role":"user","content":"Explique pourquoi 2 puissance 10 vaut 1024."}],"temperature":0,"max_tokens":100}'
```

Pour essayer la référence, lancer `servir fp16` et utiliser `"model":"fp16"` dans la requête. L’expérience se lance aussi depuis le terminal Codex Desktop via `wsl.exe`. Un environnement cloud est facultatif. S’il sert de client, il doit disposer d’un chemin réseau privé vers le serveur.

## Recréer l’environnement GPU

Le parcours WSL2 utilise Ubuntu 26.04, ROCm 10.1, Python 3.14 et la cible **gfx1100**. La pile ROCm en wheels embarque le pont ROCDXG. Les prérequis de version Windows et de pilote dépendent du matériel. [Matrice ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) · [Notes de version ROCm 10.1](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#installation-updates)

Avec `uv` installé, dans le dossier du projet :

```bash
uv venv --python 3.14 ~/.cache/quantification-llm/rocm
source ~/.cache/quantification-llm/rocm/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --index https://rocm.frameworks-prereleases.amd.com/whl-multi-arch-staging/vllm/ \
  --index https://stable.repo.amd.com/rocm/whl-next/ \
  -r requirements-rocm.txt
uv pip install --no-deps tensorizer==2.12.1
uv pip uninstall flash-attn
python -c 'import torch; print(torch.cuda.get_device_properties(0)); x = torch.ones(2, 2, device="cuda"); print(x @ x)'
```

La version de PyTorch est épinglée à ROCm 10.1 pour éviter la sélection automatique d’une wheel ROCm 10.2 dans l’index partagé. La mise à niveau de Tensorizer corrige l’erreur d’import en Python 3.14, selon le contournement documenté par AMD. Dans cet environnement dédié, la désinstallation de `flash-attn` évite l’import d’une extension compilée pour une ABI PyTorch incompatible. La comparaison utilise l’attention Triton. [ROCm CLI et vLLM](https://github.com/ROCm/rocm-cli/blob/main/docs/vllm.md#rocm-10x-wheel-discovery)

Sur ton Linux natif, une autre piste est l’[image Radeon vLLM 0.16 d’AMD](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/linux/llm/build-docker-image.html). Elle utilise les périphériques natifs `/dev/kfd` et `/dev/dri`. Monter le projet dans le conteneur et lancer la même commande `python experience.py comparer --backend rocm`.

## Relancer sur CPU

Depuis PowerShell, dans le dossier du projet, avec Python 3.13 :

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements-cpu.txt
.venv/Scripts/python experience.py preparer
.venv/Scripts/python experience.py comparer --backend cpu
```

Sur CPU, le programme reconstruit les poids à partir des codes INT4 et compare les sorties FP32 calculées avec les deux jeux de poids. Les chiffres mémoire en CPU concernent le stockage compacté. Les relevés VRAM figurent dans le rapport GPU.

Le [rapport CPU](resultats/cpu/rapport.md) et l’[essai pilote](resultats/pilote-cpu/rapport.md) accompagnent le code. Dans cet essai pilote, les deux variantes devaient générer librement une lettre. Les réponses hors format ont contribué à l’écart de score.
