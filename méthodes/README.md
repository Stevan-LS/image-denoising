# Méthodes de Débruitage Sélectionnées

Pour notre projet, nous avons identifié et sélectionné les meilleures méthodes de débruitage d'images disponibles, en fonction de leur performance et de leur applicabilité à nos données. Ces méthodes sont décrites ci-dessous, dans l'ordre d'importance pour notre cas d'utilisation.

## 1. Blind2Unblind
- **Description** : Méthode de débruitage basée sur une stratégie de restauration progressive, idéale pour les scénarios avec des bruits complexes et non uniformes.
- **Article** : [Blind2Unblind: Self-Supervised Image Denoising](https://arxiv.org/abs/2203.06967)
- **Code** : [GitHub Repository](https://github.com/zejinwang/Blind2Unblind)

## 2. IDR (Iterative Denoising and Refinement)
- **Description** : Approche itérative efficace pour le débruitage, exploitant les propriétés structurelles des images.
- **Article** : [IDR: Iterative Denoising and Refinement](https://arxiv.org/abs/2111.14358)
- **Code** : [GitHub Repository](https://github.com/zhangyi-3/IDR)

## 3. Noise2Self
- **Description** : Technique auto-supervisée pour le débruitage, qui utilise uniquement l'image bruitée sans nécessiter de ground truth.
- **Article** : [Noise2Self: Blind Denoising by Self-Supervision](https://arxiv.org/abs/1901.11365)
- **Code** : [GitHub Repository](https://github.com/czbiohub-sf/noise2self)

## 4. Noise2Void
- **Description** : Méthode de débruitage auto-supervisée qui repose sur le masquage des pixels et leur reconstruction par le modèle.
- **Article** : [Noise2Void: Learning Denoising from Single Noisy Images](https://arxiv.org/abs/1811.10980)
- **Code** :
  - [PyTorch Implementation](https://github.com/hanyoseob/pytorch-noise2void)
  - [Original Implementation](https://github.com/juglab/n2v)
