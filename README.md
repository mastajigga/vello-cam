# vello-cam

Une **caméra dans le navigateur** dont les filtres tournent **sur le GPU** et dont l'interface est
**vectorielle**, dessinée sur le même appareil graphique.

Tout est écrit en **Rust**, compilé en **WebAssembly**, et rendu via **WebGPU**. Aucun framework JS,
aucun `node_modules` : un seul `.wasm` de ~570 Ko.

> **Rust → WebAssembly → WebGPU**, avec les filtres en WGSL et l'UI en
> [Vello](https://github.com/linebender/vello) (`vello_gpu`).

---

## Ce que ça fait

- **Caméra → texture GPU en un appel par image** (`queue.copy_external_image_to_texture`) : l'image ne
  repasse **jamais** par le CPU dans le cas nominal.
- **Filtres WGSL combinables en deux passes**, colorimétrique puis spatiale si nécessaire,
  avec ping-pong de textures :

  | Touche | Filtre |
  |---|---|
  | `1` | Noir et blanc |
  | `2` | Sépia |
  | `3` | Flou (5 taps) |
  | `4` | Fisheye |
  | `5` | Vignette |
  | `6` | Prisme (aberration chromatique) |

  S'y ajoutent, toujours branchés : grain argentique, température, saturation, contraste, étalonnage.
  Tous pilotés par **un unique tampon d'uniformes** (16 `f32` = 64 octets).
- **Interface vectorielle Vello composée en `SrcOver`** : voiles à dégradé continu, six filtres
  combinables, palette anthracite / ivoire / citron vert, grand obturateur et commandes distinctes
  vidéo / grille. L'image reste au centre, sans cadre ni légende technique superposée.
- **Mobile d'abord** : une rangée de filtres en portrait, deux sur écran très étroit ; en paysage
  court, deux rangées à gauche et les commandes de capture à droite. Les cibles mesurent au moins
  **44 × 44 px CSS**, indépendamment du DPR. L'obturateur mesure 76 × 76 px.
- **Toucher, souris et stylet** : `pointerdown` donne un retour immédiat, la capture du pointeur
  assure le suivi hors du canvas et seul le relâchement sur la cible d'origine valide l'action.
  Sortir puis revenir réarme la cible ; relâcher ailleurs, `pointercancel`, perte de capture,
  perte de focus ou redimensionnement annulent le geste. Les doigts secondaires sont ignorés.
- **Animations** : compression / relâchement amortis, ressorts de sélection avec sous-pas stables,
  apparition des pictogrammes, transition progressive des filtres eux-mêmes, fondu de la grille,
  transition du point vidéo au carré stop, pulsation REC et flash photo.
- **Photo** (`Espace`, ou obturateur) : relecture GPU (`copy_texture_to_buffer` + `map_async`) → PNG
  sans UI. Command buffer de capture séparé et garde `capture_busy` conservés.
- **Vidéo** (`R`) : `canvas.captureStream(60)` + `MediaRecorder` ; le dessin Vello est inclus,
  les libellés HTML ne le sont pas. La photo reste disponible pendant l'enregistrement.
- **Grille des tiers** (`G`) et **réinitialisation des six filtres** (`0`). `1` à `6` combinent les
  filtres. `Tab` parcourt les commandes ; `Espace` / `Entrée` activent le bouton focalisé.

### Choix d'interface

Le dessin et le mouvement restent dans `src/ui.rs`. Les libellés, le statut et les boutons
sémantiques sont en HTML pour une typographie nette, les lecteurs d'écran (`aria-pressed`,
statut annoncé) et le clavier. Leurs positions viennent du **même calcul Rust** que les cibles
Vello ; ils ne capturent pas les événements du pointeur et n'ajoutent pas de logique d'UI JS.

Le conteneur commun réserve `env(safe-area-inset-*)` : les coordonnées Vello commencent donc
**à l'intérieur de la zone sûre**, sans estimation de l'encoche côté Rust. Cela laisse une bande
sombre dans les zones système plutôt que de placer la caméra ou les commandes sous celles-ci.
`touch-action: none` sur le canvas supprime les gestes navigateur et le délai de double-tap ;
le viewport conserve le zoom accessible ailleurs, sans `user-scalable=no`.

Le module conserve la `Scene` avec `reset()`, deux tracés unitaires réutilisés par transformations,
des tableaux fixes et deux dégradés calculés au redimensionnement. Aucun nouveau `Vec`, tracé
ou texte formaté n'est construit par image par la couche UI. Aucune dépendance ajoutée.
Les contrôles vidéo et photo restent distincts pour que l'action de l'obturateur soit sans ambiguïté.
Les réglages avancés sont volontairement absents pour garder le viseur lisible.

### Vérifier l'interface

Les tests purs de géométrie et de gestes s'exécutent sans navigateur ni GPU :

```bash
rustc --edition=2021 --test src/ui/layout.rs -o /tmp/vello-layout-tests
/tmp/vello-layout-tests
rustc --edition=2021 --test src/ui/gesture.rs -o /tmp/vello-gesture-tests
/tmp/vello-gesture-tests
```

Ils couvrent huit dimensions (portrait, paysage, bureau), l'absence de chevauchement, les cibles
minimales, l'appui / relâchement unique, la sortie / rentrée, l'annulation et le second doigt.
La compilation release WASM est vérifiée. Le navigateur et le serveur HTTP ne pouvant pas démarrer
dans le sandbox de cette refonte, **la validation visuelle sur téléphone, les 60 fps et les tests
médias en exécution restent à refaire**. Les preuves GPU historiques ci-dessous ne constituent
pas une nouvelle exécution des auto-tests après la refonte.

## Le pipeline d'une image

```
getUserMedia ─► <video> ─┐
                         │  copy_external_image_to_texture   (1 appel, 0 copie CPU)
                         ▼
                    texture caméra
                         │
                         ▼  passe 1 : WGSL  fs_grade      ── température, saturation, contraste,
                    ping-pong ── couleur (N&B, sépia)       N&B, sépia, vignette, grain
                         │
                         ▼  passe 2 : WGSL  fs_spatial    ── fisheye, aberration chromatique, flou
                    texture affichée
                         │
                         ▼  Vello : Scene ─ TargetInit::SrcOver   (l'UI par-dessus)
                    canvas WebGPU ─► écran, captureStream, ou relecture GPU pour la photo
```

## Compiler et lancer

Prérequis : [rustup](https://rustup.rs) et la cible WebAssembly.

```bash
rustup target add wasm32-unknown-unknown

# wasm-pack : prendre le binaire précompilé, PAS `cargo install` (qui le compile, plusieurs minutes)
curl -sL https://github.com/rustwasm/wasm-pack/releases/download/v0.15.0/wasm-pack-v0.15.0-x86_64-unknown-linux-musl.tar.gz \
  -o /tmp/wp.tgz && tar xzf /tmp/wp.tgz -C /tmp && cp /tmp/wasm-pack-*/wasm-pack ~/.cargo/bin/

wasm-pack build --release --target web     # -> pkg/
python3 serve.py 8080 .                    # http://localhost:8080
```

`serve.py` sert `application/wasm` pour les `.wasm` — un `python3 -m http.server` ne le fait pas
toujours, et un mauvais `Content-Type` casse le chargement.

> **WebGPU exige un contexte sécurisé.** `http://localhost` en fait partie ; `http://<IP-du-réseau>`
> **non** — WebGPU y disparaît silencieusement. Pour tester depuis un autre appareil, il faut du
> HTTPS.

## 🔬 Comment c'est vérifié

Le piège classique est de croire une capture d'écran. Sur un canvas WebGPU, `canvas.toBlob()` et
`createImageBitmap(canvas)` renvoient **du noir en headless** (mesuré : 0 pixel non noir sur 576 000)
alors que l'application fonctionne parfaitement. Aucune preuve ne peut venir de là.

La vérification lit donc la mémoire du GPU :

1. **Auto-test du shader au démarrage.** On rend une couleur connue à travers la passe colorimétrique
   et on relit le pixel. La valeur attendue est calculée à la main, c'est donc une vérification
   arithmétique et pas une impression :

   | Cas | Attendu | Obtenu |
   |---|---|---|
   | identité (rouge pur) | `(255, 0, 0)` | `(255, 0, 0)` |
   | noir et blanc | `(54, 54, 54)` | `(54, 54, 54)` |
   | sépia | `(100, 89, 69)` | `(100, 89, 69)` |

2. **Preuve du filtre de bout en bout.** On capture deux photos — filtre coupé, puis noir et blanc —
   et on mesure la **chroma moyenne** `|R−G| + |G−B|` dans le PNG produit :

   - image brute : **303,0**
   - noir et blanc : **0,0** (exactement gris)

   La caméra traverse donc bien tout le pipeline jusqu'au fichier.

3. **Relecture GPU pleine taille** (900×640, 2 Mo) chronométrée au démarrage : **3–4 ms**.

Résultat sur la session de test : **3/3 auto-tests OK, 0 erreur WebGPU**.

## ⚠️ Deux pièges trouvés en chemin

### 1. La copie directe `<video>` → texture n'est pas portable

C'est pourtant le chemin prévu par l'API, et il est le plus rapide. Mais sur certains backends
(Dawn sur SwiftShader), `copyExternalImageToTexture` avec une source **vidéo** échoue — alors que
`ImageData` et `ImageBitmap` passent sur le **même appareil**. Vérifié en **WebGPU brut, sans Rust** :
le problème est dans le backend, pas dans les liaisons.

`index.html` **sonde donc la capacité au démarrage** (copie d'une texture 16×16 dans une portée
d'erreur de validation) et, si elle échoue, bascule automatiquement sur un **canvas 2D caché**
alimenté par `drawImage`, que le GPU lit ensuite. L'application marche partout et conserve le chemin
direct quand il existe. Le verdict est passé à Rust, qui choisit la source.

### 2. La latence de `map_async` est un artefact de file, pas un bug

Le **même** readback de 2 Mo :

- file vide, juste après le démarrage → **3 ms** ;
- dans une boucle `requestAnimationFrame` → **20 à 41 s**.

`mapAsync` ne se résout que lorsque le travail du tampon est terminé, et ce travail attend **derrière
tout ce qui est déjà dans la file**. Un rastériseur **logiciel** ne tient pas le 60 Hz : l'arriéré
grossit d'environ 2× le temps réel (mesuré : 22 s après 2,7 s de boucle, 41 s après 8,8 s). Sur un
vrai GPU, chaque image coûte ~1 ms, il n'y a pas d'arriéré, et la lecture reste en millisecondes.

Leçon retenue : **chronométrer chaque étape** avant d'accuser le code. Deux mesures aident malgré
tout sur matériel modeste : cesser de soumettre pendant qu'une capture est en vol (la file se vide),
et envoyer la capture dans **son propre command buffer, qui ne touche jamais la surface** — une
soumission contenant la texture de surface ne se termine qu'une fois le canvas composé.

## Portée

- **Chrome / Edge / Chrome Android** : `navigator.gpu` par défaut. **Safari iOS 26+** : par défaut.
  **Firefox** : encore derrière un drapeau sur la plupart des plateformes.
- Ce n'est **pas** portable vers les magasins d'applications mobiles (du wasm dans une WebView ajoute
  une couche) : pour ça, compiler Rust en **natif** mobile.
- Le **même code de rendu** compile en binaire de bureau (Vulkan / Metal / DX12) — seuls la fenêtre
  (`winit`) et le collage WebAssembly changent.

## Suite possible

- Filtres : LUT / étalonnage avancé, bloom, pixelisation.
- Texte dessiné dans Vello (`GlyphRunBuilder`, en embarquant une police) pour supprimer la légende
  HTML.
- Cible native : `winit` + le même renderer, sans navigateur.

## Piles

`wgpu` 30 · `vello_gpu` 0.3 · `vello_common` 0.3 · `wasm-bindgen` 0.2 · `bytemuck` 1

## Licence

MIT — voir [LICENSE](LICENSE).
