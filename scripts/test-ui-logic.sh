#!/usr/bin/env bash
# Teste la logique d'interface (mise en page + gestes) sans navigateur ni GPU.
#
# Pourquoi un script plutot que `cargo test` : la crate cible le web, et `src/lib.rs`
# utilise des API de wgpu indisponibles hors wasm (`SurfaceTarget::Canvas`,
# `copy_external_image_to_texture`) — `cargo test` echoue donc a la compilation et les
# tests `#[cfg(test)]` ne s'executeraient jamais.
#
# Or `src/ui/layout.rs` et `src/ui/gesture.rs` sont du Rust pur, sans aucune dependance :
# on peut les compiler seuls avec `rustc` et les executer pour de vrai.
set -euo pipefail
cd "$(dirname "$0")/.."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cp src/ui/layout.rs src/ui/gesture.rs "$tmp/"
printf '#[path="layout.rs"] mod layout;\n#[path="gesture.rs"] mod gesture;\n' > "$tmp/harness.rs"

rustc --test "$tmp/harness.rs" -o "$tmp/t"
"$tmp/t"
