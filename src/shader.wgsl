// Filtres WGSL de vello-cam — deux passes chaînées (ping-pong).
//   fs_grade   : rendu colorimétrique  (caméra -> texture intermédiaire)
//   fs_spatial : filtres spatiaux      (texture intermédiaire -> surface)
// Un seul triangle plein écran, aucun buffer de sommets : la position vient de
// @builtin(vertex_index). Texture en space "écran" : V en haut = 0.

struct Filters {
  brightness  : f32,
  contrast    : f32,
  saturation  : f32,
  temperature : f32,
  sepia       : f32,
  grayscale   : f32,
  vignette    : f32,
  grain       : f32,
  blur        : f32,
  aberration  : f32,
  fisheye     : f32,
  time        : f32,
  uvscale     : vec2<f32>,
  texel       : vec2<f32>,
};

@group(0) @binding(0) var<uniform> F : Filters;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var tex : texture_2d<f32>;

struct VOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VOut {
  var pts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0));
  let p = pts[i];
  var o : VOut;
  o.pos = vec4<f32>(p, 0.0, 1.0);
  o.uv  = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return o;
}

fn lum(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn cl01(v : vec2<f32>) -> vec2<f32> { return clamp(v, vec2<f32>(0.0), vec2<f32>(1.0)); }

fn hash21(p : vec2<f32>) -> f32 {
  var q = fract(p * vec2<f32>(123.34, 456.21));
  q = q + dot(q, q + 45.32);
  return fract(q.x * q.y);
}

// uv écran -> uv source : fisheye puis cadrage "cover" (l'image remplit, on rogne).
fn to_src(uv0 : vec2<f32>) -> vec2<f32> {
  var uv = uv0;
  if (F.fisheye > 0.0) {
    let c = uv - vec2<f32>(0.5);
    let k = 1.0 + F.fisheye * 1.5 * dot(c, c);
    uv = c * k + vec2<f32>(0.5);
  }
  uv = (uv - vec2<f32>(0.5)) * F.uvscale + vec2<f32>(0.5);
  return cl01(uv);
}

// ---- passe 1 : colorimétrie (source = texture caméra) ----
@fragment
fn fs_grade(in : VOut) -> @location(0) vec4<f32> {
  let uv = to_src(in.uv);
  var col = textureSampleLevel(tex, samp, uv, 0.0).rgb;

  // température (chaud / froid)
  col.r = col.r * (1.0 + F.temperature * 0.18);
  col.b = col.b * (1.0 - F.temperature * 0.18);
  // saturation
  col = mix(vec3<f32>(lum(col)), col, F.saturation);
  // contraste + luminosité
  col = (col - vec3<f32>(0.5)) * F.contrast + vec3<f32>(0.5) + vec3<f32>(F.brightness);
  // noir & blanc
  col = mix(col, vec3<f32>(lum(col)), F.grayscale);
  // sépia
  let sc = vec3<f32>(
    dot(col, vec3<f32>(0.393, 0.769, 0.189)),
    dot(col, vec3<f32>(0.349, 0.686, 0.168)),
    dot(col, vec3<f32>(0.272, 0.534, 0.131)));
  col = mix(col, sc, F.sepia);
  // vignette
  let d = distance(in.uv, vec2<f32>(0.5)) * 1.41421356;
  col = col * mix(1.0, smoothstep(1.05, 0.30, d), F.vignette);
  // grain
  col = col + (hash21(in.uv * 900.0 + vec2<f32>(F.time, F.time * 1.37)) - 0.5) * F.grain * 0.30;

  return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// ---- passe 2 : spatial (source = texture intermédiaire, déjà en space écran) ----
fn blur5(uv : vec2<f32>) -> vec3<f32> {
  let t = F.texel * F.blur * 4.0;
  var s = textureSampleLevel(tex, samp, uv, 0.0).rgb * 0.28;
  s = s + textureSampleLevel(tex, samp, cl01(uv + vec2<f32>(t.x, 0.0)), 0.0).rgb * 0.18;
  s = s + textureSampleLevel(tex, samp, cl01(uv - vec2<f32>(t.x, 0.0)), 0.0).rgb * 0.18;
  s = s + textureSampleLevel(tex, samp, cl01(uv + vec2<f32>(0.0, t.y)), 0.0).rgb * 0.18;
  s = s + textureSampleLevel(tex, samp, cl01(uv - vec2<f32>(0.0, t.y)), 0.0).rgb * 0.18;
  return s;
}

@fragment
fn fs_spatial(in : VOut) -> @location(0) vec4<f32> {
  let uv = cl01(in.uv);
  var col : vec3<f32>;

  if (F.aberration > 0.0) {
    let off = (uv - vec2<f32>(0.5)) * F.aberration * 0.03;
    col = vec3<f32>(
      textureSampleLevel(tex, samp, cl01(uv + off), 0.0).r,
      textureSampleLevel(tex, samp, uv, 0.0).g,
      textureSampleLevel(tex, samp, cl01(uv - off), 0.0).b);
  } else {
    col = textureSampleLevel(tex, samp, uv, 0.0).rgb;
  }

  if (F.blur > 0.0) {
    col = mix(col, blur5(uv), min(F.blur * 3.0, 1.0));
  }

  return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
