/*
 * erosion.c — an island is born, then ten million raindrops carve it.
 *
 * A single-file C99 program (libm only, OpenMP optional) that
 *   1. generates a landscape with domain-warped gradient noise + ridged multifractals,
 *   2. erodes it with a particle-based hydraulic erosion simulation
 *      (every droplet picks up sediment on steep ground, drops it where it slows down),
 *   3. renders it with a heightfield ray marcher: soft shadows, golden-hour sky with clouds,
 *      aerial perspective, slope/altitude materials, sea with Fresnel reflections,
 *   4. writes PNG files with its own PNG encoder.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o erosion erosion.c -lm
 * Run:    ./erosion                       -> render.png (1280x720)
 *         ./erosion --before              -> also before.png (same view, no erosion)
 *         ./erosion --seed 42 -d 3.0      -> other island, 3 million droplets
 *         ./erosion --frames 60 -o fly    -> fly_000.png ... a circling fly-over
 *         ./erosion -n 512 -d 0.6 -s 1    -> quick preview
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PI 3.14159265358979f

/* ---------------------------------------------------------------- math */
typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { V3 r = {x, y, z}; return r; }
static V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 mul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static V3 vmul(V3 a, V3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static float len(V3 a) { return sqrtf(dot(a, a)); }
static V3 norm(V3 a) { float l = len(a); return l > 0 ? mul(a, 1.0f / l) : a; }
static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static float sat(float x) { return clampf(x, 0.0f, 1.0f); }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static V3 mix3(V3 a, V3 b, float t) { return add(mul(a, 1.0f - t), mul(b, t)); }
/* smooth ramp from a to b (works for a > b too) */
static float smooth01(float a, float b, float x) {
    float t = sat((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

/* --------------------------------------------------------------- noise */
static float grad_tab[256][2];
static void noise_init(void) {
    for (int i = 0; i < 256; i++) {
        float a = (float)i * (2.0f * PI / 256.0f) + 0.37f * sinf((float)i * 12.9898f);
        grad_tab[i][0] = cosf(a); grad_tab[i][1] = sinf(a);
    }
}
static uint32_t ihash(int x, int y, uint32_t s) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + s * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
static float gcorner(int ix, int iy, float dx, float dy, uint32_t s) {
    uint32_t h = ihash(ix, iy, s) & 255;
    return grad_tab[h][0] * dx + grad_tab[h][1] * dy;
}
/* gradient (Perlin-style) noise, roughly in [-1, 1] */
static float gnoise(float x, float y, uint32_t s) {
    int xi = (int)floorf(x), yi = (int)floorf(y);
    float fx = x - xi, fy = y - yi, u = fade(fx), v = fade(fy);
    float a = gcorner(xi, yi, fx, fy, s), b = gcorner(xi + 1, yi, fx - 1, fy, s);
    float c = gcorner(xi, yi + 1, fx, fy - 1, s), d = gcorner(xi + 1, yi + 1, fx - 1, fy - 1, s);
    return 1.45f * mixf(mixf(a, b, u), mixf(c, d, u), v);
}
static float fbm2(float x, float y, int oct, uint32_t s) {
    float a = 0.5f, sum = 0;
    for (int i = 0; i < oct; i++) {
        sum += a * gnoise(x, y, s + i * 31u);
        x = x * 2.02f + 17.3f; y = y * 2.02f - 9.1f; a *= 0.5f;
    }
    return sum;
}
static float ridged(float x, float y, int oct, uint32_t s) {
    float a = 0.5f, sum = 0, w = 1.0f;
    for (int i = 0; i < oct; i++) {
        float r = 1.0f - fabsf(gnoise(x, y, s + i * 57u));
        r *= r; r *= w;
        w = sat(r * 2.0f);
        sum += r * a;
        x = x * 2.1f + 5.7f; y = y * 2.1f - 3.3f; a *= 0.5f;
    }
    return sum;
}

/* ----------------------------------------------------------- terrain */
static void generate_terrain(float *h, int N, uint32_t seed) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float px = (float)x / N * 4.0f, py = (float)y / N * 4.0f;
            float wx = fbm2(px + 5.2f, py + 1.3f, 4, seed + 1), wy = fbm2(px + 1.7f, py + 9.2f, 4, seed + 2);
            float qx = px + wx * 1.1f, qy = py + wy * 1.1f;
            float base = 0.5f + 0.5f * fbm2(qx * 0.9f, qy * 0.9f, 6, seed + 3);
            float rid = ridged(qx * 1.3f + 3.1f, qy * 1.3f - 1.7f, 5, seed + 4);
            float mask = smooth01(-0.25f, 0.45f, fbm2(px * 0.55f + 3.0f, py * 0.55f - 2.0f, 3, seed + 5));
            float v = 0.62f * base + 0.50f * rid * (0.25f + 0.75f * mask);
            /* island: fall to the sea at the borders */
            float fx = (float)x / (N - 1), fy = (float)y / (N - 1);
            float d = fminf(fminf(fx, 1 - fx), fminf(fy, 1 - fy));
            float cx = fx - 0.5f, cy = fy - 0.5f;
            float radial = 1.0f - smooth01(0.28f, 0.52f, sqrtf(cx * cx + cy * cy));
            v *= smooth01(0.0f, 0.16f, d) * (0.35f + 0.65f * radial);
            h[(size_t)y * N + x] = v;
        }
    float lo = 1e9f, hi = -1e9f;
    for (size_t i = 0; i < (size_t)N * N; i++) { lo = fminf(lo, h[i]); hi = fmaxf(hi, h[i]); }
    for (size_t i = 0; i < (size_t)N * N; i++) h[i] = powf((h[i] - lo) / (hi - lo), 1.25f);
}

/* --------------------------------------------- hydraulic erosion (particles) */
static uint32_t rng_state = 2463534242u;
static float rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (rng_state & 0xffffff) / 16777216.0f;
}

typedef struct { int off; float w; } Brush;

static void erode(float *m, float *flow, int N, long drops) {
    const int R = 3, MAXLIFE = 70;
    const float inertia = 0.05f, capacity_k = 5.0f, min_slope = 0.004f, deposit_k = 0.3f,
                erode_k = 0.3f, evaporate = 0.012f, gravity = 4.0f;
    /* erosion brush: spread the removal over a small disc so valleys get width */
    Brush brush[64]; int nb = 0; float wsum = 0;
    for (int dy = -R; dy <= R; dy++)
        for (int dx = -R; dx <= R; dx++) {
            float d = sqrtf((float)(dx * dx + dy * dy));
            if (d <= R) { brush[nb].off = dy * N + dx; brush[nb].w = 1.0f - d / R; wsum += brush[nb].w; nb++; }
        }
    for (int i = 0; i < nb; i++) brush[i].w /= wsum;

    for (long it = 0; it < drops; it++) {
        float x = R + 1 + rnd() * (N - 2 * R - 3), y = R + 1 + rnd() * (N - 2 * R - 3);
        float dx = 0, dy = 0, speed = 1.0f, water = 1.0f, sed = 0.0f;
        for (int life = 0; life < MAXLIFE; life++) {
            int xi = (int)x, yi = (int)y; float u = x - xi, v = y - yi;
            size_t idx = (size_t)yi * N + xi;
            float hnw = m[idx], hne = m[idx + 1], hsw = m[idx + N], hse = m[idx + N + 1];
            float gx = (hne - hnw) * (1 - v) + (hse - hsw) * v;
            float gy = (hsw - hnw) * (1 - u) + (hse - hne) * u;
            float h0 = hnw * (1 - u) * (1 - v) + hne * u * (1 - v) + hsw * (1 - u) * v + hse * u * v;

            dx = dx * inertia - gx * (1 - inertia);
            dy = dy * inertia - gy * (1 - inertia);
            float l = sqrtf(dx * dx + dy * dy);
            if (l < 1e-6f) { float a = rnd() * 2 * PI; dx = cosf(a); dy = sinf(a); }
            else { dx /= l; dy /= l; }
            x += dx; y += dy;
            if (x < R + 1 || x >= N - R - 2 || y < R + 1 || y >= N - R - 2) break;

            int xn = (int)x, yn = (int)y; float un = x - xn, vn = y - yn;
            size_t in = (size_t)yn * N + xn;
            float h1 = m[in] * (1 - un) * (1 - vn) + m[in + 1] * un * (1 - vn) + m[in + N] * (1 - un) * vn + m[in + N + 1] * un * vn;
            float dh = h1 - h0;
            flow[idx] += water;

            float cap = fmaxf(-dh, min_slope) * speed * water * capacity_k;
            if (sed > cap || dh > 0) {
                float amt = (dh > 0) ? fminf(dh, sed) : (sed - cap) * deposit_k;
                sed -= amt;
                m[idx] += amt * (1 - u) * (1 - v); m[idx + 1] += amt * u * (1 - v);
                m[idx + N] += amt * (1 - u) * v;   m[idx + N + 1] += amt * u * v;
            } else {
                float amt = fminf((cap - sed) * erode_k, -dh);
                for (int b = 0; b < nb; b++) {
                    size_t c = idx + brush[b].off;
                    float e = amt * brush[b].w;
                    if (e > m[c]) e = m[c];
                    m[c] -= e; sed += e;
                }
            }
            speed = sqrtf(fmaxf(0.0f, speed * speed + dh * gravity));
            water *= 1.0f - evaporate;
        }
    }
}

/* ------------------------------------------------------- the world */
static int G_N;
static float *G_H, *G_F;       /* heights in [0,1], normalised river flow in [0,1] */
static float G_HS, G_SEA;      /* vertical scale (world units), sea level as a height fraction */
static V3 SUN;                 /* unit vector toward the sun */
static const V3 SUN_COL = {3.1f, 1.7f, 0.85f};

static float Ht(float x, float z) {
    if (x < 0 || z < 0 || x > G_N - 1.001f || z > G_N - 1.001f) return 0.0f;
    int xi = (int)x, zi = (int)z; float u = x - xi, v = z - zi;
    const float *h = G_H + (size_t)zi * G_N + xi;
    return ((h[0] * (1 - u) + h[1] * u) * (1 - v) + (h[G_N] * (1 - u) + h[G_N + 1] * u) * v) * G_HS;
}
static float Fl(float x, float z) {
    x = clampf(x, 0, G_N - 1.001f); z = clampf(z, 0, G_N - 1.001f);
    int xi = (int)x, zi = (int)z; float u = x - xi, v = z - zi;
    const float *h = G_F + (size_t)zi * G_N + xi;
    return (h[0] * (1 - u) + h[1] * u) * (1 - v) + (h[G_N] * (1 - u) + h[G_N + 1] * u) * v;
}
static V3 terrain_normal(float x, float z) {
    return norm(v3(Ht(x - 1, z) - Ht(x + 1, z), 2.0f, Ht(x, z - 1) - Ht(x, z + 1)));
}

/* ------------------------------------------------------------ sky */
static V3 sky_base(V3 rd) {
    float y = sat(rd.y);
    float sd = sat(dot(rd, SUN));
    V3 zen = v3(0.07f, 0.14f, 0.36f), hor = v3(0.80f, 0.50f, 0.36f);
    V3 c = mix3(hor, zen, powf(y, 0.38f));
    c = add(c, mul(v3(1.0f, 0.42f, 0.14f), powf(sd, 5.0f) * 0.75f));
    return c;
}
static V3 sky(V3 ro, V3 rd) {
    V3 c = sky_base(rd);
    float sd = sat(dot(rd, SUN));
    c = add(c, mul(v3(1.0f, 0.75f, 0.45f), powf(sd, 900.0f) * 40.0f));
    if (rd.y > 0.01f) {
        float t = (2200.0f - ro.y) / rd.y;
        float px = (ro.x + rd.x * t) * 0.0011f, pz = (ro.z + rd.z * t) * 0.0011f;
        float d = smooth01(0.02f, 0.62f, 0.5f + 0.9f * fbm2(px + 11.0f, pz + 4.0f, 5, 99));
        float lit = powf(sd, 2.0f);
        V3 cc = mix3(v3(0.30f, 0.22f, 0.32f), v3(1.7f, 0.85f, 0.50f), sat(0.15f + 0.9f * lit));
        c = mix3(c, cc, d * smooth01(0.01f, 0.25f, rd.y) * 0.9f);
    }
    return c;
}

/* ------------------------------------------------------- shading */
static float soft_shadow(V3 p, V3 L) {
    float res = 1.0f, t = 1.5f;
    for (int i = 0; i < 110; i++) {
        V3 q = add(p, mul(L, t));
        if (q.y > G_HS * 1.02f || q.x < -50 || q.z < -50 || q.x > G_N + 50 || q.z > G_N + 50) break;
        float d = q.y - Ht(q.x, q.z);
        if (d < 0.0f) return 0.0f;
        res = fminf(res, 9.0f * d / t);
        if (res < 0.004f) return 0.0f;
        t += fmaxf(0.9f, d * 0.55f);
    }
    return smooth01(0.0f, 1.0f, res);
}

static V3 terrain_albedo(V3 p, V3 n) {
    float hf = p.y / G_HS;
    float nz = fbm2(p.x * 0.045f, p.z * 0.045f, 4, 7);
    float slope = n.y;
    V3 rock = mix3(v3(0.27f, 0.23f, 0.21f), v3(0.50f, 0.41f, 0.34f), sat(0.5f + 0.5f * sinf(p.y * 0.30f + nz * 3.0f)));
    V3 grass = mix3(v3(0.10f, 0.19f, 0.05f), v3(0.34f, 0.32f, 0.12f), sat(0.5f + nz * 0.9f + (hf - 0.35f) * 1.4f));
    V3 sand = v3(0.64f, 0.56f, 0.40f), snow = v3(0.93f, 0.94f, 1.0f);
    float rockw = fmaxf(smooth01(0.82f, 0.60f, slope), smooth01(0.42f, 0.62f, hf + nz * 0.05f));
    V3 c = mix3(grass, rock, rockw);
    c = mix3(c, sand, smooth01(G_SEA + 0.030f, G_SEA + 0.006f, hf) * (1.0f - 0.5f * rockw));
    float snowline = 0.66f + nz * 0.04f;
    c = mix3(c, snow, smooth01(snowline, snowline + 0.05f, hf) * smooth01(0.45f, 0.78f, slope));
    float wet = smooth01(0.45f, 0.95f, Fl(p.x, p.z));
    return mul(c, 1.0f - 0.38f * wet);
}

/* fine-scale bump + colour variation the 1-unit heightmap can't hold */
static float micro(float x, float z) { return fbm2(x * 0.5f, z * 0.5f, 3, 41); }
static V3 detail_normal(V3 p, V3 n) {
    float e = 0.5f, b0 = micro(p.x, p.z);
    float bx = (micro(p.x + e, p.z) - b0) / e, bz = (micro(p.x, p.z + e) - b0) / e;
    float amp = 0.35f + 1.2f * (1.0f - n.y);
    return norm(v3(n.x - bx * amp, n.y, n.z - bz * amp));
}

static V3 shade_terrain(V3 p, V3 n0) {
    V3 n = detail_normal(p, n0);
    V3 alb = terrain_albedo(p, n0);
    alb = mul(alb, 0.78f + 0.44f * (0.5f + 0.5f * fbm2(p.x * 0.9f + 3.0f, p.z * 0.9f, 3, 43)));
    float ndl = sat(dot(n, SUN));
    float sh = ndl > 0.0f ? soft_shadow(add(p, mul(n, 0.8f)), SUN) : 0.0f;
    /* cheap curvature ambient occlusion */
    float avg = 0.25f * (Ht(p.x + 6, p.z) + Ht(p.x - 6, p.z) + Ht(p.x, p.z + 6) + Ht(p.x, p.z - 6));
    float ao = 1.0f - clampf((avg - p.y) * 0.05f, 0.0f, 0.65f);
    V3 amb = mix3(v3(0.10f, 0.08f, 0.07f), v3(0.20f, 0.26f, 0.42f), n.y * 0.5f + 0.5f);
    V3 light = add(mul(SUN_COL, ndl * sh), mul(amb, ao));
    return vmul(alb, light);
}

static V3 render_ray(V3 ro, V3 rd) {
    const float yS = G_SEA * G_HS, tmax = 9000.0f;
    float t = 0.5f, last = 0.5f; int hit = 0;
    for (int i = 0; i < 1600; i++) {
        V3 p = add(ro, mul(rd, t));
        if (p.y > G_HS * 1.01f && rd.y >= 0) break;
        float d = p.y - Ht(p.x, p.z);
        if (d < 0.0f) { hit = 1; break; }
        last = t;
        t += clampf(0.4f * d, 0.35f, 60.0f);
        if (t > tmax) break;
    }
    float t_hit = tmax;
    V3 p_hit = ro, n_hit = v3(0, 1, 0);
    if (hit) {
        float lo = last, hi = t;
        for (int k = 0; k < 10; k++) {
            float mid = 0.5f * (lo + hi);
            V3 p = add(ro, mul(rd, mid));
            if (p.y - Ht(p.x, p.z) < 0.0f) hi = mid; else lo = mid;
        }
        t_hit = 0.5f * (lo + hi);
        p_hit = add(ro, mul(rd, t_hit));
        n_hit = terrain_normal(p_hit.x, p_hit.z);
    }
    float t_w = (rd.y < 0 && ro.y > yS) ? (yS - ro.y) / rd.y : tmax * 2.0f;
    int water = t_w < t_hit;
    int terrain = hit;
    V3 col;
    float t_surf;

    if (water) {
        t_surf = t_w;
        V3 pw = add(ro, mul(rd, t_w));
        float nx = gnoise(pw.x * 0.06f, pw.z * 0.06f, 11) * 0.10f + gnoise(pw.x * 0.27f, pw.z * 0.27f, 12) * 0.05f;
        float nzz = gnoise(pw.x * 0.06f + 7.0f, pw.z * 0.06f, 13) * 0.10f + gnoise(pw.x * 0.27f, pw.z * 0.27f + 3.0f, 14) * 0.05f;
        V3 n = norm(v3(nx, 1.0f, nzz));
        V3 R = sub(rd, mul(n, 2.0f * dot(rd, n)));
        if (R.y < 0.02f) R.y = 0.02f;
        R = norm(R);
        float F = 0.02f + 0.98f * powf(1.0f - sat(dot(mul(rd, -1.0f), n)), 5.0f);
        V3 refl = sky(pw, R);
        float depth = terrain ? (yS - p_hit.y) : 400.0f;
        V3 deep = vmul(v3(0.012f, 0.06f, 0.09f), add(mul(SUN_COL, 0.10f + 0.25f * SUN.y), v3(0.5f, 0.6f, 0.8f)));
        V3 under;
        if (terrain && depth < 60.0f) {
            V3 alb = terrain_albedo(p_hit, n_hit);
            V3 lit = add(mul(SUN_COL, 0.55f * SUN.y), v3(0.16f, 0.2f, 0.3f));
            V3 trans = v3(expf(-depth * 0.11f), expf(-depth * 0.045f), expf(-depth * 0.028f));
            float mean = (trans.x + trans.y + trans.z) / 3.0f;
            under = add(vmul(vmul(alb, lit), trans), mul(deep, 1.0f - mean));
        } else under = deep;
        col = add(mul(under, 1.0f - F), mul(refl, F));
        col = add(col, mul(SUN_COL, powf(sat(dot(R, SUN)), 300.0f) * 5.0f));
        if (terrain && depth < 1.6f) {
            float foam = (1.0f - depth / 1.6f) * (0.55f + 0.45f * gnoise(pw.x * 0.5f, pw.z * 0.5f, 21));
            col = mix3(col, v3(0.95f, 0.9f, 0.85f), sat(foam) * 0.55f);
        }
    } else if (hit) {
        t_surf = t_hit;
        col = shade_terrain(p_hit, n_hit);
    } else {
        /* open sky (or ocean beyond the world, handled above) */
        return sky(ro, rd);
    }
    /* aerial perspective: distance haze that picks up the colour of the sky */
    float T = expf(-t_surf * 0.00042f);
    V3 haze = sky_base(v3(rd.x, fmaxf(rd.y, 0.0f) * 0.4f, rd.z));
    haze = mul(haze, 0.9f);
    return add(mul(col, T), mul(haze, 1.0f - T));
}

/* ------------------------------------------------------------ output */
static float aces(float x) {
    x *= 0.9f;
    return sat((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

static uint32_t crc_tab[256];
static void crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_tab[n] = c;
    }
}
static uint32_t crc_update(uint32_t c, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ b[i]) & 0xff] ^ (c >> 8);
    return c;
}
static void put32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void write_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t hdr[8]; put32(hdr, n); memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    uint32_t c = crc_update(0xffffffffu, (const uint8_t *)type, 4);
    if (n) c = crc_update(c, data, n);
    uint8_t tail[4]; put32(tail, ~c);
    fwrite(tail, 1, 4, f);
}
static int write_png(const char *path, int w, int h, const uint8_t *rgb) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    crc_init();
    size_t rowb = (size_t)w * 3 + 1, raw_n = rowb * h;
    uint8_t *raw = malloc(raw_n);
    for (int y = 0; y < h; y++) { raw[y * rowb] = 0; memcpy(raw + y * rowb + 1, rgb + (size_t)y * w * 3, (size_t)w * 3); }
    size_t nblocks = raw_n / 65535 + 1, z_n = 2 + raw_n + nblocks * 5 + 4;
    uint8_t *z = malloc(z_n), *zp = z;
    *zp++ = 0x78; *zp++ = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw_n; i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw_n;) {
        size_t n = raw_n - off > 65535 ? 65535 : raw_n - off;
        *zp++ = (off + n >= raw_n) ? 1 : 0;
        *zp++ = n & 0xff; *zp++ = n >> 8; *zp++ = ~n & 0xff; *zp++ = (~n >> 8) & 0xff;
        memcpy(zp, raw + off, n); zp += n; off += n;
    }
    put32(zp, (b << 16) | a); zp += 4;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13]; put32(ihdr, w); put32(ihdr + 4, h);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    write_chunk(f, "IHDR", ihdr, 13);
    write_chunk(f, "IDAT", z, (uint32_t)(zp - z));
    write_chunk(f, "IEND", NULL, 0);
    fclose(f); free(raw); free(z);
    return 1;
}

/* ----------------------------------------------------------- camera + frame */
static void camera_for(float a, V3 *eye, V3 *target) {
    float c = G_N * 0.5f, R = G_N * 0.46f;
    V3 e = v3(c + R * cosf(a), 0, c + R * sinf(a));
    e.y = fmaxf(G_HS * 1.05f, Ht(e.x, e.z) + 60.0f);
    *eye = e;
    *target = v3(c, G_HS * 0.10f, c);
}

static void render_frame(float a, int W, int H, int SS, uint8_t *out) {
    int RW = W * SS, RH = H * SS;
    float *color = malloc((size_t)RW * RH * 3 * sizeof(float));
    V3 eye, target; camera_for(a, &eye, &target);
    V3 f = norm(sub(target, eye)), r = norm(cross(v3(0, 1, 0), f)), u = cross(f, r);
    float th = tanf(0.44f), aspect = (float)W / H;
#pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < RH; y++)
        for (int x = 0; x < RW; x++) {
            float nx = ((x + 0.5f) / RW) * 2 - 1, ny = 1 - ((y + 0.5f) / RH) * 2;
            V3 rd = norm(add(f, add(mul(r, nx * aspect * th), mul(u, ny * th))));
            V3 c = render_ray(eye, rd);
            float *p = color + ((size_t)y * RW + x) * 3;
            p[0] = c.x; p[1] = c.y; p[2] = c.z;
        }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float acc[3] = {0, 0, 0};
            for (int sy = 0; sy < SS; sy++)
                for (int sx = 0; sx < SS; sx++) {
                    const float *p = color + ((size_t)(y * SS + sy) * RW + (x * SS + sx)) * 3;
                    for (int k = 0; k < 3; k++) acc[k] += powf(aces(p[k]), 1.0f / 2.2f);
                }
            float nx = ((x + 0.5f) / W) * 2 - 1, ny = ((y + 0.5f) / H) * 2 - 1;
            float vig = 1.0f - 0.25f * (nx * nx + ny * ny) * 0.5f;
            for (int k = 0; k < 3; k++)
                out[((size_t)y * W + x) * 3 + k] = (uint8_t)(sat(acc[k] / (SS * SS) * vig) * 255.0f + 0.5f);
        }
    free(color);
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int W = 1280, H = 720, SS = 2, N = 1024, frames = 0, before = 0;
    uint32_t seed = 7; double drops_m = 3.0; float angle = 0.55f;
    const char *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) SS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) drops_m = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--angle") && i + 1 < argc) angle = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--before")) before = 1;
        else { fprintf(stderr, "usage: %s [-n size] [-d million_droplets] [--seed S] [--angle A] [-w W] [-h H] [-s ssaa] [--frames F] [-o name] [--before]\n", argv[0]); return 1; }
    }
    if (N < 128 || N > 4096 || W < 16 || H < 16 || SS < 1 || SS > 6) { fprintf(stderr, "bad size\n"); return 1; }

    noise_init();
    rng_state = 2463534242u ^ (seed * 2654435761u);
    SUN = norm(v3(-0.62f, 0.20f, 0.50f));
    G_N = N; G_HS = 0.20f * N; G_SEA = 0.30f;

    size_t cells = (size_t)N * N;
    float *h = malloc(cells * sizeof(float)), *h0 = malloc(cells * sizeof(float));
    float *flow = calloc(cells, sizeof(float)), *fl_norm = malloc(cells * sizeof(float));

    double t0 = now();
    generate_terrain(h, N, seed);
    memcpy(h0, h, cells * sizeof(float));
    printf("terrain generated      %.2fs\n", now() - t0);

    long drops = (long)(drops_m * 1e6 * ((double)N * N / (1024.0 * 1024.0)) * 1.0);
    if (drops_m > 0) {
        t0 = now();
        erode(h, flow, N, drops);
        printf("eroded with %ld droplets  %.2fs\n", drops, now() - t0);
    }
    float fmax_ = 1e-6f;
    for (size_t i = 0; i < cells; i++) fmax_ = fmaxf(fmax_, flow[i]);
    float k = 40.0f / fmax_;
    for (size_t i = 0; i < cells; i++) fl_norm[i] = sat(logf(1.0f + flow[i] * k) / logf(41.0f) * 1.15f - 0.1f);
    G_F = fl_norm;

    uint8_t *img = malloc((size_t)W * H * 3);
    if (before) {
        G_H = h0;
        t0 = now();
        render_frame(angle, W, H, SS, img);
        write_png("before.png", W, H, img);
        printf("before.png             %.2fs\n", now() - t0);
    }
    G_H = h;
    int n = frames > 0 ? frames : 1;
    for (int i = 0; i < n; i++) {
        char name[512];
        float a = frames > 0 ? angle + 2.0f * PI * i / frames : angle;
        if (frames > 0) snprintf(name, sizeof name, "%s_%03d.png", out ? out : "frame", i);
        else snprintf(name, sizeof name, "%s", out ? out : "render.png");
        t0 = now();
        render_frame(a, W, H, SS, img);
        if (!write_png(name, W, H, img)) { fprintf(stderr, "cannot write %s\n", name); return 1; }
        printf("%s  %dx%d  %.2fs\n", name, W, H, now() - t0);
    }
    return 0;
}
