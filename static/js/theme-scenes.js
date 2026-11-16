/* ==========================================================================
   Animated background scenes — one per dataset theme (canvas 2D).
   Presentation only. Loaded before fx.js, which drives them (see initSceneFx).
   Each scene is a factory  S => ({ draw(t), click?(x, y) })  where S holds:
     ctx, W, H, t, mouse{x,y,vx}, col(alpha), burst(x,y,n)
   The engine sets ctx.globalAlpha = .6, so alphas below are "pre-fade".
   ========================================================================== */
(function () {
  "use strict";
  const TAU = Math.PI * 2;
  const rnd = (a, b) => a + Math.random() * (b - a);
  const pick = a => a[(Math.random() * a.length) | 0];
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const RGB = {
    health: "13,148,136", virus: "124,58,237", energy: "245,158,11", solar: "234,179,8",
    wind: "14,165,233", fire: "249,115,22", finance: "37,99,235", sports: "22,163,74",
    car: "79,70,229", flight: "2,132,199", education: "219,39,119", retail: "225,29,72",
    music: "147,51,234", movie: "239,68,68", water: "14,165,233", plant: "22,163,74",
    ship: "2,132,199", sensor: "6,182,212", house: "234,88,12",
  };

  const rr = (ctx, x, y, w, h, r) => {
    r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  };
  const star = (ctx, x, y, s) => {
    ctx.beginPath();
    for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4 - Math.PI / 2, r = i % 2 ? s * .32 : s; ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r); }
    ctx.closePath(); ctx.fill();
  };
  const near = S => S.mouse.x > -900;

  /* ---- water surface helpers (water + ship) ---- */
  function waveY(S, t, L, x) {
    let y = S.H * L.base + Math.sin(x * L.k + t * L.sp + L.ph) * L.amp + Math.sin(x * L.k * 2.3 - t * L.sp * 1.3) * L.amp * .4;
    const m = S.mouse;
    if (m.y > S.H * .6) { const d = (x - m.x) / 90, g = Math.exp(-d * d); y -= g * (8 + 16 * Math.sin(t * 5 - Math.abs(x - m.x) * .05)); }
    return y;
  }
  function drawWaves(S, t, layers) {
    const { ctx, W, H } = S;
    layers.forEach(L => {
      ctx.beginPath(); ctx.moveTo(0, H);
      for (let x = 0; x <= W; x += 8) ctx.lineTo(x, waveY(S, t, L, x));
      ctx.lineTo(W, H); ctx.closePath(); ctx.fillStyle = S.col(L.a); ctx.fill();
    });
  }

  const SCENES = {

    /* Three ECG heartbeat signals sweeping at different speeds; the signal swells near the cursor; click = pulse ring */
    health: S => {
      const beat = [0, 0, 0, .08, .16, .08, 0, 0, -.1, 1, -.4, 0, 0, .1, .2, .1, 0, 0, 0, 0, 0, 0];
      const L = beat.length;
      const lines = [
        { y: .56, amp: .085, P: 340, v: 120, o: 0 },
        { y: .74, amp: .105, P: 400, v: 165, o: 140 },
        { y: .90, amp: .075, P: 300, v: 210, o: 60 },
      ];
      const wave = (x, P) => { const u = (((x % P) + P) % P) / P * L, i = Math.floor(u), f = u - i; return beat[i] + (beat[(i + 1) % L] - beat[i]) * f; };
      const rings = [];
      return {
        click(x, y) { rings.push({ x, y, r: 6, a: .8 }); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          ctx.lineWidth = 2.4; ctx.lineCap = "round";
          lines.forEach((ln, li) => {
            const base = H * ln.y, amp = Math.min(80, H * ln.amp), head = (t * ln.v + ln.o) % (W + 260) - 60;
            const Y = x => {
              let a = amp; if (near(S)) { const d = (x - mouse.x) / 140; a *= 1 + .9 * Math.exp(-d * d) * (Math.abs(mouse.y - base) < 160 ? 1 : .3); }
              return base - wave(x, ln.P) * a;
            };
            const fa = li === 0 ? 1 : li === 1 ? .85 : .7;
            for (let i = 0; i < 70; i++) {
              const x0 = head - (70 - i) * 5, x1 = x0 + 5;
              ctx.strokeStyle = S.col((.1 + .8 * i / 70) * fa);
              ctx.beginPath(); ctx.moveTo(x0, Y(x0)); ctx.lineTo(x1, Y(x1)); ctx.stroke();
            }
            const hy = Y(head);
            ctx.fillStyle = S.col(.9 * fa); ctx.beginPath(); ctx.arc(head, hy, 4, 0, TAU); ctx.fill();
            ctx.strokeStyle = S.col(.35 * fa); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(head, hy, 9 + 3 * Math.sin(t * 8 + li), 0, TAU); ctx.stroke(); ctx.lineWidth = 2.4;
          });
          for (let i = rings.length - 1; i >= 0; i--) {
            const q = rings[i]; q.r += 3; q.a -= .014; if (q.a <= 0) { rings.splice(i, 1); continue; }
            ctx.strokeStyle = S.col(q.a); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(q.x, q.y, q.r, 0, TAU); ctx.stroke(); ctx.lineWidth = 2.4;
          }
        },
      };
    },

    /* Spiky virus particles drifting, fleeing the cursor; click pops them */
    virus: S => {
      const mk = () => ({ x: rnd(0, S.W), y: rnd(0, S.H), vx: rnd(-.5, .5), vy: rnd(-.5, .5), r: rnd(11, 24), a: rnd(0, TAU), w: rnd(-.012, .012), k: pick([8, 10, 12]) });
      const vs = Array.from({ length: clamp(Math.round(S.W * S.H / 85000), 6, 14) }, mk);
      return {
        click(x, y) {
          vs.forEach((v, i) => {
            if (Math.hypot(v.x - x, v.y - y) < v.r + 30) { S.burst(v.x, v.y, 12); const n = mk(); n.y = S.H + 40; n.vy = -rnd(.3, .6); vs[i] = n; }
          });
        },
        draw() {
          const { ctx, W, H, mouse } = S;
          vs.forEach(v => {
            const dx = v.x - mouse.x, dy = v.y - mouse.y, d = Math.hypot(dx, dy);
            if (d < 150 && d > 0) { v.vx += dx / d * .06; v.vy += dy / d * .06; }
            v.vx *= .995; v.vy *= .995;
            if (Math.hypot(v.vx, v.vy) < .25) { v.vx += rnd(-.02, .02); v.vy += rnd(-.02, .02); }
            v.x += v.vx; v.y += v.vy; v.a += v.w;
            if (v.x < -40) v.x = W + 40; if (v.x > W + 40) v.x = -40; if (v.y < -40) v.y = H + 40; if (v.y > H + 40) v.y = -40;
            ctx.save(); ctx.translate(v.x, v.y); ctx.rotate(v.a); ctx.lineWidth = 1.6;
            for (let i = 0; i < v.k; i++) {
              const a = i / v.k * TAU, c = Math.cos(a), s = Math.sin(a);
              ctx.strokeStyle = S.col(.55); ctx.beginPath(); ctx.moveTo(c * v.r, s * v.r); ctx.lineTo(c * (v.r + 8), s * (v.r + 8)); ctx.stroke();
              ctx.fillStyle = S.col(.55); ctx.beginPath(); ctx.arc(c * (v.r + 10), s * (v.r + 10), 2.6, 0, TAU); ctx.fill();
            }
            ctx.fillStyle = S.col(.16); ctx.strokeStyle = S.col(.55); ctx.beginPath(); ctx.arc(0, 0, v.r, 0, TAU); ctx.fill(); ctx.stroke();
            ctx.fillStyle = S.col(.35);
            [[-.3, -.2, .18], [.25, .15, .14], [0, .35, .1]].forEach(q => { ctx.beginPath(); ctx.arc(q[0] * v.r, q[1] * v.r, q[2] * v.r, 0, TAU); ctx.fill(); });
            ctx.restore();
          });
        },
      };
    },

    /* Circuit traces with current pulses, high-voltage pylons with travelling charge, sparks and lightning */
    energy: S => {
      const ps = Array.from({ length: clamp(Math.round(S.W * S.H / 60000), 10, 26) }, () => ({ x: rnd(0, S.W), y: rnd(0, S.H), vx: rnd(-.4, .4), vy: rnd(-.4, .4), r: rnd(1.4, 3) }));
      const bolts = []; let next = 1;
      const bolt = (x1, y1, x2, y2) => {
        const pts = [[x1, y1]], n = 12, nx = -(y2 - y1), ny = x2 - x1, nl = Math.hypot(nx, ny) || 1;
        for (let i = 1; i < n; i++) { const k = i / n, o = rnd(-28, 28); pts.push([x1 + (x2 - x1) * k + nx / nl * o, y1 + (y2 - y1) * k + ny / nl * o]); }
        pts.push([x2, y2]); bolts.push({ pts, life: 1 });
      };
      /* circuit traces: orthogonal polylines, each carrying 1-2 pulses */
      const paths = Array.from({ length: clamp(Math.round(S.W / 180), 4, 9) }, () => {
        let x = rnd(0, S.W), y = rnd(S.H * .04, S.H * .55), horiz = Math.random() < .5; const pts = [[x, y]];
        for (let i = 0; i < 6; i++) { const d = rnd(70, 230) * (Math.random() < .5 ? -1 : 1); if (horiz) x = clamp(x + d, 10, S.W - 10); else y = clamp(y + d, 10, S.H * .62); pts.push([x, y]); horiz = !horiz; }
        const cum = [0]; for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
        const len = cum[cum.length - 1] || 1;
        return { pts, cum, len, pulses: Array.from({ length: 2 }, (_, k) => ({ d: len * k / 2 + rnd(0, 60), v: rnd(1.6, 2.8) })) };
      });
      const at = (P, d) => {
        d = clamp(d, 0, P.len); let i = 1; while (i < P.cum.length - 1 && P.cum[i] < d) i++;
        const k = (d - P.cum[i - 1]) / ((P.cum[i] - P.cum[i - 1]) || 1);
        return [P.pts[i - 1][0] + (P.pts[i][0] - P.pts[i - 1][0]) * k, P.pts[i - 1][1] + (P.pts[i][1] - P.pts[i - 1][1]) * k];
      };
      /* pylons + sagging wires with travelling charge */
      const n = Math.ceil(S.W / 320) + 1, gap = S.W / (n - 1), ph = S.H * .24;
      const tx = i => i * gap, ty = S.H - ph;
      const arms = [[-30, 0], [0, .0], [30, 0]];
      const wires = []; for (let i = 0; i < n - 1; i++) arms.forEach(([ax], k) => wires.push({ x0: tx(i) + ax, x1: tx(i + 1) + ax, y: ty + (k === 1 ? -12 : 0), sag: 28 + k * 4, p: [rnd(0, 1), rnd(0, 1)], v: rnd(.004, .007) }));
      return {
        click(x, y) { bolt(x + rnd(-80, 80), 0, x, y); const P = paths[(Math.random() * paths.length) | 0]; P.pulses.push({ d: 0, v: 4 }); },
        draw() {
          const { ctx, W, H, mouse } = S;
          next -= 1 / 60; if (next <= 0) { next = rnd(1.6, 3.4); bolt(rnd(W * .1, W * .9), 0, rnd(W * .1, W * .9), rnd(H * .45, H * .9)); }
          /* circuit traces */
          ctx.lineJoin = "round"; ctx.lineCap = "round";
          paths.forEach(P => {
            ctx.strokeStyle = S.col(.2); ctx.lineWidth = 1.6; ctx.beginPath(); P.pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.stroke();
            ctx.fillStyle = S.col(.35); P.pts.forEach(q => { ctx.beginPath(); ctx.arc(q[0], q[1], 3, 0, TAU); ctx.fill(); });
            for (let i = P.pulses.length - 1; i >= 0; i--) {
              const u = P.pulses[i], h = at(P, u.d);
              const boost = near(S) && Math.hypot(h[0] - mouse.x, h[1] - mouse.y) < 150 ? 2.6 : 1;
              u.d += u.v * boost; if (u.d > P.len + 40) { if (P.pulses.length > 2) { P.pulses.splice(i, 1); continue; } u.d = -40; }
              for (let k = 0; k < 8; k++) { const a = at(P, u.d - k * 6), b = at(P, u.d - k * 6 + 6); ctx.strokeStyle = S.col(.9 * (1 - k / 8)); ctx.lineWidth = 3.2 - k * .3; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
              const hp = at(P, u.d); ctx.fillStyle = S.col(.95); ctx.beginPath(); ctx.arc(hp[0], hp[1], 3.6, 0, TAU); ctx.fill();
              ctx.fillStyle = S.col(.25); ctx.beginPath(); ctx.arc(hp[0], hp[1], 9, 0, TAU); ctx.fill();
            }
          });
          /* pylons */
          ctx.strokeStyle = S.col(.38); ctx.lineWidth = 2.4;
          for (let i = 0; i < n; i++) {
            const x = tx(i);
            ctx.beginPath(); ctx.moveTo(x - 18, H); ctx.lineTo(x - 4, ty); ctx.moveTo(x + 18, H); ctx.lineTo(x + 4, ty);
            for (let k = 0; k < 6; k++) { const f0 = k / 6, f1 = (k + 1) / 6, w0 = 18 - 14 * f0, w1 = 18 - 14 * f1; ctx.moveTo(x - w0, H - ph * f0); ctx.lineTo(x + w1, H - ph * f1); ctx.moveTo(x + w0, H - ph * f0); ctx.lineTo(x - w1, H - ph * f1); }
            ctx.moveTo(x - 30, ty); ctx.lineTo(x + 30, ty); ctx.moveTo(x - 22, ty + ph * .16); ctx.lineTo(x + 22, ty + ph * .16); ctx.stroke();
          }
          /* wires + charge travelling along them */
          wires.forEach(w => {
            const cx = (w.x0 + w.x1) / 2, cy = w.y + w.sag * 2, wy = w.y;
            ctx.strokeStyle = S.col(.3); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(w.x0, wy); ctx.quadraticCurveTo(cx, cy, w.x1, wy); ctx.stroke();
            w.p.forEach((p0, i) => {
              let p = p0 + w.v * (1 + (near(S) && Math.abs(mouse.x - cx) < 200 ? 2 : 0)); if (p > 1) p -= 1; w.p[i] = p;
              const q = 1 - p, x = q * q * w.x0 + 2 * q * p * cx + p * p * w.x1, y = q * q * wy + 2 * q * p * cy + p * p * wy;
              ctx.fillStyle = S.col(.28); ctx.beginPath(); ctx.arc(x, y, 8, 0, TAU); ctx.fill();
              ctx.fillStyle = S.col(.95); ctx.beginPath(); ctx.arc(x, y, 3.2, 0, TAU); ctx.fill();
            });
          });
          /* sparks */
          ps.forEach(p => {
            const dx = mouse.x - p.x, dy = mouse.y - p.y, d = Math.hypot(dx, dy);
            if (d < 240 && d > 1) { p.vx += dx / d * .06; p.vy += dy / d * .06; }
            p.vx += rnd(-.03, .03); p.vy += rnd(-.03, .03); p.vx *= .985; p.vy *= .985; p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > W) p.vx *= -1; if (p.y < 0 || p.y > H) p.vy *= -1;
            ctx.fillStyle = S.col(.8); ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, TAU); ctx.fill();
            if (d < 150) {
              ctx.strokeStyle = S.col((1 - d / 150) * .7); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(p.x, p.y);
              for (let k = 1; k < 4; k++) ctx.lineTo(p.x + dx * k / 4 + rnd(-7, 7), p.y + dy * k / 4 + rnd(-7, 7));
              ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
            }
          });
          /* lightning */
          for (let i = bolts.length - 1; i >= 0; i--) {
            const b = bolts[i]; b.life -= .035; if (b.life <= 0) { bolts.splice(i, 1); continue; }
            [[9, .22], [2.6, .95]].forEach(([w, a]) => {
              ctx.strokeStyle = S.col(b.life * a); ctx.lineWidth = w * b.life + .5; ctx.beginPath();
              b.pts.forEach((q, k) => k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.stroke();
            });
          }
        },
      };
    },

    /* Rotating sun with beams + rising light motes; sun follows the cursor a little; click = pulse */
    solar: S => {
      const motes = Array.from({ length: 26 }, () => ({ x: rnd(0, S.W), y: rnd(0, S.H), v: rnd(.15, .5), r: rnd(1.2, 3), ph: rnd(0, TAU) }));
      const pulses = [];
      return {
        click() { pulses.push({ r: 40, a: .7 }); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          const px = near(S) ? (mouse.x / W - .5) * -30 : 0, py = near(S) ? (mouse.y / H - .5) * -20 : 0;
          const R = Math.min(54, W * .06), cx = W * .82 + px, cy = H * .2 + py;
          const g = ctx.createRadialGradient(cx, cy, R * .4, cx, cy, R * 5);
          g.addColorStop(0, S.col(.4)); g.addColorStop(1, S.col(0));
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R * 5, 0, TAU); ctx.fill();
          [2.1, 2.5, 2.9].forEach((a0, i) => {
            const a = a0 + Math.sin(t * .4 + i) * .06, L = H * 1.2;
            const bg = ctx.createLinearGradient(cx, cy, cx + Math.cos(a) * L, cy + Math.sin(a) * L);
            bg.addColorStop(0, S.col(.18)); bg.addColorStop(1, S.col(0));
            ctx.fillStyle = bg; ctx.beginPath(); ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(a - .07) * L, cy + Math.sin(a - .07) * L); ctx.lineTo(cx + Math.cos(a + .07) * L, cy + Math.sin(a + .07) * L); ctx.closePath(); ctx.fill();
          });
          ctx.lineCap = "round"; ctx.lineWidth = 4;
          for (let i = 0; i < 16; i++) {
            const a = i / 16 * TAU + t * .12, l = R * (1.5 + .4 * Math.sin(t * 2 + i));
            ctx.strokeStyle = S.col(.55); ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * (R + 8), cy + Math.sin(a) * (R + 8)); ctx.lineTo(cx + Math.cos(a) * (R + l * .6), cy + Math.sin(a) * (R + l * .6)); ctx.stroke();
          }
          ctx.fillStyle = S.col(.6); ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
          motes.forEach(m => {
            m.ph += .02; m.y -= m.v; m.x += Math.sin(m.ph) * .4; if (m.y < -10) { m.y = H + 10; m.x = rnd(0, W); }
            ctx.fillStyle = S.col(.55); ctx.beginPath(); ctx.arc(m.x, m.y, m.r, 0, TAU); ctx.fill();
          });
          for (let i = pulses.length - 1; i >= 0; i--) {
            const q = pulses[i]; q.r += 6; q.a -= .014; if (q.a <= 0) { pulses.splice(i, 1); continue; }
            ctx.strokeStyle = S.col(q.a); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, q.r, 0, TAU); ctx.stroke();
          }
        },
      };
    },

    /* Wind streaks + spinning turbines that speed up with the mouse; click = gust */
    wind: S => {
      const mk = () => ({ x: rnd(-300, S.W), y: rnd(S.H * .05, S.H * .7), len: rnd(110, 260), v: rnd(2.2, 5), amp: rnd(5, 14), ph: rnd(0, TAU) });
      const st = Array.from({ length: 16 }, mk);
      const tw = [.16, .5, .84].map(f => ({ f, ang: rnd(0, TAU) }));
      let gust = 0;
      return {
        click() { gust = 14; },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          gust *= .96; const boost = 1 + Math.min(4, Math.abs(mouse.vx) * .05 + gust * .25);
          ctx.lineCap = "round"; ctx.lineWidth = 2;
          st.forEach(s => {
            s.x += s.v * boost; if (s.x - s.len > W) Object.assign(s, mk(), { x: -20 });
            for (let k = 0; k < 8; k++) {
              const x0 = s.x - s.len + s.len * k / 8, x1 = s.x - s.len + s.len * (k + 1) / 8;
              const y0 = s.y + Math.sin(x0 * .02 + s.ph + t * 1.2) * s.amp, y1 = s.y + Math.sin(x1 * .02 + s.ph + t * 1.2) * s.amp;
              ctx.strokeStyle = S.col(.1 + .5 * k / 8); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
            }
            ctx.strokeStyle = S.col(.45); ctx.beginPath(); ctx.arc(s.x, s.y + Math.sin(s.x * .02 + s.ph + t * 1.2) * s.amp - 5, 5, 1.2, 4.6); ctx.stroke();
          });
          const h = H * .3;
          tw.forEach(w => {
            const x = W * w.f; w.ang += .02 * boost;
            ctx.strokeStyle = S.col(.35); ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(x, H); ctx.lineTo(x, H - h); ctx.stroke();
            ctx.lineWidth = 5; ctx.strokeStyle = S.col(.6);
            for (let i = 0; i < 3; i++) { const a = w.ang + i * TAU / 3; ctx.beginPath(); ctx.moveTo(x, H - h); ctx.lineTo(x + Math.cos(a) * h * .42, H - h + Math.sin(a) * h * .42); ctx.stroke(); }
            ctx.fillStyle = S.col(.8); ctx.beginPath(); ctx.arc(x, H - h, 5, 0, TAU); ctx.fill();
          });
        },
      };
    },

    /* Flames licking up from the bottom + embers drawn to the cursor; click = ember burst */
    fire: S => {
      const fl = [], em = [], spots = [.12, .3, .5, .7, .88];
      return {
        click(x, y) { for (let i = 0; i < 16; i++) em.push({ x, y, vx: rnd(-2, 2), vy: rnd(-4, -1), life: 1, r: rnd(1.5, 3) }); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          for (let i = 0; i < 3; i++) fl.push({ x: pick(spots) * W + rnd(-70, 70), y: H + 10, vx: rnd(-.4, .4), vy: -rnd(1.2, 3), r: rnd(16, 34), life: 1, d: rnd(.007, .014) });
          if (Math.random() < .5) em.push({ x: rnd(0, W), y: H, vx: rnd(-.5, .5), vy: -rnd(1, 3), life: 1, r: rnd(1, 2.4) });
          while (fl.length > 220) fl.shift(); while (em.length > 120) em.shift();
          for (let i = fl.length - 1; i >= 0; i--) {
            const f = fl[i]; f.x += f.vx + Math.sin(t * 3 + f.y * .02) * .3; f.y += f.vy; f.life -= f.d; f.r *= .993;
            if (f.life <= 0) { fl.splice(i, 1); continue; }
            const k = f.life, c = `${Math.round(239 + 12 * k)},${Math.round(68 + 123 * k)},${Math.round(68 - 32 * k)}`;
            const g = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.r);
            g.addColorStop(0, `rgba(${c},${k * .42})`); g.addColorStop(1, `rgba(${c},0)`);
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(f.x, f.y, f.r, 0, TAU); ctx.fill();
          }
          for (let i = em.length - 1; i >= 0; i--) {
            const e = em[i];
            if (near(S) && Math.hypot(mouse.x - e.x, mouse.y - e.y) < 320) e.vx += (mouse.x - e.x) * .00018;
            e.x += e.vx + rnd(-.4, .4); e.y += e.vy; e.vy *= .995; e.life -= .008;
            if (e.life <= 0) { em.splice(i, 1); continue; }
            ctx.fillStyle = `rgba(251,146,60,${e.life * .9})`; ctx.beginPath(); ctx.arc(e.x, e.y, e.r, 0, TAU); ctx.fill();
          }
        },
      };
    },

    /* Live scrolling candlestick chart + closing-price line + rising $ ; hover highlights candles */
    finance: S => {
      const CW = 20; let off = 0, price = 100, cs = [];
      const next = () => { const o = price, c = o + rnd(-6, 6.4), h = Math.max(o, c) + rnd(0, 4), l = Math.min(o, c) - rnd(0, 4); price = clamp(c, 60, 140); return { o, c, h, l }; };
      for (let i = 0; i < Math.ceil(S.W / CW) + 3; i++) cs.push(next());
      const coins = Array.from({ length: 8 }, () => ({ x: rnd(0, S.W), y: rnd(0, S.H), v: rnd(.3, .7), ph: rnd(0, TAU) }));
      return {
        click(x, y) { S.burst(x, y, 10); },
        draw() {
          const { ctx, W, H, mouse } = S;
          off += .6; if (off >= CW) { off -= CW; cs.push(next()); cs.shift(); }
          const top = H * .55, bot = H * .94; let mn = 1e9, mx = -1e9; cs.forEach(c => { mn = Math.min(mn, c.l); mx = Math.max(mx, c.h); });
          const Y = v => bot - (v - mn) / (mx - mn || 1) * (bot - top);
          ctx.strokeStyle = S.col(.12); ctx.lineWidth = 1;
          for (let i = 0; i < 4; i++) { const y = top + (bot - top) * i / 3; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
          cs.forEach((c, i) => {
            const x = i * CW - off, col = c.c >= c.o ? "16,185,129" : "244,63,94", a = Math.abs(mouse.x - (x + 6)) < CW * .8 ? .9 : .5;
            ctx.strokeStyle = `rgba(${col},${a})`; ctx.fillStyle = `rgba(${col},${a})`; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(x + 6, Y(c.h)); ctx.lineTo(x + 6, Y(c.l)); ctx.stroke();
            ctx.fillRect(x, Y(Math.max(c.o, c.c)), 12, Math.max(2, Math.abs(Y(c.o) - Y(c.c))));
          });
          ctx.strokeStyle = S.col(.55); ctx.lineWidth = 2; ctx.beginPath();
          cs.forEach((c, i) => i ? ctx.lineTo(i * CW - off + 6, Y(c.c)) : ctx.moveTo(i * CW - off + 6, Y(c.c))); ctx.stroke();
          ctx.font = "bold 18px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = S.col(.5);
          coins.forEach(c => { c.ph += .02; c.y -= c.v; c.x += Math.sin(c.ph) * .4; if (c.y < -20) { c.y = H + 20; c.x = rnd(0, W); } ctx.fillText("$", c.x, c.y); });
        },
      };
    },

    /* Soccer balls bouncing with gravity; cursor kicks them, click spawns / kicks */
    sports: S => {
      const mk = (x, y) => ({ x: x == null ? rnd(30, S.W - 30) : x, y: y == null ? rnd(0, S.H * .5) : y, vx: rnd(-2, 2), vy: rnd(-1, 1), r: rnd(14, 22), a: 0 });
      const bs = Array.from({ length: 6 }, () => mk());
      return {
        click(x, y) {
          if (bs.length < 12) bs.push(mk(x, y));
          bs.forEach(b => { const dx = b.x - x, dy = b.y - y, d = Math.hypot(dx, dy) || 1; if (d < 220) { b.vx += dx / d * 6; b.vy -= 7; } });
        },
        draw() {
          const { ctx, W, H, mouse } = S;
          bs.forEach(b => {
            const dx = b.x - mouse.x, dy = b.y - mouse.y, d = Math.hypot(dx, dy);
            if (d < b.r + 60 && d > 0) { b.vx += dx / d * .5; b.vy += dy / d * .5; }
            b.vy += .22; b.x += b.vx; b.y += b.vy; b.a += b.vx * .03;
            if (b.y > H - b.r) { b.y = H - b.r; b.vy *= -.82; b.vx *= .985; if (Math.abs(b.vy) < 1.5) b.vy = -rnd(9, 13); }
            if (b.x < b.r) { b.x = b.r; b.vx *= -1; } if (b.x > W - b.r) { b.x = W - b.r; b.vx *= -1; }
            ctx.fillStyle = S.col(.12 * clamp(1 - (H - b.y) / 300, .2, 1)); ctx.beginPath(); ctx.ellipse(b.x, H - 3, b.r * .9, 4, 0, 0, TAU); ctx.fill();
            ctx.save(); ctx.translate(b.x, b.y); ctx.rotate(b.a);
            ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.strokeStyle = S.col(.7); ctx.lineWidth = 1.8;
            ctx.beginPath(); ctx.arc(0, 0, b.r, 0, TAU); ctx.fill(); ctx.stroke();
            ctx.fillStyle = S.col(.75); ctx.beginPath();
            for (let i = 0; i < 5; i++) { const a = i / 5 * TAU - Math.PI / 2; ctx.lineTo(Math.cos(a) * b.r * .38, Math.sin(a) * b.r * .38); } ctx.closePath(); ctx.fill();
            ctx.lineWidth = 1.4; ctx.beginPath();
            for (let i = 0; i < 5; i++) { const a = i / 5 * TAU - Math.PI / 2; ctx.moveTo(Math.cos(a) * b.r * .38, Math.sin(a) * b.r * .38); ctx.lineTo(Math.cos(a) * b.r * .95, Math.sin(a) * b.r * .95); } ctx.stroke();
            ctx.restore();
          });
        },
      };
    },

    /* Cars driving along three lanes over a scrolling road; mouse speed / click = boost */
    car: S => {
      const hues = [RGB.car, "14,165,233", "244,63,94", "16,185,129"];
      const cars = [.66, .78, .9].flatMap(l => [0, 1].map(() => ({ x: rnd(-200, S.W), y: S.H * l, v: rnd(1.6, 3.8), w: rnd(70, 96), hue: pick(hues), b: 1 })));
      let off = 0;
      return {
        click() { cars.forEach(c => { c.b = 2.6; }); },
        draw() {
          const { ctx, W, H, mouse } = S;
          off += 3; ctx.strokeStyle = S.col(.35); ctx.lineWidth = 3; ctx.setLineDash([26, 18]); ctx.lineDashOffset = -off;
          ctx.beginPath(); ctx.moveTo(0, H * .975); ctx.lineTo(W, H * .975); ctx.stroke(); ctx.setLineDash([]);
          const m = 1 + Math.min(2, Math.abs(mouse.vx) * .02);
          cars.forEach(c => {
            c.b += (1 - c.b) * .02; c.x += c.v * c.b * m; if (c.x > W + 140) { c.x = -160; c.hue = pick(hues); }
            const h = c.w * .36, x = c.x, y = c.y;
            ctx.strokeStyle = `rgba(${c.hue},.3)`; ctx.lineWidth = 2;
            for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(x - 6, y - h * (.3 + i * .22)); ctx.lineTo(x - 30 - i * 8, y - h * (.3 + i * .22)); ctx.stroke(); }
            ctx.fillStyle = `rgba(${c.hue},.55)`; rr(ctx, x, y - h * .62, c.w, h * .62, 6); ctx.fill();
            ctx.beginPath(); ctx.moveTo(x + c.w * .2, y - h * .62); ctx.lineTo(x + c.w * .32, y - h); ctx.lineTo(x + c.w * .66, y - h); ctx.lineTo(x + c.w * .8, y - h * .62); ctx.closePath(); ctx.fill();
            ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.beginPath(); ctx.moveTo(x + c.w * .27, y - h * .66); ctx.lineTo(x + c.w * .35, y - h * .92); ctx.lineTo(x + c.w * .63, y - h * .92); ctx.lineTo(x + c.w * .73, y - h * .66); ctx.closePath(); ctx.fill();
            ctx.fillStyle = "rgba(30,41,59,.7)";
            [.22, .78].forEach(f => { ctx.beginPath(); ctx.arc(x + c.w * f, y, h * .2, 0, TAU); ctx.fill(); });
            ctx.fillStyle = "rgba(253,224,71,.8)"; ctx.fillRect(x + c.w - 4, y - h * .45, 4, 5);
          });
        },
      };
    },

    /* Planes crossing with contrails over drifting clouds; they dodge the cursor; click = new plane */
    flight: S => {
      const mkP = y => ({ x: -rnd(60, 400), y: y == null ? rnd(S.H * .08, S.H * .6) : y, v: rnd(1.6, 3.2), s: rnd(.7, 1.2) });
      const ps = [mkP(), mkP(), mkP()]; ps.forEach(p => { p.x = rnd(0, S.W); });
      const cl = Array.from({ length: 6 }, () => ({ x: rnd(0, S.W), y: rnd(0, S.H * .8), s: rnd(.6, 1.4), v: rnd(.12, .35) }));
      return {
        click(x, y) { if (ps.length < 7) ps.push(mkP(y)); },
        draw() {
          const { ctx, W, H, mouse } = S;
          ctx.fillStyle = S.col(.14);
          cl.forEach(c => {
            c.x += c.v; if (c.x > W + 120) { c.x = -120; c.y = rnd(0, H * .8); }
            [[0, 0, 22], [24, -10, 28], [52, 0, 22]].forEach(q => { ctx.beginPath(); ctx.arc(c.x + q[0] * c.s, c.y + q[1] * c.s, q[2] * c.s, 0, TAU); ctx.fill(); });
          });
          for (let i = ps.length - 1; i >= 0; i--) {
            const p = ps[i], dy = p.y - mouse.y;
            if (Math.abs(mouse.x - p.x) < 160 && Math.abs(dy) < 90) p.y += (dy >= 0 ? 1 : -1) * 1.2;
            p.x += p.v;
            if (p.x > W + 100) { if (i >= 3) { ps.splice(i, 1); continue; } p.x = -100; p.y = rnd(H * .08, H * .6); }
            const g = ctx.createLinearGradient(p.x - 240, p.y, p.x - 20, p.y); g.addColorStop(0, S.col(0)); g.addColorStop(1, S.col(.55));
            ctx.strokeStyle = g; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(p.x - 240, p.y); ctx.lineTo(p.x - 20, p.y); ctx.stroke();
            ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(-.05); ctx.scale(p.s, p.s); ctx.fillStyle = S.col(.75);
            ctx.beginPath(); ctx.ellipse(0, 0, 26, 4.5, 0, 0, TAU); ctx.fill();
            [1, -1].forEach(d => { ctx.beginPath(); ctx.moveTo(6, 0); ctx.lineTo(-8, d * 28); ctx.lineTo(-17, d * 28); ctx.lineTo(-8, 0); ctx.closePath(); ctx.fill(); ctx.beginPath(); ctx.moveTo(-18, 0); ctx.lineTo(-27, d * 10); ctx.lineTo(-30, d * 10); ctx.lineTo(-25, 0); ctx.closePath(); ctx.fill(); });
            ctx.restore();
          }
        },
      };
    },

    /* Letters, digits & math symbols drifting like chalk dust; cursor scatters them; click = shockwave */
    education: S => {
      const G = ["A", "B", "C", "π", "Σ", "∞", "+", "=", "√", "α", "β", "%", "x²", "∫", "7", "42"];
      const mk = init => ({ g: pick(G), x: rnd(0, S.W), y: init ? rnd(0, S.H) : S.H + 30, s: rnd(14, 32), v: rnd(.25, .65), ph: rnd(0, TAU), rot: rnd(-.5, .5), vr: rnd(-.01, .01), vx: 0, vy: 0, a: rnd(.4, .8) });
      const ps = Array.from({ length: 22 }, () => mk(true));
      return {
        click(x, y) { ps.forEach(p => { const dx = p.x - x, dy = p.y - y, d = Math.hypot(dx, dy) || 1; if (d < 260) { p.vx += dx / d * 6; p.vy += dy / d * 6; } }); },
        draw() {
          const { ctx, mouse } = S;
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ps.forEach((p, i) => {
            const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
            if (d < 130 && d > 0) { p.vx += dx / d * .6; p.vy += dy / d * .6; }
            p.vx *= .93; p.vy *= .93; p.ph += .012; p.x += Math.sin(p.ph) * .5 + p.vx; p.y -= p.v - p.vy; p.rot += p.vr;
            if (p.y < -40) ps[i] = mk(false);
            ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.font = `${p.s}px Georgia, serif`; ctx.fillStyle = S.col(p.a * .8); ctx.fillText(p.g, 0, 0); ctx.restore();
          });
        },
      };
    },

    /* Shopping carts rolling past + floating price tags / percent signs; click = rush */
    retail: S => {
      const carts = [0, 1].map(i => ({ x: rnd(0, S.W), y: S.H * (i ? .93 : .8), v: rnd(1.2, 2.2), s: i ? 1.1 : .85, b: 1, ph: rnd(0, TAU) }));
      const tags = Array.from({ length: 12 }, () => ({ g: pick(["%", "$", "€", "-20%", "★", "NEW"]), x: rnd(0, S.W), y: rnd(0, S.H), v: rnd(.25, .6), ph: rnd(0, TAU), s: rnd(13, 20), vx: 0, vy: 0 }));
      return {
        click(x, y) { carts.forEach(c => { c.b = 2.6; }); S.burst(x, y, 8); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          tags.forEach(g => {
            const dx = g.x - mouse.x, dy = g.y - mouse.y, d = Math.hypot(dx, dy);
            if (d < 120 && d > 0) { g.vx += dx / d * .5; g.vy += dy / d * .5; }
            g.vx *= .93; g.vy *= .93; g.ph += .015; g.x += Math.sin(g.ph) * .4 + g.vx; g.y -= g.v - g.vy;
            if (g.y < -20) { g.y = H + 20; g.x = rnd(0, W); }
            ctx.font = `bold ${g.s}px sans-serif`; ctx.fillStyle = S.col(.55); ctx.fillText(g.g, g.x, g.y);
          });
          carts.forEach(c => {
            c.b += (1 - c.b) * .02; c.x += c.v * c.b; if (c.x > W + 140) c.x = -140;
            const s = c.s, x = c.x, y = c.y + Math.sin(t * 6 + c.ph) * 1.5;
            ctx.strokeStyle = S.col(.65); ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.lineCap = "round";
            ctx.beginPath(); ctx.moveTo(x - 8 * s, y - 60 * s); ctx.lineTo(x + 6 * s, y - 60 * s); ctx.lineTo(x + 18 * s, y - 18 * s); ctx.lineTo(x + 78 * s, y - 18 * s); ctx.lineTo(x + 90 * s, y - 48 * s); ctx.lineTo(x + 12 * s, y - 48 * s); ctx.stroke();
            ctx.fillStyle = S.col(.4); ctx.fillRect(x + 26 * s, y - 66 * s, 14 * s, 18 * s); ctx.fillRect(x + 44 * s, y - 62 * s, 12 * s, 14 * s); ctx.fillRect(x + 60 * s, y - 70 * s, 14 * s, 22 * s);
            ctx.fillStyle = S.col(.7); [28, 68].forEach(f => { ctx.beginPath(); ctx.arc(x + f * s, y - 6 * s, 6 * s, 0, TAU); ctx.fill(); });
          });
        },
      };
    },

    /* Equalizer bars dancing to a beat + floating notes; mouse boosts nearby bars; click = kick */
    music: S => {
      const BW = 16, bars = Array.from({ length: Math.ceil(S.W / BW) }, () => ({ h: 0, ph: rnd(0, TAU), f: rnd(1.2, 3) }));
      const notes = []; let kick = 0, nn = 0;
      return {
        click(x) { kick = 1; for (let i = 0; i < 4; i++) notes.push({ g: pick(["♪", "♫", "♬"]), x: x + rnd(-60, 60), y: S.H * .7, v: rnd(.8, 1.6), ph: rnd(0, TAU), s: rnd(20, 34) }); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          kick *= .93; const beat = Math.pow(Math.max(0, Math.sin(t * 4.2)), 6);
          bars.forEach((b, i) => {
            let tgt = (.35 + .35 * Math.sin(t * b.f + b.ph) + .3 * Math.sin(t * b.f * 2.1 + i * .4)) * 90 + beat * 50 + kick * 120;
            if (near(S)) { const d = (i * BW - mouse.x) / 110; tgt += Math.exp(-d * d) * 110; }
            b.h += (tgt - b.h) * .18; const h = clamp(b.h, 4, H * .4);
            ctx.fillStyle = i % 2 ? S.col(.42) : "rgba(236,72,153,.38)"; rr(ctx, i * BW + 2, H - h, BW - 4, h, 4); ctx.fill();
          });
          nn -= 1 / 60; if (nn <= 0) { nn = rnd(.35, .8); notes.push({ g: pick(["♪", "♫", "♬"]), x: rnd(0, W), y: H * .72, v: rnd(.6, 1.2), ph: rnd(0, TAU), s: rnd(18, 32) }); }
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          for (let i = notes.length - 1; i >= 0; i--) {
            const n = notes[i]; n.ph += .03; n.y -= n.v; n.x += Math.sin(n.ph) * .8;
            if (n.y < -30) { notes.splice(i, 1); continue; }
            ctx.font = `${n.s}px sans-serif`; ctx.fillStyle = S.col(clamp(n.y / (H * .5), .15, .7)); ctx.fillText(n.g, n.x, n.y);
          }
        },
      };
    },

    /* Film strips scrolling at the edges, sweeping spotlights, twinkling stars; click = camera flash */
    movie: S => {
      let off = 0, flash = 0;
      const stars = Array.from({ length: 24 }, () => ({ x: rnd(0, S.W), y: rnd(0, S.H), ph: rnd(0, TAU), f: rnd(.8, 2.2), s: rnd(4, 9) }));
      return {
        click() { flash = 1; },
        draw(t) {
          const { ctx, W, H } = S;
          off = (off + 1.3) % 34; flash *= .9;
          [[W * .2, 1], [W * .8, -1]].forEach(([bx, dir]) => {
            const a = -Math.PI / 2 + Math.sin(t * .5 + dir) * .45 * dir, L = H * 1.1;
            const g = ctx.createLinearGradient(bx, H, bx + Math.cos(a) * L, H + Math.sin(a) * L); g.addColorStop(0, S.col(.24)); g.addColorStop(1, S.col(0));
            ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(bx, H);
            ctx.lineTo(bx + Math.cos(a - .12) * L, H + Math.sin(a - .12) * L); ctx.lineTo(bx + Math.cos(a + .12) * L, H + Math.sin(a + .12) * L); ctx.closePath(); ctx.fill();
          });
          [8, W - 46].forEach(sx => {
            ctx.fillStyle = S.col(.14); ctx.fillRect(sx, 0, 38, H); ctx.fillStyle = S.col(.4);
            for (let y = -34 + off; y < H; y += 34) { rr(ctx, sx + 6, y + 6, 10, 16, 3); ctx.fill(); rr(ctx, sx + 22, y + 6, 10, 16, 3); ctx.fill(); }
          });
          stars.forEach(s => { ctx.fillStyle = S.col(.15 + .6 * (.5 + .5 * Math.sin(t * s.f + s.ph))); star(ctx, s.x, s.y, s.s); });
          if (flash > .02) { ctx.fillStyle = S.col(flash * .3); ctx.fillRect(0, 0, W, H); }
        },
      };
    },

    /* Rolling waves that ripple under the cursor + rising bubbles; click = splash of bubbles */
    water: S => {
      const L = [{ base: .86, amp: 10, k: .011, sp: 1.1, ph: 0, a: .22 }, { base: .9, amp: 12, k: .008, sp: -.8, ph: 2, a: .22 }, { base: .95, amp: 9, k: .014, sp: 1.5, ph: 4, a: .26 }];
      const mkb = init => ({ x: rnd(0, S.W), y: init ? rnd(0, S.H) : S.H + 10, r: rnd(2, 8), v: rnd(.4, 1.1), ph: rnd(0, TAU) });
      const bub = Array.from({ length: 18 }, () => mkb(true));
      return {
        click(x, y) { S.burst(x, y, 8); for (let i = 0; i < 8 && bub.length < 40; i++) bub.push({ x: x + rnd(-30, 30), y: y + rnd(-10, 20), r: rnd(2, 8), v: rnd(.6, 1.4), ph: rnd(0, TAU) }); },
        draw(t) {
          const { ctx, mouse } = S;
          bub.forEach((b, i) => {
            const dx = b.x - mouse.x, dy = b.y - mouse.y, d = Math.hypot(dx, dy);
            if (d < 90 && d > 0) { b.x += dx / d * 1.5; b.y += dy / d * 1.5; }
            b.ph += .03; b.y -= b.v; b.x += Math.sin(b.ph) * .5;
            if (b.y < -10) { if (bub.length > 18) { bub.splice(i, 1); return; } Object.assign(b, mkb(false)); }
            ctx.strokeStyle = S.col(.55); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, TAU); ctx.stroke();
            ctx.fillStyle = "rgba(255,255,255,.7)"; ctx.beginPath(); ctx.arc(b.x - b.r * .3, b.y - b.r * .3, Math.max(.8, b.r * .22), 0, TAU); ctx.fill();
          });
          drawWaves(S, t, L);
        },
      };
    },

    /* Sprouts that grow and bloom + falling leaves and petals; cursor blows them away; click = plant a sprout */
    plant: S => {
      const cols = ["22,163,74", "74,222,128", "244,114,182", "192,132,252"];
      const mk = init => ({ x: rnd(0, S.W), y: init ? rnd(0, S.H) : -20, s: rnd(8, 15), vy: rnd(.4, .9), ph: rnd(0, TAU), rot: rnd(0, TAU), vr: rnd(-.03, .03), c: pick(cols), leaf: Math.random() < .6, vx: 0 });
      const ps = Array.from({ length: 20 }, () => mk(true));
      const sp = []; let nextS = 0;
      const sprout = (x, t0) => sp.push({ x, t0, h: rnd(90, 170), sw: rnd(-25, 25), c: pick(["244,114,182", "192,132,252", "251,191,36"]) });
      return {
        click(x) { if (sp.length < 8) sprout(x, S.t); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          if (t > nextS && sp.length < 5) { nextS = t + 3.5; sprout(rnd(W * .05, W * .95), t); }
          ctx.lineCap = "round";
          for (let i = sp.length - 1; i >= 0; i--) {
            const s = sp[i], age = t - s.t0; if (age > 13) { sp.splice(i, 1); continue; }
            const g = 1 - Math.pow(1 - clamp(age / 3, 0, 1), 3), bl = clamp((age - 2.6) / 1.2, 0, 1), fade = age > 11 ? 1 - (age - 11) / 2 : 1;
            const tx = s.x + s.sw * g, ty = H - s.h * g;
            ctx.globalAlpha = .6 * fade; ctx.strokeStyle = "rgba(22,163,74,.7)"; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(s.x, H); ctx.quadraticCurveTo(s.x - s.sw, H - s.h * g * .5, tx, ty); ctx.stroke();
            ctx.fillStyle = "rgba(74,222,128,.7)"; ctx.beginPath(); ctx.ellipse(s.x + 10, H - s.h * g * .45, 12 * g, 5 * g, -.5, 0, TAU); ctx.fill();
            if (bl > 0) {
              ctx.save(); ctx.translate(tx, ty); ctx.rotate(Math.sin(t) * .1); ctx.fillStyle = `rgba(${s.c},.7)`;
              for (let k = 0; k < 6; k++) { ctx.rotate(TAU / 6); ctx.beginPath(); ctx.ellipse(0, -11 * bl, 5 * bl, 11 * bl, 0, 0, TAU); ctx.fill(); }
              ctx.fillStyle = "rgba(253,224,71,.9)"; ctx.beginPath(); ctx.arc(0, 0, 4.5 * bl, 0, TAU); ctx.fill(); ctx.restore();
            }
          }
          ctx.globalAlpha = .6;
          ps.forEach((p, i) => {
            const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
            if (d < 140 && d > 0) { p.vx += dx / d * .7; p.y -= 1; }
            p.vx *= .94; p.ph += .02; p.x += Math.sin(p.ph) * .7 + p.vx; p.y += p.vy; p.rot += p.vr;
            if (p.y > H + 30) ps[i] = mk(false);
            ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.fillStyle = `rgba(${p.c},.55)`;
            ctx.beginPath(); ctx.ellipse(0, 0, p.s, p.s * .45, 0, 0, TAU); ctx.fill();
            if (p.leaf) { ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(-p.s, 0); ctx.lineTo(p.s, 0); ctx.stroke(); }
            ctx.restore();
          });
        },
      };
    },

    /* A ship sailing and bobbing on the waves, smoke from its funnels; waves ripple under the cursor */
    ship: S => {
      const L = [{ base: .86, amp: 12, k: .009, sp: 1.0, ph: 0, a: .2 }, { base: .92, amp: 10, k: .012, sp: -1.2, ph: 1.5, a: .24 }];
      const cl = Array.from({ length: 5 }, () => ({ x: rnd(0, S.W), y: rnd(S.H * .05, S.H * .5), s: rnd(.6, 1.3), v: rnd(.1, .3) }));
      const smoke = []; let sx = -120, tick = 0;
      return {
        click() { S.burst(S.mouse.x, S.H * .86, 12); },
        draw(t) {
          const { ctx, W } = S;
          ctx.fillStyle = S.col(.12);
          cl.forEach(c => { c.x += c.v; if (c.x > W + 120) c.x = -120; [[0, 0, 20], [22, -9, 26], [48, 0, 20]].forEach(q => { ctx.beginPath(); ctx.arc(c.x + q[0] * c.s, c.y + q[1] * c.s, q[2] * c.s, 0, TAU); ctx.fill(); }); });
          sx += .5; if (sx > W + 160) sx = -160;
          const y0 = waveY(S, t, L[0], sx), ang = Math.atan2(waveY(S, t, L[0], sx + 30) - y0, 30);
          if (++tick % 8 === 0) smoke.push({ x: sx, y: y0 - 40, r: 4, a: .5 });
          for (let i = smoke.length - 1; i >= 0; i--) {
            const p = smoke[i]; p.y -= .5; p.x += .4; p.r += .08; p.a -= .004; if (p.a <= 0) { smoke.splice(i, 1); continue; }
            ctx.fillStyle = S.col(p.a * .6); ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, TAU); ctx.fill();
          }
          ctx.save(); ctx.translate(sx, y0); ctx.rotate(ang);
          ctx.fillStyle = S.col(.55); ctx.beginPath(); ctx.moveTo(-60, 0); ctx.lineTo(60, 0); ctx.lineTo(46, 22); ctx.lineTo(-46, 22); ctx.closePath(); ctx.fill();
          ctx.fillStyle = S.col(.38); ctx.fillRect(-30, -18, 50, 18);
          ctx.fillStyle = S.col(.6); ctx.fillRect(-14, -34, 10, 16); ctx.fillRect(4, -34, 10, 16);
          ctx.fillStyle = "rgba(255,255,255,.7)"; for (let i = 0; i < 5; i++) { ctx.beginPath(); ctx.arc(-24 + i * 10, -9, 2.4, 0, TAU); ctx.fill(); }
          ctx.restore();
          drawWaves(S, t, L);
        },
      };
    },

    /* Radar sweep with blips + echo pings + a live oscilloscope trace; click = ping */
    sensor: S => {
      const blips = [], pings = []; let ang = 0, nb = 0;
      return {
        click(x, y) { pings.push({ x, y, r: 6, a: .8 }); },
        draw(t) {
          const { ctx, W, H, mouse } = S;
          const R = Math.min(150, W * .18), cx = W - R - 40, cy = R + 80;
          ctx.lineWidth = 1.2; ctx.strokeStyle = S.col(.3);
          for (let i = 1; i <= 4; i++) { ctx.beginPath(); ctx.arc(cx, cy, R * i / 4, 0, TAU); ctx.stroke(); }
          ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
          ang += .03;
          for (let i = 0; i < 28; i++) { const a0 = ang - i * .045; ctx.fillStyle = S.col(.3 * (1 - i / 28)); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, a0 - .045, a0); ctx.closePath(); ctx.fill(); }
          ctx.strokeStyle = S.col(.85); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(ang) * R, cy + Math.sin(ang) * R); ctx.stroke();
          nb -= 1 / 60; if (nb <= 0) { nb = rnd(.4, .9); const a = rnd(0, TAU), r = rnd(.2, .95) * R; blips.push({ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r, life: 1 }); }
          for (let i = blips.length - 1; i >= 0; i--) { const b = blips[i]; b.life -= .008; if (b.life <= 0) { blips.splice(i, 1); continue; } ctx.fillStyle = S.col(b.life * .9); ctx.beginPath(); ctx.arc(b.x, b.y, 3 + (1 - b.life) * 2, 0, TAU); ctx.fill(); }
          const ph = (t % 2.4) / 2.4; ctx.strokeStyle = S.col(.3 * (1 - ph)); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, R * (1 + ph * .5), 0, TAU); ctx.stroke();
          for (let i = pings.length - 1; i >= 0; i--) { const q = pings[i]; q.r += 3; q.a -= .014; if (q.a <= 0) { pings.splice(i, 1); continue; } ctx.strokeStyle = S.col(q.a); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(q.x, q.y, q.r, 0, TAU); ctx.stroke(); }
          ctx.strokeStyle = S.col(.55); ctx.lineWidth = 2; ctx.beginPath();
          for (let x = 0; x <= W; x += 6) {
            let y = H * .9 + Math.sin(x * .03 + t * 3) * 10 + Math.sin(x * .011 - t * 1.7) * 14 + rnd(-2, 2);
            if (near(S)) { const d = (x - mouse.x) / 60; y -= Math.exp(-d * d) * 30 * Math.sin(t * 20); }
            x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          ctx.stroke();
        },
      };
    },

    /* City skyline whose windows light up and flicker; hovering a building lights it; click = lights on */
    house: S => {
      const bl = []; let x = -10;
      while (x < S.W) {
        const w = rnd(46, 92), h = rnd(S.H * .12, S.H * .38), cols = Math.max(1, Math.floor((w - 10) / 14)), rows = Math.max(1, Math.floor((h - 10) / 18));
        bl.push({ x, w, h, cols, rows, win: Array.from({ length: cols * rows }, () => Math.random() < .35) }); x += w + rnd(4, 14);
      }
      const cl = Array.from({ length: 5 }, () => ({ x: rnd(0, S.W), y: rnd(S.H * .05, S.H * .45), s: rnd(.6, 1.3), v: rnd(.1, .3) }));
      return {
        click(x) { bl.forEach(b => { if (x >= b.x && x <= b.x + b.w) b.win = b.win.map(() => Math.random() < .8); }); },
        draw() {
          const { ctx, W, H, mouse } = S;
          ctx.fillStyle = S.col(.1);
          cl.forEach(c => { c.x += c.v; if (c.x > W + 120) c.x = -120; [[0, 0, 20], [22, -9, 26], [48, 0, 20]].forEach(q => { ctx.beginPath(); ctx.arc(c.x + q[0] * c.s, c.y + q[1] * c.s, q[2] * c.s, 0, TAU); ctx.fill(); }); });
          bl.forEach(b => {
            const hov = mouse.x >= b.x && mouse.x <= b.x + b.w && mouse.y > H - b.h - 40;
            if (hov && Math.random() < .08) b.win[(Math.random() * b.win.length) | 0] = true;
            ctx.fillStyle = S.col(.18); ctx.fillRect(b.x, H - b.h, b.w, b.h);
            for (let r = 0; r < b.rows; r++) for (let c = 0; c < b.cols; c++) {
              const i = r * b.cols + c; if (Math.random() < .002) b.win[i] = !b.win[i];
              ctx.fillStyle = b.win[i] ? "rgba(250,204,21,.8)" : S.col(.12); ctx.fillRect(b.x + 8 + c * 14, H - b.h + 8 + r * 18, 8, 10);
            }
          });
        },
      };
    },
  };

  window.MLS_SCENES = SCENES;
  window.MLS_SCENE_RGB = RGB;
})();
