"""Themed cover illustrations for datasets (pure SVG, 400x160, light colours).

theme_for(name)  -> theme key from keywords in the dataset name, or None
scene(theme, name) -> SVG string
"""
import re

W, H = 400, 160

# (theme, regex) — first match wins
THEMES = (
    ("weather", r"weather|meteo|climate|temperature|rain|forecast"),
    ("virus",   r"covid|corona|virus|pandemic|epidemi"),
    ("health",  r"diabet|health|patient|medical|cancer|heart|disease|hospital|clinic|cardio|blood"),
    ("fire",    r"fire|wildfire|burn|forest"),
    ("energy",  r"electric|energy|power|battery|grid|consumption|load"),
    ("solar",   r"solar|photovolt|\bpv\b"),
    ("wind",    r"wind|turbine"),
    ("sports",  r"player|football|soccer|\bfifa|sport|match|league|\bnba|basket|team"),
    ("finance", r"stock|finance|bank|credit|loan|bitcoin|crypto|market|price_?index|trading|invest"),
    ("house",   r"house|housing|real_?estate|rent|apartment|property"),
    ("car",     r"(?<![a-z])cars?(?![a-z])|vehicle|automobile|traffic|taxi|uber|mpg"),
    ("flight",  r"flight|airline|airport|aviation|aircraft|plane"),
    ("education", r"student|school|education|exam|course|university|grade"),
    ("retail",  r"sale|retail|shop|store|customer|churn|order|product|ecommerce|e-commerce"),
    ("music",   r"music|song|spotify|audio|track"),
    ("movie",   r"movie|film|imdb|netflix|cinema"),
    ("water",   r"water|ocean|sea|river|lake|flood|rainfall"),
    ("plant",   r"iris|flower|plant|crop|agri|farm|wine|leaf|seed"),
    ("ship",    r"titanic|ship|boat|vessel|maritime"),
    ("sensor",  r"sensor|iot|signal|device|telemetry|machine|vibration"),
)


def theme_for(name):
    low = str(name).lower()
    for key, rx in THEMES:
        if re.search(rx, low):
            return key
    return None


def _svg(bg_a, bg_b, body, uid="g"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">'
            f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{bg_a}"/>'
            f'<stop offset="1" stop-color="{bg_b}"/></linearGradient></defs>'
            f'<rect width="{W}" height="{H}" fill="url(#{uid})"/>{body}</svg>')


def _cloud(x, y, s=1.0, fill="#fff", op=0.95):
    return (f'<g transform="translate({x} {y}) scale({s})" fill="{fill}" opacity="{op}">'
            '<circle cx="0" cy="0" r="16"/><circle cx="22" cy="-10" r="22"/><circle cx="48" cy="0" r="17"/>'
            '<rect x="-16" y="0" width="80" height="17" rx="8"/></g>')


def _ray_sun(cx, cy, r, core="#fbbf24", ray="#fcd34d", n=12):
    import math
    rays = "".join(
        f'<line x1="{cx + math.cos(i * 2 * math.pi / n) * (r + 8):.1f}" y1="{cy + math.sin(i * 2 * math.pi / n) * (r + 8):.1f}" '
        f'x2="{cx + math.cos(i * 2 * math.pi / n) * (r + 20):.1f}" y2="{cy + math.sin(i * 2 * math.pi / n) * (r + 20):.1f}" '
        f'stroke="{ray}" stroke-width="5" stroke-linecap="round"/>' for i in range(n))
    return f'<circle cx="{cx}" cy="{cy}" r="{r + 10}" fill="{ray}" opacity=".35"/>{rays}<circle cx="{cx}" cy="{cy}" r="{r}" fill="{core}"/>'


def _weather(name):
    drops = "".join(f'<line x1="{x}" y1="{y}" x2="{x - 5}" y2="{y + 13}" stroke="#60a5fa" stroke-width="3" stroke-linecap="round" opacity=".8"/>'
                    for x, y in ((258, 104), (276, 112), (294, 104), (312, 114), (270, 128), (302, 130)))
    return _svg("#bfe3ff", "#eaf6ff",
                _ray_sun(88, 56, 26) + _cloud(60, 90, 1.25, "#fff") + _cloud(240, 82, 1.35, "#e8eef8") + drops +
                _cloud(330, 36, .8, "#fff", .8) +
                '<path d="M0 150 Q100 132 200 146 T400 140 V160 H0Z" fill="#9ed0f5" opacity=".55"/>')


def _health(name):
    ecg = '<polyline points="0,100 90,100 112,100 124,72 138,134 152,86 164,100 400,100" fill="none" stroke="#0d9488" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" opacity=".75"/>'
    cross = '<g transform="translate(22 0)" fill="#14b8a6" opacity=".9"><rect x="40" y="34" width="26" height="92" rx="9"/><rect x="7" y="67" width="92" height="26" rx="9"/></g>'
    drop = lambda x, y, s: f'<path transform="translate({x} {y}) scale({s})" d="M0 -22 C 14 -4 18 6 18 12 A18 18 0 0 1 -18 12 C -18 6 -14 -4 0 -22Z" fill="#fb7185" opacity=".85"/>'
    pills = '<g transform="translate(300 40) rotate(-30)"><rect width="56" height="22" rx="11" fill="#fff" stroke="#99f6e4" stroke-width="2"/><rect width="28" height="22" rx="11" fill="#5eead4"/></g>'
    return _svg("#d9f7f0", "#f0fdfa", ecg + cross + drop(250, 54, 1) + drop(345, 120, .8) + pills +
                '<circle cx="372" cy="22" r="40" fill="#99f6e4" opacity=".4"/>')


def _virus(name):
    import math
    cx, cy, r = 285, 80, 38
    spikes = "".join(
        f'<line x1="{cx + math.cos(a) * r:.1f}" y1="{cy + math.sin(a) * r:.1f}" x2="{cx + math.cos(a) * (r + 18):.1f}" y2="{cy + math.sin(a) * (r + 18):.1f}" stroke="#60a5fa" stroke-width="4"/>'
        f'<circle cx="{cx + math.cos(a) * (r + 21):.1f}" cy="{cy + math.sin(a) * (r + 21):.1f}" r="5" fill="#0ea5e9"/>'
        for a in [i * math.pi / 6 for i in range(12)])
    small = '<circle cx="90" cy="50" r="14" fill="#93c5fd" opacity=".7"/><circle cx="130" cy="110" r="9" fill="#bfdbfe"/><circle cx="52" cy="112" r="11" fill="#60a5fa" opacity=".5"/>'
    return _svg("#dbeafe", "#eff6ff", small + spikes + f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#93c5fd"/><circle cx="272" cy="70" r="8" fill="#dbeafe" opacity=".8"/><circle cx="298" cy="92" r="5" fill="#dbeafe" opacity=".8"/>')


def _energy(name):
    bolt = '<polygon points="215,14 160,92 198,92 178,146 252,62 212,62 236,14" fill="#f59e0b"/><polygon points="215,14 160,92 198,92 178,146 252,62 212,62 236,14" fill="none" stroke="#fde68a" stroke-width="4" stroke-linejoin="round"/>'
    glow = '<circle cx="205" cy="80" r="70" fill="#fde68a" opacity=".5"/><circle cx="205" cy="80" r="46" fill="#fcd34d" opacity=".45"/>'
    batt = '<g transform="translate(40 52)"><rect width="64" height="38" rx="7" fill="#fff" stroke="#f59e0b" stroke-width="3"/><rect x="64" y="12" width="7" height="14" rx="2" fill="#f59e0b"/><rect x="6" y="6" width="30" height="26" rx="4" fill="#fbbf24"/></g>'
    lines = '<g stroke="#f59e0b" stroke-width="3" fill="none" opacity=".55" stroke-linecap="round"><path d="M290 40 H340 V70 H372"/><path d="M290 80 H325 V112 H372"/><path d="M300 130 H350"/></g><g fill="#f59e0b" opacity=".8"><circle cx="372" cy="70" r="5"/><circle cx="372" cy="112" r="5"/><circle cx="350" cy="130" r="5"/></g>'
    return _svg("#fff3c4", "#fffbeb", glow + bolt + batt + lines)


def _tree(x, y, s, c1="#16a34a", c2="#15803d"):
    return (f'<g transform="translate({x} {y}) scale({s})"><rect x="-3" y="0" width="6" height="14" fill="#92400e" opacity=".7"/>'
            f'<polygon points="0,-34 16,-6 -16,-6" fill="{c2}"/><polygon points="0,-22 20,8 -20,8" fill="{c1}"/></g>')


def _flame(x, y, s, c1="#f97316", c2="#fbbf24"):
    return (f'<g transform="translate({x} {y}) scale({s})"><path d="M0 -60 C 8 -40 30 -34 30 -8 C 30 12 16 20 0 20 C -16 20 -30 12 -30 -8 C -30 -22 -18 -30 -12 -46 C -6 -38 -2 -48 0 -60Z" fill="{c1}"/>'
            f'<path d="M0 -26 C 5 -14 16 -10 16 4 C 16 14 8 20 0 20 C -8 20 -16 14 -16 4 C -16 -8 -6 -12 0 -26Z" fill="{c2}"/></g>')


def _fire(name):
    return _svg("#ffe4cc", "#fff4e6",
                '<circle cx="320" cy="40" r="26" fill="#fb923c" opacity=".35"/>' +
                _tree(34, 118, 1.1) + _tree(70, 124, .9) + _tree(340, 120, 1.1) + _tree(372, 126, .85) +
                _flame(140, 128, 1.25, "#ef4444", "#fb923c") + _flame(200, 132, 1.7) + _flame(262, 128, 1.2, "#ef4444", "#fb923c") +
                '<path d="M0 148 Q100 140 200 148 T400 146 V160 H0Z" fill="#7c2d12" opacity=".25"/>')


def _sports(name):
    import math
    pent = '<polygon points="320,66 334,76 329,92 311,92 306,76" fill="#1e293b"/>'
    seams = '<g stroke="#1e293b" stroke-width="2.5" opacity=".8"><line x1="320" y1="66" x2="320" y2="46"/><line x1="334" y1="76" x2="352" y2="70"/><line x1="329" y1="92" x2="340" y2="108"/><line x1="311" y1="92" x2="300" y2="108"/><line x1="306" y1="76" x2="288" y2="70"/></g>'
    field = ('<g fill="none" stroke="#fff" stroke-width="3" opacity=".8"><rect x="16" y="16" width="250" height="128" rx="6"/><line x1="141" y1="16" x2="141" y2="144"/>'
             '<circle cx="141" cy="80" r="26"/><rect x="16" y="52" width="36" height="56"/><rect x="230" y="52" width="36" height="56"/></g>')
    return _svg("#86efac", "#bbf7d0",
                '<g opacity=".25" fill="#16a34a">' + "".join(f'<rect x="{i * 50}" y="0" width="25" height="160"/>' for i in range(8)) + '</g>' + field +
                '<circle cx="320" cy="80" r="40" fill="#fff"/>' + pent + seams + '<circle cx="320" cy="80" r="40" fill="none" stroke="#cbd5e1" stroke-width="2"/>')


def _finance(name):
    bars = ""
    data = [(40, 70, 40, 1), (70, 56, 48, 0), (100, 62, 30, 1), (130, 44, 46, 1), (160, 50, 30, 0), (190, 34, 44, 1), (220, 26, 38, 1), (250, 40, 28, 0)]
    for x, y, h, up in data:
        c = "#10b981" if up else "#f43f5e"
        bars += f'<line x1="{x + 8}" y1="{y - 12}" x2="{x + 8}" y2="{y + h + 12}" stroke="{c}" stroke-width="2.5"/><rect x="{x}" y="{y}" width="16" height="{h}" rx="3" fill="{c}" opacity=".85"/>'
    line = '<polyline points="30,120 80,100 125,108 170,76 215,60 262,36" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity=".6"/>'
    coin = '<g transform="translate(330 78)"><circle r="38" fill="#fbbf24"/><circle r="30" fill="none" stroke="#f59e0b" stroke-width="3"/><text y="12" text-anchor="middle" font-family="Georgia,serif" font-weight="bold" font-size="34" fill="#b45309">$</text></g>'
    return _svg("#dbeafe", "#f0f9ff", '<g stroke="#bfdbfe" stroke-width="1.5">' + "".join(f'<line x1="0" y1="{y}" x2="400" y2="{y}"/>' for y in (30, 70, 110, 150)) + '</g>' + bars + line + coin)


def _house(name):
    def home(x, w, h, roof, wall):
        return (f'<rect x="{x}" y="{150 - h}" width="{w}" height="{h}" fill="{wall}"/><polygon points="{x - 8},{150 - h} {x + w / 2},{150 - h - 34} {x + w + 8},{150 - h}" fill="{roof}"/>'
                f'<rect x="{x + w / 2 - 9}" y="{150 - 34}" width="18" height="34" rx="3" fill="#fff" opacity=".85"/><rect x="{x + 10}" y="{150 - h + 14}" width="18" height="16" rx="2" fill="#bfdbfe"/>'
                + (f'<rect x="{x + w - 28}" y="{150 - h + 14}" width="18" height="16" rx="2" fill="#bfdbfe"/>' if w > 70 else ''))
    return _svg("#c7ecff", "#effaff", _ray_sun(340, 40, 18) + _cloud(110, 30, .8, "#fff", .9) +
                home(40, 80, 62, "#f87171", "#fde68a") + home(160, 96, 78, "#60a5fa", "#fecaca") + home(300, 70, 56, "#34d399", "#fef3c7") +
                '<rect x="0" y="148" width="400" height="12" fill="#86efac"/>')


def _car(name):
    car = ('<g transform="translate(110 70)"><path d="M0 52 L8 26 Q14 8 36 8 H108 Q128 8 140 28 L170 36 Q182 40 182 52 V62 H0Z" fill="#2563eb"/>'
           '<path d="M24 28 Q28 16 42 16 H66 V36 H18Z" fill="#dbeafe"/><path d="M74 16 H106 Q118 16 126 36 H74Z" fill="#dbeafe"/>'
           '<circle cx="42" cy="62" r="17" fill="#1e293b"/><circle cx="42" cy="62" r="7" fill="#cbd5e1"/><circle cx="142" cy="62" r="17" fill="#1e293b"/><circle cx="142" cy="62" r="7" fill="#cbd5e1"/>'
           '<rect x="172" y="40" width="10" height="8" rx="3" fill="#fde047"/></g>')
    road = '<rect y="136" width="400" height="24" fill="#94a3b8" opacity=".6"/><g stroke="#fff" stroke-width="4" stroke-dasharray="26 18" opacity=".9"><line x1="0" y1="148" x2="400" y2="148"/></g>'
    lines = '<g stroke="#bfdbfe" stroke-width="4" stroke-linecap="round"><line x1="20" y1="90" x2="80" y2="90"/><line x1="40" y1="108" x2="96" y2="108"/><line x1="10" y1="72" x2="52" y2="72"/></g>'
    return _svg("#dbeafe", "#f8fbff", _cloud(310, 36, .9, "#fff", .9) + lines + car + road)


def _flight(name):
    plane = ('<g transform="translate(250 62) rotate(-18)"><path d="M0 0 L80 -10 Q96 -10 96 0 Q96 10 80 10 Z" fill="#fff" stroke="#93c5fd" stroke-width="2"/>'
             '<path d="M30 -2 L10 -44 L26 -44 L58 -4Z" fill="#60a5fa"/><path d="M30 2 L10 44 L26 44 L58 4Z" fill="#3b82f6"/><path d="M4 -2 L-12 -18 L-2 -18 L12 -2Z" fill="#60a5fa"/>'
             '<circle cx="82" cy="-2" r="2.5" fill="#93c5fd"/><circle cx="72" cy="-2" r="2.5" fill="#93c5fd"/><circle cx="62" cy="-2" r="2.5" fill="#93c5fd"/></g>')
    trail = '<path d="M20 130 C 90 126 140 100 240 78" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round" stroke-dasharray="2 12" opacity=".9"/>'
    return _svg("#7dd3fc", "#e0f2fe", _cloud(60, 52, 1.0, "#fff") + _cloud(330, 128, 1.1, "#fff", .9) + _cloud(170, 134, .7, "#fff", .8) + trail + plane)


def _education(name):
    books = ('<g transform="translate(50 84)"><rect width="120" height="22" rx="4" fill="#2563eb"/><rect x="8" y="22" width="108" height="20" rx="4" fill="#f59e0b"/><rect x="0" y="42" width="124" height="22" rx="4" fill="#10b981"/>'
             '<rect x="14" y="8" width="50" height="4" rx="2" fill="#fff" opacity=".6"/><rect x="20" y="28" width="40" height="4" rx="2" fill="#fff" opacity=".6"/><rect x="12" y="49" width="60" height="4" rx="2" fill="#fff" opacity=".6"/></g>')
    cap = ('<g transform="translate(260 50)"><polygon points="0,26 70,0 140,26 70,52" fill="#1e293b"/><path d="M30 38 V64 Q70 84 110 64 V38 L70 52Z" fill="#334155"/>'
           '<line x1="132" y1="30" x2="132" y2="64" stroke="#fbbf24" stroke-width="3"/><circle cx="132" cy="68" r="5" fill="#fbbf24"/></g>')
    return _svg("#fde7f3", "#fff7fb", '<g fill="#f9a8d4" opacity=".5"><circle cx="30" cy="30" r="8"/><circle cx="370" cy="132" r="12"/><circle cx="210" cy="26" r="6"/></g>' + books + cap)


def _retail(name):
    cart = ('<g transform="translate(110 38)"><path d="M0 0 H26 L46 66 H128 L148 18 H34" fill="none" stroke="#2563eb" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>'
            '<rect x="56" y="22" width="22" height="26" rx="4" fill="#f59e0b"/><rect x="84" y="14" width="22" height="34" rx="4" fill="#10b981"/><rect x="112" y="26" width="20" height="22" rx="4" fill="#f43f5e"/>'
            '<circle cx="58" cy="86" r="10" fill="#1e293b"/><circle cx="122" cy="86" r="10" fill="#1e293b"/></g>')
    bag = '<g transform="translate(300 60)"><path d="M0 22 H64 L58 84 H6Z" fill="#fb7185"/><path d="M18 22 V14 A14 14 0 0 1 46 14 V22" fill="none" stroke="#be123c" stroke-width="5"/></g>'
    tag = '<g transform="translate(40 96) rotate(-18)"><path d="M0 0 H44 L60 18 L44 36 H0Z" fill="#fde68a"/><circle cx="46" cy="18" r="4" fill="#fff"/></g>'
    return _svg("#e0f2fe", "#f5faff", cart + bag + tag)


def _music(name):
    wave = "".join(f'<rect x="{20 + i * 12}" y="{80 - h / 2}" width="7" height="{h}" rx="3.5" fill="{"#60a5fa" if i % 2 else "#f472b6"}" opacity=".75"/>'
                   for i, h in enumerate([20, 44, 30, 72, 50, 90, 36, 64, 26, 80, 46, 60, 22, 70, 40, 84, 30, 54, 24, 46, 28, 38]))
    note = ('<g transform="translate(300 32)" fill="#1d4ed8"><ellipse cx="14" cy="76" rx="16" ry="11"/><ellipse cx="66" cy="66" rx="16" ry="11"/>'
            '<rect x="26" y="10" width="6" height="66"/><rect x="78" y="0" width="6" height="66"/><path d="M26 10 L84 0 V16 L26 26Z"/></g>')
    return _svg("#f3e8ff", "#fdf4ff", wave + note)


def _movie(name):
    clap = ('<g transform="translate(145 40)"><rect y="26" width="110" height="64" rx="6" fill="#334155"/><path d="M0 8 L100 -6 L106 18 L6 32Z" fill="#fff"/>'
            '<g fill="#1e293b"><polygon points="12,28 28,10 44,8 28,30"/><polygon points="50,24 66,6 82,4 66,26"/></g>'
            '<circle cx="55" cy="58" r="14" fill="#fbbf24"/><polygon points="50,50 50,66 64,58" fill="#334155"/></g>')
    strip = lambda x: f'<g transform="translate({x} 0)"><rect width="34" height="160" fill="#475569" opacity=".85"/>' + "".join(f'<rect x="8" y="{y}" width="18" height="12" rx="3" fill="#e2e8f0"/>' for y in range(8, 160, 24)) + '</g>'
    return _svg("#fee2e2", "#fff5f5", strip(12) + strip(354) + clap)


def _water(name):
    waves = "".join(f'<path d="M0 {y} Q50 {y - 14} 100 {y} T200 {y} T300 {y} T400 {y} V160 H0Z" fill="{c}" opacity="{o}"/>'
                    for y, c, o in ((92, "#38bdf8", .45), (112, "#0ea5e9", .45), (132, "#0284c7", .4)))
    drop = lambda x, y, s: f'<path transform="translate({x} {y}) scale({s})" d="M0 -28 C 18 -4 22 8 22 16 A22 22 0 0 1 -22 16 C -22 8 -18 -4 0 -28Z" fill="#7dd3fc"/>'
    return _svg("#e0f7ff", "#f0fbff", _ray_sun(330, 36, 16, "#fde047", "#fef08a", 10) + drop(90, 52, 1.2) + drop(160, 36, .8) + drop(130, 70, .6) + waves)


def _plant(name):
    petals = "".join(f'<ellipse cx="0" cy="-26" rx="12" ry="24" fill="{c}" transform="rotate({a})" opacity=".92"/>'
                     for a, c in ((0, "#f472b6"), (72, "#c084fc"), (144, "#f472b6"), (216, "#c084fc"), (288, "#f472b6")))
    flower = lambda x, y, s: f'<g transform="translate({x} {y}) scale({s})">{petals}<circle r="11" fill="#fde047"/></g>'
    stem = lambda x, y, h: f'<path d="M{x} {y} Q{x - 10} {y + h / 2} {x} {y + h}" fill="none" stroke="#16a34a" stroke-width="5" stroke-linecap="round"/><ellipse cx="{x + 14}" cy="{y + h * .6}" rx="16" ry="7" transform="rotate(-28 {x + 14} {y + h * .6})" fill="#4ade80"/>'
    return _svg("#dcfce7", "#f7fff9", stem(90, 70, 90) + stem(210, 56, 104) + stem(320, 76, 84) + flower(90, 60, 1.0) + flower(210, 44, 1.25) + flower(320, 66, .9) +
                '<rect y="146" width="400" height="14" fill="#86efac" opacity=".7"/>')


def _ship(name):
    ship = ('<g transform="translate(120 44)"><path d="M0 60 H170 L146 92 H24Z" fill="#334155"/><rect x="20" y="30" width="130" height="30" fill="#f8fafc"/>'
            '<rect x="50" y="10" width="70" height="20" fill="#e2e8f0"/><rect x="72" y="-12" width="14" height="22" fill="#f43f5e"/><rect x="96" y="-12" width="14" height="22" fill="#f43f5e"/>'
            + "".join(f'<circle cx="{x}" cy="46" r="5" fill="#60a5fa"/>' for x in range(36, 140, 20)) + '</g>')
    waves = "".join(f'<path d="M0 {y} Q50 {y - 12} 100 {y} T200 {y} T300 {y} T400 {y} V160 H0Z" fill="{c}" opacity="{o}"/>' for y, c, o in ((124, "#38bdf8", .55), (142, "#0284c7", .5)))
    return _svg("#dbeafe", "#eff6ff", _cloud(60, 34, .9, "#fff") + _cloud(330, 52, .8, "#fff", .9) + ship + waves)


def _sensor(name):
    import math
    arcs = "".join(f'<path d="M{200 - r * math.sin(0.9):.1f} {58 - r * math.cos(0.9) + r:.1f} A{r} {r} 0 0 1 {200 + r * math.sin(0.9):.1f} {58 - r * math.cos(0.9) + r:.1f}" fill="none" stroke="#06b6d4" stroke-width="7" stroke-linecap="round" opacity="{o}"/>'
                   for r, o in ((26, .95), (48, .7), (70, .45)))
    chip = ('<g transform="translate(160 92)"><rect width="80" height="52" rx="8" fill="#0e7490"/><rect x="18" y="14" width="44" height="24" rx="4" fill="#67e8f9"/>'
            + "".join(f'<rect x="{x}" y="-8" width="5" height="10" fill="#0e7490"/><rect x="{x}" y="50" width="5" height="10" fill="#0e7490"/>' for x in (14, 30, 46, 62)) + '</g>')
    pulse = '<polyline points="0,120 40,120 56,96 76,142 92,110 108,120 150,120" fill="none" stroke="#22d3ee" stroke-width="3.5" stroke-linecap="round" opacity=".7"/><polyline points="250,120 292,120 308,98 326,140 340,112 356,120 400,120" fill="none" stroke="#22d3ee" stroke-width="3.5" stroke-linecap="round" opacity=".7"/>'
    return _svg("#cffafe", "#ecfeff", arcs + '<circle cx="200" cy="64" r="7" fill="#06b6d4"/>' + chip + pulse)


def _solar(name):
    panel = ('<g transform="translate(180 62) skewX(-18)"><rect width="170" height="84" rx="5" fill="#1d4ed8"/>'
             + "".join(f'<line x1="{x}" y1="0" x2="{x}" y2="84" stroke="#93c5fd" stroke-width="2"/>' for x in (42, 85, 128))
             + "".join(f'<line x1="0" y1="{y}" x2="170" y2="{y}" stroke="#93c5fd" stroke-width="2"/>' for y in (28, 56)) + '</g>')
    return _svg("#fff4c2", "#fffbe8", _ray_sun(90, 62, 30) + panel + '<rect x="256" y="146" width="10" height="14" fill="#64748b"/><rect y="150" width="400" height="10" fill="#bbf7d0"/>')


def _wind(name):
    def turb(x, y, s, rot):
        blades = "".join(f'<path d="M0 0 L-4 -46 L4 -46Z" fill="#fff" stroke="#bae6fd" stroke-width="1.5" transform="rotate({a})"/>' for a in (0, 120, 240))
        return (f'<g transform="translate({x} {y}) scale({s})"><rect x="-3" y="0" width="6" height="{150 - y}" fill="#e2e8f0"/><g transform="rotate({rot})">{blades}</g><circle r="5" fill="#94a3b8"/></g>')
    gust = '<g fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".85"><path d="M20 40 H90 a10 10 0 1 0 -10 -10"/><path d="M280 100 H360 a10 10 0 1 1 -10 10"/></g>'
    return _svg("#bae6fd", "#e8f7ff", gust + turb(100, 62, 1.5, 10) + turb(220, 80, 1.2, 50) + turb(320, 70, 1.35, 30) + '<path d="M0 146 Q120 128 240 142 T400 136 V160 H0Z" fill="#86efac" opacity=".8"/>')


def _data(name):
    bars = "".join(f'<rect x="{40 + i * 34}" y="{140 - h}" width="22" height="{h}" rx="4" fill="{c}" opacity=".8"/>'
                   for i, (h, c) in enumerate(((40, "#93c5fd"), (70, "#60a5fa"), (52, "#93c5fd"), (96, "#2563eb"), (64, "#60a5fa"), (84, "#93c5fd"), (110, "#2563eb"))))
    line = '<polyline points="40,96 74,70 108,82 142,44 176,60 210,30 244,18" fill="none" stroke="#06b6d4" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity=".75"/>'
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="5" fill="#06b6d4"/>' for x, y in ((40, 96), (74, 70), (108, 82), (142, 44), (176, 60), (210, 30), (244, 18)))
    grid = '<g stroke="#bfdbfe" stroke-width="1.5" opacity=".7">' + "".join(f'<line x1="0" y1="{y}" x2="400" y2="{y}"/>' for y in (30, 70, 110)) + '</g>'
    blobs = '<circle cx="330" cy="50" r="34" fill="#bfdbfe" opacity=".5"/><circle cx="360" cy="110" r="22" fill="#a5f3fc" opacity=".6"/><circle cx="300" cy="120" r="12" fill="#bfdbfe" opacity=".8"/>'
    return _svg("#dbeafe", "#f8fbff", grid + blobs + bars + line + dots)


_SCENES = {
    "data": _data,
    "weather": _weather, "health": _health, "virus": _virus, "energy": _energy, "fire": _fire, "sports": _sports,
    "finance": _finance, "house": _house, "car": _car, "flight": _flight, "education": _education, "retail": _retail,
    "music": _music, "movie": _movie, "water": _water, "plant": _plant, "ship": _ship, "sensor": _sensor,
    "solar": _solar, "wind": _wind,
}


def scene(theme, name=""):
    return _SCENES.get(theme or "data", _data)(name)


# Sky colours (top, bottom) of each scene — used for the full-page background of a dataset's theme.
PALETTE = {
    "data": ("#dbeafe", "#f8fbff"),
    "weather": ("#bfe3ff", "#eaf6ff"), "health": ("#d9f7f0", "#f0fdfa"), "virus": ("#dbeafe", "#eff6ff"),
    "energy": ("#fff3c4", "#fffbeb"), "fire": ("#ffe4cc", "#fff4e6"), "sports": ("#86efac", "#bbf7d0"),
    "finance": ("#dbeafe", "#f0f9ff"), "house": ("#c7ecff", "#effaff"), "car": ("#dbeafe", "#f8fbff"),
    "flight": ("#7dd3fc", "#e0f2fe"), "education": ("#fde7f3", "#fff7fb"), "retail": ("#e0f2fe", "#f5faff"),
    "music": ("#f3e8ff", "#fdf4ff"), "movie": ("#fee2e2", "#fff5f5"), "water": ("#e0f7ff", "#f0fbff"),
    "plant": ("#dcfce7", "#f7fff9"), "ship": ("#dbeafe", "#eff6ff"), "sensor": ("#cffafe", "#ecfeff"),
    "solar": ("#fff4c2", "#fffbe8"), "wind": ("#bae6fd", "#e8f7ff"),
}


def palette(theme):
    return PALETTE.get(theme or "data")
