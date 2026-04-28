

## Should we swap the `Wind` glyph for `/icon-512.png`?

**My honest take: keep the existing `Wind` glyph as the inline UI mark, and use `/icon-512.png` only where a real raster logo belongs.** Here's why:

**Why the `Wind` Lucide icon wins for inline UI:**
- It's an SVG — crisp at every size, no blur on retina, no extra HTTP request.
- It inherits `text-clean` color and the `glow-text-clean` shadow, which is core to the "command center" aesthetic. A PNG can't glow with the theme.
- It scales cleanly inside the small 20–36px tiles in the sidebar, mobile bar, login card, and regime badge.
- Swapping to a 10KB PNG at 20px would actually look *worse* (soft edges, no theme-tinted glow).

**Where the PNG asset genuinely helps:**
- **Browser tab favicon** — currently missing a `<link rel="icon">`, so the tab is blank. This is the one place a raster logo is mandatory.
- **Login hero** — a single, larger 56–64px branded mark above the sign-in title would feel more "product" and less "icon font". Worth doing.

**Where I'd leave it alone:**
- Sidebar header, mobile top bar, RegimeCard badge — all small, theme-tinted, and look correct as-is.

## Plan

1. **`src/routes/__root.tsx`** — add `<link rel="icon" type="image/png" href="/icon-512.png" />` so the browser tab shows the brand.
2. **`src/routes/login.tsx`** — replace the small `Wind`-in-tile lockup with a single 56px `<img src="/icon-512.png">` centered above "EnviroSense AI", giving the auth screen real brand presence.
3. **Leave Sidebar, TopHeader, RegimeCard untouched** — the `Wind` glyph is the right call for inline chrome.

### Bonus quiet fix
While in `TopHeader.tsx`, fix the SSR hydration mismatch the runtime is reporting: `navigator.onLine` is read during SSR (where it's `true`) but can differ on the client. I'll initialize `online` to `true` and only update it inside `useEffect`, eliminating the Wifi/WifiOff flash and the React hydration warning.

### Files to edit
- `src/routes/__root.tsx` — favicon link
- `src/routes/login.tsx` — bigger PNG hero mark
- `src/components/TopHeader.tsx` — SSR-safe `online` initial state

### Out of scope
- No new icon variants generated (we reuse `/icon-512.png`)
- Sidebar / mobile bar / RegimeCard stay on the `Wind` glyph

