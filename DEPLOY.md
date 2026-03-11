# Despliegue en GitHub Pages (o similar)

## GitHub Pages

### 1. Configuración en el repo

- Ve a **Settings → Pages**.
- En **Build and deployment → Source** elige **GitHub Actions**.

### 2. Despliegue automático

Al hacer **push a `main`** (o ejecutar el workflow manualmente desde **Actions**), el workflow `.github/workflows/deploy-gh-pages.yml`:

1. Instala dependencias y construye la app con `--base-href /<nombre-repo>/`
2. Copia `index.html` a `404.html` para que las rutas de la SPA (/portfolio, /publications, etc.) funcionen al recargar o al entrar por enlace directo
3. Publica el contenido en GitHub Pages

**URL del sitio:**
- Si el repo se llama **`<usuario>.github.io`** (p. ej. `Ivancf1995.github.io`): la web queda en **`https://<usuario>.github.io/`** (raíz). El workflow usa `--base-href /` en ese caso.
- Si no: **`https://<tu-usuario>.github.io/<nombre-repo>/`**

### 3. Build local (para probar o subir a otro host)

Si el nombre del repo no es `portfolio`, cambia el `--base-href` en el script:

```bash
npm run build:gh-pages
```

El resultado queda en `dist/portfolio/browser/` (Angular 19). Para probar en local con ese base-href:

```bash
npx serve dist/portfolio/browser -s
```

(`-s` redirige 404 a index.html; útil para probar rutas.)

### 4. Repo con la app en una subcarpeta

Si la app está en una subcarpeta (por ejemplo `portfolio/`):

- En el workflow, en **Install dependencies** y **Build for GitHub Pages** añade `working-directory: portfolio`.
- En **Upload artifact** usa `path: portfolio/dist/portfolio/browser`.
- En **SPA fallback** usa `cp portfolio/dist/portfolio/browser/index.html portfolio/dist/portfolio/browser/404.html`.

---

## Otros hosts estáticos (Netlify, Vercel, etc.)

- **Build:** `npm run build` (producción) o `npm run build:gh-pages` si el sitio va en una subruta.
- **Directorio a publicar:** `dist/portfolio/browser` (Angular 19 application builder)
- **Base href:** Si el sitio está en la raíz del dominio, usa `--base-href /` en el build. Si está en una subruta (ej. `/mi-portfolio/`), usa `--base-href /mi-portfolio/`.
- **SPA:** Configura redirección de todas las rutas a `index.html` (en Netlify: `/* /index.html 200`, en Vercel suele venir por defecto para SPAs).
