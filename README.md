# Portfolio — Iván Cortés Fernández

Web portfolio minimalista (www.ivancortesfernandez.com). Angular + Supabase.

## Desarrollo

```bash
npm install
npm start
```

Antes de usar datos reales, configura Supabase (ver más abajo) y define las variables de entorno.

## Variables de entorno (Supabase)

Copia en `src/environments/environment.ts` (y en `environment.prod.ts` para producción):

- `supabaseUrl`: URL del proyecto (ej. `https://xxx.supabase.co`)
- `supabaseAnonKey`: clave anónima (Settings → API)

Para desarrollo local sin backend, deja las cadenas vacías; las listas quedarán vacías y el login fallará.

## Supabase: esquema y Edge Function

1. **Tablas**: En el SQL Editor de Supabase, ejecuta el contenido de `supabase/schema.sql`. Crea las tablas `publications`, `apps`, `projects`, `contact_messages` y las políticas RLS.

2. **Usuario admin**: En Authentication → Users, crea un usuario (email + contraseña) para el login de admin.

3. **Edge Function `resolve-doi`**: Despliega la función que resuelve DOIs (CrossRef/DataCite) y guarda en `publications`:
   - En Supabase Dashboard: Edge Functions → New function → nombre `resolve-doi`.
   - Pega el código de `supabase/functions/resolve-doi/index.ts`.
   - Variables de la función: `SUPABASE_URL`, `SUPABASE_ANON_KEY` (y opcionalmente `SUPABASE_SERVICE_ROLE_KEY` si cambias la función para usarlo).

## Build y despliegue (GitHub Pages)

```bash
npm run build
```

Salida en `dist/portfolio/browser/`. Para GitHub Pages con dominio personalizado usa `baseHref: '/'` (ya por defecto). Puedes usar `npx angular-cli-ghpages` o un workflow de GitHub Actions que ejecute `ng build` y suba la rama `gh-pages`.

## Estructura

- **Público**: Inicio, Portfolio (apps), Publicaciones (DOI), Proyectos, Sobre mí, Contacto. Sin páginas individuales por artículo/app/proyecto; solo listados con descripción breve y enlace externo.
- **Admin** (`/admin`): Login y dashboard para añadir publicaciones por DOI y crear/eliminar apps. Protegido con guard de autenticación.
- **i18n**: Español e inglés con ngx-translate (`assets/i18n/es.json`, `en.json`).
