# Información sensible y variables de entorno

## Qué no debe ser público en el repo

| Dato | Riesgo | Dónde usarlo |
|------|--------|--------------|
| **SUPABASE_SERVICE_ROLE_KEY** | Alto. Bypasea RLS; acceso total al proyecto. | Solo en scripts que se ejecutan en tu máquina o en CI (ej. `create-admin-user.js`). **Nunca** en el frontend ni en `environment.ts`. |
| **SUPABASE_URL** y **SUPABASE_ANON_KEY** | Bajo. La anon key está pensada para el navegador y se protege con RLS. Muchos proyectos la commitean. | Si no quieres que aparezcan en el repo, ponlas en `.env` y el script `scripts/replace-env.js` las inyecta al construir. |
| **Enlaces sociales** (LinkedIn, ORCID, etc.) | Ninguno. Son URLs públicas. | Opcional meterlas en `.env` para cambiarlas sin tocar código. |

## Dónde va cada variable

- **`.env`** (local, no se sube al repo): copia de `.env.example` con valores reales. La usas en desarrollo y, si quieres, para generar los `environment` sin commitear claves.
- **`.env.example`** (sí se sube): plantilla con los nombres de las variables y valores de ejemplo o vacíos. Quién clone el repo sabe qué tiene que definir.
- **GitHub Secrets** (para deploy): en un repo con GitHub Actions, en Settings → Secrets defines `SUPABASE_URL` y `SUPABASE_ANON_KEY` (y si hace falta `SUPABASE_SERVICE_ROLE_KEY` solo para un job que cree usuarios, etc.). El workflow inyecta esas variables al ejecutar `replace-env.js` y luego `ng build`.

## Flujo con plantillas y script

En el repo solo se versionan las **plantillas** (`src/environments/environment.ts.template` y `environment.prod.ts.template`) con placeholders (`__SUPABASE_URL__`, etc.). Los archivos `environment.ts` y `environment.prod.ts` se **generan** con `scripts/replace-env.js` y están en `.gitignore`.

1. **Local**
   - Copia `.env.example` a `.env` y rellena los valores.
   - Al ejecutar `npm start` o `npm run build` se ejecuta antes `env:generate`, que lee `.env` y genera los `environment.*.ts`.
2. **CI (GitHub Actions)**
   - En el repo: **Settings → Secrets and variables → Actions** crea los secrets: `SUPABASE_URL`, `SUPABASE_ANON_KEY` y, opcionalmente, `SOCIAL_LINKEDIN`, `SOCIAL_ORCID`, `SOCIAL_GOOGLE_SCHOLAR`, `SOCIAL_GITHUB`.
   - El workflow ejecuta `npm run env:generate` con esas variables y luego el build.

Comando manual para generar los environment sin hacer build: `npm run env:generate`.

## Resumen

- **Service Role Key**: solo en `.env` local o en GitHub Secrets, y solo para scripts/admin (nunca en la app Angular).
- **URL y Anon Key de Supabase**: en `.env` y en GitHub Secrets; el script los inyecta en los `environment.*.ts` generados (estos no se suben al repo).
- **Enlaces sociales**: opcional en `.env` y en Secrets; si no se definen, quedan como cadena vacía en el build.
