/**
 * Genera environment.ts y environment.prod.ts desde las plantillas .template
 * reemplazando placeholders con variables de entorno (o .env).
 *
 * Uso (desde la raíz del proyecto):
 *   node scripts/replace-env.js
 *   npm run env:generate
 *
 * Variables: SUPABASE_URL, SUPABASE_ANON_KEY, SOCIAL_LINKEDIN, SOCIAL_ORCID,
 *            SOCIAL_GOOGLE_SCHOLAR, SOCIAL_GITHUB
 */

const fs = require('fs');
const path = require('path');

// Cargar .env si existe (desde la raíz del proyecto)
try {
  require('dotenv').config({ path: path.join(process.cwd(), '.env') });
} catch (_) {
  // dotenv opcional si se usan solo variables de entorno
}

const ENV_DIR = path.join(process.cwd(), 'src', 'environments');
const PLACEHOLDERS = [
  'SUPABASE_URL',
  'SUPABASE_ANON_KEY',
  'SOCIAL_LINKEDIN',
  'SOCIAL_ORCID',
  'SOCIAL_GOOGLE_SCHOLAR',
  'SOCIAL_GITHUB'
];

function replaceInContent(content) {
  let out = content;
  for (const key of PLACEHOLDERS) {
    const placeholder = `__${key}__`;
    const raw = process.env[key];
    const replacement =
      raw != null && raw !== '' ? "'" + String(raw).replace(/'/g, "\\'") + "'" : "''";
    out = out.split(placeholder).join(replacement);
  }
  return out;
}

function generate(templateName, outputName) {
  const templatePath = path.join(ENV_DIR, templateName);
  const outputPath = path.join(ENV_DIR, outputName);
  if (!fs.existsSync(templatePath)) {
    console.warn(`No existe plantilla: ${templatePath}`);
    return;
  }
  const content = fs.readFileSync(templatePath, 'utf8');
  fs.writeFileSync(outputPath, replaceInContent(content), 'utf8');
  console.log(`Generado: ${outputName}`);
}

generate('environment.ts.template', 'environment.ts');
generate('environment.prod.ts.template', 'environment.prod.ts');
