/**
 * Crea un usuario admin en Supabase (ejecutar una sola vez).
 * No guardes la contraseña en el repo.
 *
 * Uso (desde la raíz del proyecto):
 *   node scripts/create-admin-user.js <email> "<contraseña>"
 *
 * Requiere variables de entorno:
 *   SUPABASE_URL=https://xboezmgxqpoxsefzqutc.supabase.co
 *   SUPABASE_SERVICE_ROLE_KEY=<tu service role key desde Dashboard → Settings → API>
 *
 * Ejemplo:
 *   SUPABASE_URL=https://xboezmgxqpoxsefzqutc.supabase.co SUPABASE_SERVICE_ROLE_KEY=eyJ... node scripts/create-admin-user.js ivancf1995@hotmail.com "TuContraseña"
 */

const { createClient } = require('@supabase/supabase-js');

const url = process.env.SUPABASE_URL || 'https://xboezmgxqpoxsefzqutc.supabase.co';
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const email = process.argv[2];
const password = process.argv[3];

if (!serviceRoleKey) {
  console.error('Falta SUPABASE_SERVICE_ROLE_KEY (Dashboard → Settings → API → service_role)');
  process.exit(1);
}
if (!email || !password) {
  console.error('Uso: node scripts/create-admin-user.js <email> "<contraseña>"');
  process.exit(1);
}

const supabase = createClient(url, serviceRoleKey, { auth: { autoRefreshToken: false, persistSession: false } });

async function main() {
  const { data, error } = await supabase.auth.admin.createUser({
    email,
    password,
    email_confirm: true
  });
  if (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
  console.log('Usuario creado:', data.user?.email);
}

main();
