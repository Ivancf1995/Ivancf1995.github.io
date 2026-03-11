-- Portfolio Iván Cortés: tablas y RLS
-- Ejecutar en Supabase SQL Editor (o via MCP si está disponible).

-- Publicaciones (desde DOI; imagen opcional)
create table if not exists public.publications (
  id uuid primary key default gen_random_uuid(),
  doi text not null unique,
  title text not null,
  authors text,
  year int,
  journal text,
  url text,
  abstract text,
  image_url text,
  created_at timestamptz default now()
);

alter table public.publications add column if not exists image_url text;

alter table public.publications enable row level security;

create policy "Publicaciones: lectura pública"
  on public.publications for select
  using (true);

create policy "Publicaciones: solo autenticados pueden insertar/actualizar/eliminar"
  on public.publications for all
  using (auth.role() = 'authenticated')
  with check (auth.role() = 'authenticated');

-- Apps (portfolio)
create table if not exists public.apps (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  description text,
  image_url text,
  web_url text not null,
  "order" int not null default 0,
  created_at timestamptz default now()
);

alter table public.apps enable row level security;

create policy "Apps: lectura pública"
  on public.apps for select using (true);

create policy "Apps: solo autenticados escriben"
  on public.apps for all
  using (auth.role() = 'authenticated')
  with check (auth.role() = 'authenticated');

-- Proyectos (título, descripción, presupuesto, equipo, imagen opcional)
create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  description text,
  status text,
  url text,
  budget numeric(12,2),
  team text,
  image_url text,
  "order" int not null default 0,
  created_at timestamptz default now()
);

-- Si la tabla ya existía sin budget/team/image_url, añadir columnas
alter table public.projects add column if not exists budget numeric(12,2);
alter table public.projects add column if not exists team text;
alter table public.projects add column if not exists image_url text;

alter table public.projects enable row level security;

create policy "Projects: lectura pública"
  on public.projects for select using (true);

create policy "Projects: solo autenticados escriben"
  on public.projects for all
  using (auth.role() = 'authenticated')
  with check (auth.role() = 'authenticated');

-- Mensajes de contacto
create table if not exists public.contact_messages (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null,
  message text not null,
  created_at timestamptz default now()
);

alter table public.contact_messages enable row level security;

create policy "Contact: cualquiera puede insertar"
  on public.contact_messages for insert
  with check (true);

create policy "Contact: solo autenticados pueden leer"
  on public.contact_messages for select
  using (auth.role() = 'authenticated');

-- Formación: trabajos, estudios, cursos, idiomas, lenguajes de programación (texto libre + nivel)
create table if not exists public.formation (
  id uuid primary key default gen_random_uuid(),
  type text not null check (type in ('job', 'study', 'course', 'language', 'programming')),
  title text,
  content text not null,
  level text,
  "order" int not null default 0,
  created_at timestamptz default now()
);

alter table public.formation enable row level security;

create policy "Formation: lectura pública"
  on public.formation for select using (true);

create policy "Formation: solo autenticados escriben"
  on public.formation for all
  using (auth.role() = 'authenticated')
  with check (auth.role() = 'authenticated');

-- Storage: bucket para imágenes de apps (y demás).
-- Si el INSERT falla por permisos, crea el bucket desde Dashboard → Storage → New bucket (id: portfolio-images, public: sí).
insert into storage.buckets (id, name, public)
values ('portfolio-images', 'portfolio-images', true)
on conflict (id) do update set public = true;

create policy "portfolio-images: lectura pública"
  on storage.objects for select
  using (bucket_id = 'portfolio-images');

create policy "portfolio-images: subida solo autenticados"
  on storage.objects for insert
  to authenticated
  with check (bucket_id = 'portfolio-images');

create policy "portfolio-images: actualizar/borrar solo autenticados"
  on storage.objects for update
  to authenticated
  using (bucket_id = 'portfolio-images');

create policy "portfolio-images: delete solo autenticados"
  on storage.objects for delete
  to authenticated
  using (bucket_id = 'portfolio-images');
