// Edge Function: resuelve DOI con CrossRef (y DataCite si falla) e inserta en publications
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type'
};

interface DoiBody {
  doi?: string;
}

interface CrossRefAuthor {
  given?: string;
  family?: string;
  name?: string;
}

function normalizeDoi(doi: string): string {
  const s = doi.trim().replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '');
  return s;
}

function formatAuthors(authors: CrossRefAuthor[] | undefined): string | null {
  if (!authors?.length) return null;
  return authors
    .map((a) => a.name || [a.given, a.family].filter(Boolean).join(' '))
    .filter(Boolean)
    .join(', ') || null;
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response(null, { headers: corsHeaders });

  try {
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    const jwt = authHeader.replace('Bearer ', '');
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      { global: { headers: { Authorization: authHeader } } }
    );
    const { data: { user }, error: authError } = await supabase.auth.getUser(jwt);
    if (authError || !user) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    const body = (await req.json()) as DoiBody;
    const rawDoi = body?.doi;
    if (!rawDoi || typeof rawDoi !== 'string') {
      return new Response(JSON.stringify({ error: 'Missing doi' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    const doi = normalizeDoi(rawDoi);
    let title = '';
    let authors: string | null = null;
    let year: number | null = null;
    let journal: string | null = null;
    let url = `https://doi.org/${doi}`;
    let abstract: string | null = null;

    // CrossRef
    const crossRefRes = await fetch(
      `https://api.crossref.org/works/${encodeURIComponent(doi)}`,
      { headers: { Accept: 'application/json' } }
    );
    if (crossRefRes.ok) {
      const data = await crossRefRes.json();
      const message = data?.message;
      if (message) {
        const titles = message.title;
        title = Array.isArray(titles) ? titles[0] : titles || '';
        authors = formatAuthors(message.author);
        const published = message.published?.['date-parts']?.[0];
        year = published?.[0] ? parseInt(String(published[0]), 10) : null;
        const container = message.container;
        journal = Array.isArray(container) ? container[0] : container || null;
        abstract = message.abstract || null;
      }
    }

    if (!title && crossRefRes.status === 404) {
      // DataCite fallback
      const dataCiteRes = await fetch(
        `https://api.datacite.org/dois/${encodeURIComponent(doi)}`,
        { headers: { Accept: 'application/vnd.api+json' } }
      );
      if (dataCiteRes.ok) {
        const data = await dataCiteRes.json();
        const attrs = data?.data?.attributes;
        if (attrs) {
          const titles = attrs.title;
          title = typeof titles === 'string' ? titles : titles?.['en'] || titles?.[Object.keys(titles || {})[0]] || '';
          const creators = attrs.creator;
          authors = Array.isArray(creators)
            ? creators.map((c: { name?: string }) => c?.name).filter(Boolean).join(', ')
            : null;
          const pubYear = attrs.publicationYear;
          year = pubYear ? parseInt(String(pubYear), 10) : null;
          journal = attrs.publisher || null;
          abstract = attrs.descriptions?.find((d: { descriptionType?: string }) => d.descriptionType === 'Abstract')?.description || null;
        }
      }
    }

    if (!title) {
      return new Response(
        JSON.stringify({ error: 'DOI not found or invalid', doi }),
        { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const row = {
      doi,
      title,
      authors,
      year,
      journal,
      url,
      abstract
    };

    const { data: inserted, error } = await supabase
      .from('publications')
      .upsert(row, { onConflict: 'doi' })
      .select('*')
      .single();

    if (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    return new Response(JSON.stringify(inserted), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  } catch (e) {
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : 'Internal error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
