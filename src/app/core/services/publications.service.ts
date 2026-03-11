import { Injectable } from '@angular/core';
import { map, Observable } from 'rxjs';
import { SupabaseService } from './supabase.service';
import { Publication } from '../models/publication.model';

interface CrossRefMessage {
  title?: string[];
  author?: Array<{ given?: string; family?: string; name?: string }>;
  published?: { 'date-parts'?: number[][] };
  container?: string[];
  abstract?: string;
}

interface DataCiteAttrs {
  title?: string | Record<string, string>;
  creator?: Array<{ name?: string }>;
  publicationYear?: number | string;
  publisher?: string;
  descriptions?: Array<{ descriptionType?: string; description?: string }>;
}

@Injectable({ providedIn: 'root' })
export class PublicationsService {
  private readonly table = 'publications';

  constructor(private supabase: SupabaseService) {}

  getPublications(): Observable<Publication[]> {
    return new Observable((subscriber) => {
      this.supabase.supabase
        .from(this.table)
        .select('*')
        .order('year', { ascending: false })
        .then(({ data, error }) => {
          if (error) subscriber.error(error);
          else subscriber.next((data as Publication[]) ?? []);
          subscriber.complete();
        });
    });
  }

  getCount(): Observable<number> {
    return this.getPublications().pipe(
      map((list) => list.length)
    );
  }

  async addFromDoi(doi: string): Promise<{ data?: Publication; error?: Error }> {
    const rawDoi = doi.trim();
    if (!rawDoi) return { error: new Error('DOI vacío') };

    const normalizedDoi = rawDoi.replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '').trim();

    const tryEdge = await this.supabase.supabase.functions.invoke('resolve-doi', { body: { doi: normalizedDoi } });
    if (!tryEdge.error && tryEdge.data) {
      return { data: tryEdge.data as Publication };
    }

    const resolved = await this.resolveDoiInClient(normalizedDoi);
    if (resolved.error) return { error: resolved.error };

    const row = resolved.data!;
    const { data: inserted, error } = await this.supabase.supabase
      .from(this.table)
      .upsert(row, { onConflict: 'doi' })
      .select('*')
      .single();

    if (error) return { error: error as unknown as Error };
    return { data: inserted as Publication };
  }

  private async resolveDoiInClient(doi: string): Promise<{ data?: Omit<Publication, 'id' | 'created_at'>; error?: Error }> {
    const url = `https://doi.org/${doi}`;
    let title = '';
    let authors: string | null = null;
    let year: number | null = null;
    let journal: string | null = null;
    let abstract: string | null = null;

    const crossRefRes = await fetch(
      `https://api.crossref.org/works/${encodeURIComponent(doi)}`,
      { headers: { Accept: 'application/json' } }
    );
    if (crossRefRes.ok) {
      const data = await crossRefRes.json();
      const msg = data?.message as CrossRefMessage | undefined;
      if (msg?.title) {
        title = Array.isArray(msg.title) ? msg.title[0] : String(msg.title);
        if (msg.author?.length) {
          authors = msg.author
            .map((a) => a.name || [a.given, a.family].filter(Boolean).join(' '))
            .filter(Boolean)
            .join(', ') || null;
        }
        const parts = msg.published?.['date-parts']?.[0];
        year = parts?.[0] ? parseInt(String(parts[0]), 10) : null;
        journal = Array.isArray(msg.container) ? msg.container[0] : msg.container || null;
        abstract = msg.abstract ?? null;
      }
    }

    if (!title && crossRefRes.status === 404) {
      const dataCiteRes = await fetch(
        `https://api.datacite.org/dois/${encodeURIComponent(doi)}`,
        { headers: { Accept: 'application/vnd.api+json' } }
      );
      if (dataCiteRes.ok) {
        const data = await dataCiteRes.json();
        const attrs = data?.data?.attributes as DataCiteAttrs | undefined;
        if (attrs?.title) {
          title = typeof attrs.title === 'string' ? attrs.title : (attrs.title?.['en'] || Object.values(attrs.title || {})[0] || '');
          const creators = attrs.creator;
          authors = Array.isArray(creators) ? creators.map((c) => c?.name).filter(Boolean).join(', ') || null : null;
          const py = attrs.publicationYear;
          year = py != null ? parseInt(String(py), 10) : null;
          journal = attrs.publisher ?? null;
          const desc = attrs.descriptions?.find((d) => d.descriptionType === 'Abstract');
          abstract = desc?.description ?? null;
        }
      }
    }

    if (!title) {
      return { error: new Error('DOI no encontrado o inválido') };
    }

    return {
      data: {
        doi,
        title,
        authors,
        year,
        journal,
        url: `https://doi.org/${doi}`,
        abstract,
        image_url: null
      }
    };
  }

  async update(
    id: string,
    payload: Partial<Pick<Publication, 'title' | 'authors' | 'year' | 'journal' | 'url' | 'abstract' | 'image_url'>>
  ): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).update(payload).eq('id', id);
    return { error: r.error };
  }

  async delete(id: string): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).delete().eq('id', id);
    return { error: r.error };
  }
}
