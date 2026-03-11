import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SupabaseService } from './supabase.service';
import { GalleryItem } from '../models/gallery.model';

@Injectable({ providedIn: 'root' })
export class GalleryService {
  private readonly table = 'gallery';

  constructor(private supabase: SupabaseService) {}

  getGallery(): Observable<GalleryItem[]> {
    return new Observable((subscriber) => {
      this.supabase.supabase
        .from(this.table)
        .select('*')
        .order('order', { ascending: true })
        .then(({ data, error }) => {
          if (error) subscriber.error(error);
          else subscriber.next((data as GalleryItem[]) ?? []);
          subscriber.complete();
        });
    });
  }

  async create(payload: {
    title?: string;
    image_url: string;
    author?: string;
    tags?: string;
    order?: number;
  }): Promise<{ data?: GalleryItem; error?: Error }> {
    const { data, error } = await this.supabase.supabase
      .from(this.table)
      .insert({
        title: payload.title ?? null,
        image_url: payload.image_url,
        author: payload.author ?? null,
        tags: payload.tags ?? null,
        order: payload.order ?? 0
      })
      .select('*')
      .single();
    return error ? { error: error as unknown as Error } : { data: data as GalleryItem };
  }

  async update(
    id: string,
    payload: Partial<Pick<GalleryItem, 'title' | 'image_url' | 'author' | 'tags' | 'order'>>
  ): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).update(payload).eq('id', id);
    return { error: r.error };
  }

  async delete(id: string): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).delete().eq('id', id);
    return { error: r.error };
  }
}
