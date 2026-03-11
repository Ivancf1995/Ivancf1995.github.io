import { Injectable } from '@angular/core';
import { map, Observable } from 'rxjs';
import { SupabaseService } from './supabase.service';
import { AppItem } from '../models/app.model';

@Injectable({ providedIn: 'root' })
export class AppsService {
  private readonly table = 'apps';

  constructor(private supabase: SupabaseService) {}

  getApps(): Observable<AppItem[]> {
    return new Observable((subscriber) => {
      this.supabase.supabase
        .from(this.table)
        .select('*')
        .order('order', { ascending: true })
        .then(({ data, error }) => {
          if (error) subscriber.error(error);
          else subscriber.next((data as AppItem[]) ?? []);
          subscriber.complete();
        });
    });
  }

  getCount(): Observable<number> {
    return this.getApps().pipe(
      map((list) => list.length)
    );
  }

  async create(payload: { title: string; description?: string; image_url?: string; web_url: string; order?: number }): Promise<{ data?: AppItem; error?: Error }> {
    const { data, error } = await this.supabase.supabase
      .from(this.table)
      .insert({
        title: payload.title,
        description: payload.description ?? null,
        image_url: payload.image_url ?? null,
        web_url: payload.web_url,
        order: payload.order ?? 0
      })
      .select('*')
      .single();
    return error ? { error: error as unknown as Error } : { data: data as AppItem };
  }

  async update(id: string, payload: Partial<Pick<AppItem, 'title' | 'description' | 'image_url' | 'web_url' | 'order'>>): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).update(payload).eq('id', id);
    return { error: r.error };
  }

  async delete(id: string): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).delete().eq('id', id);
    return { error: r.error };
  }
}
