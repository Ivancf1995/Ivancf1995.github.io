import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SupabaseService } from './supabase.service';
import { FormationItem, FormationType } from '../models/formation.model';

@Injectable({ providedIn: 'root' })
export class FormationService {
  private readonly table = 'formation';

  constructor(private supabase: SupabaseService) {}

  getFormation(): Observable<FormationItem[]> {
    return new Observable((subscriber) => {
      this.supabase.supabase
        .from(this.table)
        .select('*')
        .order('order', { ascending: true })
        .order('created_at', { ascending: true })
        .then(({ data, error }) => {
          if (error) subscriber.error(error);
          else subscriber.next((data as FormationItem[]) ?? []);
          subscriber.complete();
        });
    });
  }

  async create(payload: {
    type: FormationType;
    title?: string;
    content: string;
    level?: string;
    order?: number;
  }): Promise<{ data?: FormationItem; error?: Error }> {
    const { data, error } = await this.supabase.supabase
      .from(this.table)
      .insert({
        type: payload.type,
        title: payload.title ?? null,
        content: payload.content,
        level: payload.level ?? null,
        order: payload.order ?? 0
      })
      .select('*')
      .single();
    if (error) return { error: error as unknown as Error };
    return { data: data as FormationItem };
  }

  async update(
    id: string,
    payload: Partial<Pick<FormationItem, 'type' | 'title' | 'content' | 'level' | 'order'>>
  ): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).update(payload).eq('id', id);
    return { error: r.error };
  }

  async delete(id: string): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).delete().eq('id', id);
    return { error: r.error };
  }
}
