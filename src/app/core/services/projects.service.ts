import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SupabaseService } from './supabase.service';
import { Project } from '../models/project.model';

@Injectable({ providedIn: 'root' })
export class ProjectsService {
  private readonly table = 'projects';

  constructor(private supabase: SupabaseService) {}

  getProjects(): Observable<Project[]> {
    return new Observable((subscriber) => {
      this.supabase.supabase
        .from(this.table)
        .select('*')
        .order('order', { ascending: true })
        .then(({ data, error }) => {
          if (error) subscriber.error(error);
          else subscriber.next((data as Project[]) ?? []);
          subscriber.complete();
        });
    });
  }

  async create(payload: {
    title: string;
    description?: string;
    status?: string;
    url?: string;
    budget?: number;
    team?: string;
    image_url?: string;
    order?: number;
  }): Promise<{ data?: Project; error?: Error }> {
    const { data, error } = await this.supabase.supabase
      .from(this.table)
      .insert({
        title: payload.title,
        description: payload.description ?? null,
        status: payload.status ?? null,
        url: payload.url ?? null,
        budget: payload.budget ?? null,
        team: payload.team ?? null,
        image_url: payload.image_url ?? null,
        order: payload.order ?? 0
      })
      .select('*')
      .single();
    if (error) return { error: error as unknown as Error };
    return { data: data as Project };
  }

  async update(
    id: string,
    payload: Partial<Pick<Project, 'title' | 'description' | 'status' | 'url' | 'budget' | 'team' | 'image_url' | 'order'>>
  ): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).update(payload).eq('id', id);
    return { error: r.error };
  }

  async delete(id: string): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase.from(this.table).delete().eq('id', id);
    return { error: r.error };
  }
}
