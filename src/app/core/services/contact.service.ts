import { Injectable } from '@angular/core';
import { SupabaseService } from './supabase.service';

@Injectable({ providedIn: 'root' })
export class ContactService {
  private readonly table = 'contact_messages';

  constructor(private supabase: SupabaseService) {}

  async send(name: string, email: string, message: string): Promise<{ error: Error | null }> {
    const r = await this.supabase.supabase
      .from(this.table)
      .insert({ name, email, message });
    return { error: r.error };
  }
}
