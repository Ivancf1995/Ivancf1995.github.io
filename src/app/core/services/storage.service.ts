import { Injectable } from '@angular/core';
import { SupabaseService } from './supabase.service';

const BUCKET = 'portfolio-images';

@Injectable({ providedIn: 'root' })
export class StorageService {
  constructor(private supabase: SupabaseService) {}

  /**
   * Sube un archivo y devuelve la URL pública (para buckets públicos).
   * path: ej. 'apps/uuid.png' o 'projects/uuid.jpg'
   */
  async uploadImage(path: string, file: File): Promise<{ url?: string; error?: Error }> {
    const { data, error } = await this.supabase.supabase.storage
      .from(BUCKET)
      .upload(path, file, { cacheControl: '3600', upsert: true });

    if (error) return { error: error as unknown as Error };

    const { data: urlData } = this.supabase.supabase.storage.from(BUCKET).getPublicUrl(data.path);
    return { url: urlData.publicUrl };
  }

  /** Genera un path único para una imagen (apps/proyectos). */
  uniquePath(prefix: string, file: File): string {
    const ext = file.name.split('.').pop()?.toLowerCase() || 'png';
    const id = crypto.randomUUID();
    return `${prefix}/${id}.${ext}`;
  }
}
