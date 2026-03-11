import { Component, signal, computed, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { TranslateModule } from '@ngx-translate/core';
import { GalleryService } from '../../core/services/gallery.service';

@Component({
  selector: 'app-gallery',
  standalone: true,
  imports: [FormsModule, RouterLink, TranslateModule],
  templateUrl: './gallery.component.html',
  styleUrl: './gallery.component.scss'
})
export class GalleryComponent {
  private galleryService = inject(GalleryService);

  searchQuery = signal('');
  private items = toSignal(this.galleryService.getGallery(), { initialValue: [] });

  filteredItems = computed(() => {
    const list = this.items();
    const q = this.searchQuery().toLowerCase().trim();
    if (!q) return list;
    return list.filter(
      item =>
        (item.title ?? '').toLowerCase().includes(q) ||
        (item.author ?? '').toLowerCase().includes(q) ||
        (item.tags ?? '')
          .split(',')
          .map(t => t.trim().toLowerCase())
          .some(t => t && (q.includes(t) || t.includes(q)))
    );
  });
}
