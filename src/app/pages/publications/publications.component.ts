import { Component, inject } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TranslateModule } from '@ngx-translate/core';
import { PublicationsService } from '../../core/services/publications.service';
import { Publication } from '../../core/models/publication.model';

@Component({
  selector: 'app-publications',
  standalone: true,
  imports: [AsyncPipe, FormsModule, TranslateModule],
  templateUrl: './publications.component.html',
  styleUrl: './publications.component.scss'
})
export class PublicationsComponent {
  private publications = inject(PublicationsService);
  publications$ = this.publications.getPublications();

  selectedYear: number | null = null;

  /** Años distintos con publicación, orden descendente */
  yearsFromPublications(pubs: Publication[]): number[] {
    const years = [...new Set(pubs.map((p) => p.year).filter((y): y is number => y != null))];
    return years.sort((a, b) => b - a);
  }

  filteredPublications(pubs: Publication[]): Publication[] {
    if (this.selectedYear == null) return pubs;
    return pubs.filter((p) => p.year === this.selectedYear);
  }

  onYearChange(value: string): void {
    this.selectedYear = value === '' ? null : parseInt(value, 10);
  }
}
