import { Component, inject } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { TranslateModule } from '@ngx-translate/core';
import { FormationService } from '../../core/services/formation.service';
import { FormationItem, FormationType } from '../../core/models/formation.model';

const FORMATION_ORDER: FormationType[] = ['job', 'study', 'course', 'language', 'programming'];

/** Tecnologías/lenguajes que tienen icono; orden: más específico primero (ej. "Google Scholar" antes de "Google"). */
const TECH_MATCHES: { pattern: RegExp; slug: string }[] = [
  { pattern: /\bpython\b/i, slug: 'python' },
  { pattern: /\bR\b/, slug: 'r' },
  { pattern: /\bangular\b/i, slug: 'angular' },
  { pattern: /\bgithub\b/i, slug: 'github' },
  { pattern: /\bjavascript\b/i, slug: 'javascript' },
  { pattern: /\btypescript\b/i, slug: 'typescript' },
  { pattern: /\breact\b/i, slug: 'react' },
  { pattern: /\bnode\.?js\b|nodejs/i, slug: 'node' },
  { pattern: /\bsql\b/i, slug: 'sql' },
  { pattern: /\bhtml\b/i, slug: 'html' },
  { pattern: /\bcss\b/i, slug: 'css' },
  { pattern: /\bgit\b/i, slug: 'git' },
  { pattern: /\bjava\b/i, slug: 'java' },
  { pattern: /\bmatlab\b/i, slug: 'matlab' },
];

@Component({
  selector: 'app-formacion',
  standalone: true,
  imports: [AsyncPipe, TranslateModule],
  templateUrl: './formacion.component.html',
  styleUrl: './formacion.component.scss'
})
export class FormacionComponent {
  private formation = inject(FormationService);
  formation$ = this.formation.getFormation();

  /** Devuelve el slug del icono si el ítem menciona una tecnología conocida (title, content o level). */
  getTechSlug(item: FormationItem): string | null {
    const text = [item.title, item.content, item.level].filter(Boolean).join(' ');
    if (!text) return null;
    const found = TECH_MATCHES.find(({ pattern }) => pattern.test(text));
    return found?.slug ?? null;
  }

  /** Ruta al icono en assets (solo para los que tienes en assets/i18n/images). */
  getTechIconPath(slug: string): string | null {
    const withImage = ['python', 'r', 'angular', 'github'];
    if (!withImage.includes(slug)) return null;
    const file = slug === 'r' ? 'R.png' : `${slug}.png`;
    return `assets/i18n/images/${file}`;
  }

  typeLabelKey(type: FormationType): string {
    const keys: Record<FormationType, string> = {
      job: 'FORMATION.JOBS',
      study: 'FORMATION.STUDIES',
      course: 'FORMATION.COURSES',
      language: 'FORMATION.LANGUAGES',
      programming: 'FORMATION.PROGRAMMING'
    };
    return keys[type];
  }

  /** Agrupa por tipo y devuelve array { type, list } para la plantilla */
  groupByType(items: FormationItem[]): { type: FormationType; list: FormationItem[] }[] {
    const map = new Map<FormationType, FormationItem[]>();
    for (const t of FORMATION_ORDER) map.set(t, []);
    for (const item of items) {
      const list = map.get(item.type);
      if (list) list.push(item);
    }
    return FORMATION_ORDER.map((type) => ({ type, list: map.get(type)! })).filter((g) => g.list.length > 0);
  }
}
