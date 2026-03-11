import { Component, inject } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { TranslateModule } from '@ngx-translate/core';
import { FormationService } from '../../core/services/formation.service';
import { FormationItem, FormationType } from '../../core/models/formation.model';

const FORMATION_ORDER: FormationType[] = ['job', 'study', 'course', 'language', 'programming'];

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
