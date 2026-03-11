import { Component, inject } from '@angular/core';
import { AsyncPipe, DecimalPipe } from '@angular/common';
import { ProjectsService } from '../../core/services/projects.service';
import { Project } from '../../core/models/project.model';

@Component({
  selector: 'app-projects',
  standalone: true,
  imports: [AsyncPipe, DecimalPipe],
  templateUrl: './projects.component.html',
  styleUrl: './projects.component.scss'
})
export class ProjectsComponent {
  private projects = inject(ProjectsService);
  projects$ = this.projects.getProjects();

  /** Datos para el gráfico de barras de presupuestos (solo proyectos con budget > 0). */
  budgetChartData(projects: Project[]): { id: string; title: string; value: number; percent: number }[] {
    const withBudget = projects.filter((p) => p.budget != null && Number(p.budget) > 0);
    const max = Math.max(...withBudget.map((p) => Number(p.budget)), 1);
    return withBudget.map((p) => ({
      id: p.id,
      title: p.title,
      value: Number(p.budget),
      percent: (Number(p.budget) / max) * 100
    }));
  }
}
