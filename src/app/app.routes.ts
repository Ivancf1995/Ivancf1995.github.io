import { Routes } from '@angular/router';
import { LayoutComponent } from './layout/layout.component';

import { authGuard } from './core/guards/auth.guard';

/** Sin páginas individuales: artículos, proyectos y apps solo tienen listado con descripción breve + enlace externo (DOI/web). */
export const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    children: [
      { path: '', loadComponent: () => import('./pages/home/home.component').then(m => m.HomeComponent) },
      { path: 'portfolio', loadComponent: () => import('./pages/portfolio/portfolio.component').then(m => m.PortfolioComponent) },
      { path: 'publications', loadComponent: () => import('./pages/publications/publications.component').then(m => m.PublicationsComponent) },
      { path: 'projects', loadComponent: () => import('./pages/projects/projects.component').then(m => m.ProjectsComponent) },
      { path: 'formacion', loadComponent: () => import('./pages/formacion/formacion.component').then(m => m.FormacionComponent) },
      { path: 'contact', loadComponent: () => import('./pages/contact/contact.component').then(m => m.ContactComponent) }
    ]
  },
  {
    path: 'admin',
    children: [
      { path: 'login', loadComponent: () => import('./admin/login/login.component').then(m => m.LoginComponent) },
      { path: '', loadComponent: () => import('./admin/dashboard/dashboard.component').then(m => m.DashboardComponent), canActivate: [authGuard] }
    ]
  },
  { path: '**', redirectTo: '' }
];
