import { Routes } from '@angular/router';
import { LayoutComponent } from './layout/layout.component';

import { authGuard } from './core/guards/auth.guard';

/** Sin páginas individuales: artículos, proyectos y apps solo tienen listado con descripción breve + enlace externo (DOI/web). */
const SITE_TITLE = 'Iván Cortés Fernández — Portfolio';

export const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    children: [
      {
        path: '',
        loadComponent: () => import('./pages/home/home.component').then(m => m.HomeComponent),
        data: { title: SITE_TITLE, description: 'Biólogo, investigador y desarrollador. IA en Ecología y Salud. Publicaciones, proyectos y webs.' }
      },
      {
        path: 'portfolio',
        loadComponent: () => import('./pages/portfolio/portfolio.component').then(m => m.PortfolioComponent),
        data: { title: `Webs y apps | ${SITE_TITLE}`, description: 'Webs y aplicaciones que desarrollo. Enlaces y descripción de cada proyecto.' }
      },
      {
        path: 'galeria',
        loadComponent: () => import('./pages/gallery/gallery.component').then(m => m.GalleryComponent),
        data: { title: `Galería | ${SITE_TITLE}`, description: 'Galería personal de imágenes y fotos.' }
      },
      {
        path: 'publications',
        loadComponent: () => import('./pages/publications/publications.component').then(m => m.PublicationsComponent),
        data: { title: `Publicaciones | ${SITE_TITLE}`, description: 'Artículos científicos con DOI. Título, autores, año y enlace.' }
      },
      {
        path: 'projects',
        loadComponent: () => import('./pages/projects/projects.component').then(m => m.ProjectsComponent),
        data: { title: `Proyectos de investigación | ${SITE_TITLE}`, description: 'Proyectos de investigación con presupuesto, equipo y descripción.' }
      },
      {
        path: 'formacion',
        loadComponent: () => import('./pages/formacion/formacion.component').then(m => m.FormacionComponent),
        data: { title: `Formación | ${SITE_TITLE}`, description: 'Trabajos, estudios, cursos, idiomas y lenguajes de programación.' }
      },
      {
        path: 'contact',
        loadComponent: () => import('./pages/contact/contact.component').then(m => m.ContactComponent),
        data: { title: `Contacto | ${SITE_TITLE}`, description: 'Formulario de contacto.' }
      }
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
