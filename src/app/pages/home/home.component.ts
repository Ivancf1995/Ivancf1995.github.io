import { Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { TranslateModule } from '@ngx-translate/core';
import { AsyncPipe } from '@angular/common';
import { AppsService } from '../../core/services/apps.service';
import { PublicationsService } from '../../core/services/publications.service';
import { ProjectsService } from '../../core/services/projects.service';
import { environment } from '../../../environments/environment';

const LATEST_COUNT = 3;

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [TranslateModule, AsyncPipe, RouterLink],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {
  private apps = inject(AppsService);
  private publications = inject(PublicationsService);
  private projects = inject(ProjectsService);

  readonly socialLinks = environment.socialLinks ?? { linkedin: '', orcid: '', googleScholar: '', github: '' };

  appsCount$ = this.apps.getCount();
  publicationsCount$ = this.publications.getCount();
  projects$ = this.projects.getProjects();
  publications$ = this.publications.getPublications();
  apps$ = this.apps.getApps();

  readonly latestCount = LATEST_COUNT;
}
