import { Component, inject } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { TranslateModule } from '@ngx-translate/core';
import { AppsService } from '../../core/services/apps.service';

@Component({
  selector: 'app-portfolio',
  standalone: true,
  imports: [AsyncPipe, TranslateModule],
  templateUrl: './portfolio.component.html',
  styleUrl: './portfolio.component.scss'
})
export class PortfolioComponent {
  private apps = inject(AppsService);
  apps$ = this.apps.getApps();
}
