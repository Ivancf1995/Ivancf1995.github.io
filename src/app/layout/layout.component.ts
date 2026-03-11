import { Component, inject, OnInit } from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { Title, Meta } from '@angular/platform-browser';
import { filter } from 'rxjs';
import { HeaderComponent } from './header/header.component';
import { FooterComponent } from './footer/footer.component';

@Component({
  selector: 'app-layout',
  standalone: true,
  imports: [RouterOutlet, HeaderComponent, FooterComponent],
  templateUrl: './layout.component.html',
  styleUrl: './layout.component.scss'
})
export class LayoutComponent implements OnInit {
  private router = inject(Router);
  private title = inject(Title);
  private meta = inject(Meta);

  ngOnInit(): void {
    this.updateSeo();
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe(() => this.updateSeo());
  }

  private updateSeo(): void {
    let route = this.router.routerState.root;
    while (route.firstChild) route = route.firstChild;
    const data = route.snapshot.data;
    const title = data['title'] as string | undefined;
    const description = data['description'] as string | undefined;
    if (title) this.title.setTitle(title);
    if (description) {
      this.meta.updateTag({ name: 'description', content: description });
      this.meta.updateTag({ property: 'og:description', content: description });
      this.meta.updateTag({ name: 'twitter:description', content: description });
    }
  }
}
