import { Component, signal, effect, inject, HostListener } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { TranslateModule, TranslateService } from '@ngx-translate/core';

const MOBILE_BREAKPOINT = 768;

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, TranslateModule],
  templateUrl: './header.component.html',
  styleUrl: './header.component.scss'
})
export class HeaderComponent {
  private translate = inject(TranslateService);

  currentLang: string;
  mobileMenuOpen = signal(false);
  /** True mientras se reproduce la animación de cierre (panel se desliza a la izquierda). */
  menuClosing = signal(false);

  constructor() {
    this.currentLang = this.translate.currentLang || this.translate.defaultLang || 'es';
    effect(() => {
      const open = this.mobileMenuOpen();
      const closing = this.menuClosing();
      if (typeof document !== 'undefined') {
        document.body.classList.toggle('has-mobile-menu-open', open || closing);
      }
    });
  }

  onLogoClick(event: Event): void {
    if (typeof window !== 'undefined' && window.innerWidth <= MOBILE_BREAKPOINT) {
      event.preventDefault();
      event.stopPropagation();
      this.toggleMobileMenu();
    }
  }

  toggleMobileMenu(): void {
    if (this.menuClosing()) return;
    this.mobileMenuOpen.update((v) => !v);
  }

  closeMobileMenu(): void {
    if (this.menuClosing()) return;
    this.menuClosing.set(true);
    setTimeout(() => {
      this.mobileMenuOpen.set(false);
      this.menuClosing.set(false);
    }, 300);
  }

  @HostListener('window:resize')
  onResize(): void {
    if (typeof window !== 'undefined' && window.innerWidth > MOBILE_BREAKPOINT) {
      this.closeMobileMenu();
    }
  }

  toggleLang(): void {
    this.currentLang = this.currentLang === 'es' ? 'en' : 'es';
    this.translate.use(this.currentLang);
  }
}
