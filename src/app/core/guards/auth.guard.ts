import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { Observable } from 'rxjs';
import { map, take } from 'rxjs';
import { SupabaseService } from '../services/supabase.service';

export const authGuard: CanActivateFn = () => {
  const router = inject(Router);
  const supabase = inject(SupabaseService);
  return new Observable<boolean>((subscriber) => {
    supabase.session.then(({ data: { session } }) => {
      subscriber.next(!!session);
      subscriber.complete();
    });
  }).pipe(
    take(1),
    map((loggedIn) => (loggedIn ? true : router.createUrlTree(['/admin/login'])))
  );
};
