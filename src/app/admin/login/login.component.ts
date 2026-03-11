import { Component, inject } from '@angular/core';
import { Router } from '@angular/router';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { SupabaseService } from '../../core/services/supabase.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  private fb = inject(NonNullableFormBuilder);
  private supabase = inject(SupabaseService);
  private router = inject(Router);

  form = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    password: ['', Validators.required]
  });
  loading = false;
  error = '';

  async onSubmit(): Promise<void> {
    if (this.form.invalid || this.loading) return;
    this.loading = true;
    this.error = '';
    const { email, password } = this.form.getRawValue();
    const { error } = await this.supabase.auth.signInWithPassword({ email, password });
    this.loading = false;
    if (error) {
      this.error = error.message;
      return;
    }
    this.router.navigate(['/admin']);
  }
}
