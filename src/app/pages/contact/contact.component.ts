import { Component, inject } from '@angular/core';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { ContactService } from '../../core/services/contact.service';

@Component({
  selector: 'app-contact',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './contact.component.html',
  styleUrl: './contact.component.scss'
})
export class ContactComponent {
  private fb = inject(NonNullableFormBuilder);
  private contact = inject(ContactService);

  form = this.fb.group({
    name: ['', Validators.required],
    email: ['', [Validators.required, Validators.email]],
    message: ['', Validators.required]
  });
  sending = false;
  success = false;
  error = '';

  async onSubmit(): Promise<void> {
    if (this.form.invalid || this.sending) return;
    this.sending = true;
    this.error = '';
    this.success = false;
    const { name, email, message } = this.form.getRawValue();
    const { error } = await this.contact.send(name, email, message);
    this.sending = false;
    if (error) {
      this.error = (error as { message?: string }).message || 'Error al enviar';
      return;
    }
    this.success = true;
    this.form.reset();
  }
}
