import { Component, inject } from '@angular/core';
import { Router } from '@angular/router';
import { AsyncPipe } from '@angular/common';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators, FormsModule } from '@angular/forms';
import { SupabaseService } from '../../core/services/supabase.service';
import { PublicationsService } from '../../core/services/publications.service';
import { AppsService } from '../../core/services/apps.service';
import { ProjectsService } from '../../core/services/projects.service';
import { FormationService } from '../../core/services/formation.service';
import { GalleryService } from '../../core/services/gallery.service';
import { StorageService } from '../../core/services/storage.service';
import type { FormationType } from '../../core/models/formation.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [AsyncPipe, ReactiveFormsModule, FormsModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent {
  private fb = inject(NonNullableFormBuilder);
  private supabase = inject(SupabaseService);
  private publications = inject(PublicationsService);
  private apps = inject(AppsService);
  private projects = inject(ProjectsService);
  private formation = inject(FormationService);
  private gallery = inject(GalleryService);
  private storage = inject(StorageService);
  private router = inject(Router);

  doiInput = '';
  doiLoading = false;
  doiError = '';
  doiSuccess = '';
  appSaving = false;
  appImageUploading = false;
  appImagePreview: string | null = null;
  publicationImagePreview: string | null = null;
  projectImagePreview: string | null = null;
  projectAddImagePreview: string | null = null;

  publications$ = this.publications.getPublications();
  apps$ = this.apps.getApps();
  projects$ = this.projects.getProjects();
  formation$ = this.formation.getFormation();
  gallery$ = this.gallery.getGallery();
  projectSaving = false;
  formationSaving = false;
  gallerySaving = false;
  editingPubId: string | null = null;
  editingProjId: string | null = null;
  editingAppId: string | null = null;
  editingFormationId: string | null = null;
  editingGalleryId: string | null = null;
  galleryImagePreview: string | null = null;
  galleryEditImagePreview: string | null = null;

  readonly formationTypes: { value: FormationType; label: string }[] = [
    { value: 'job', label: 'Trabajo' },
    { value: 'study', label: 'Estudio' },
    { value: 'course', label: 'Curso' },
    { value: 'language', label: 'Idioma' },
    { value: 'programming', label: 'Lenguaje de programación' }
  ];

  refreshApps(): void {
    this.apps$ = this.apps.getApps();
  }

  refreshProjects(): void {
    this.projects$ = this.projects.getProjects();
  }

  refreshFormation(): void {
    this.formation$ = this.formation.getFormation();
  }

  refreshGallery(): void {
    this.gallery$ = this.gallery.getGallery();
  }

  appForm = this.fb.group({
    title: ['', Validators.required],
    description: [''],
    web_url: ['', [Validators.required]],
    image_url: ['']
  });

  projectForm = this.fb.group({
    title: ['', Validators.required],
    description: [''],
    budget: [''],
    team: [''],
    image_url: ['']
  });

  editPublicationForm = this.fb.group({
    title: ['', Validators.required],
    authors: [''],
    year: [null as number | null],
    journal: [''],
    url: [''],
    abstract: [''],
    image_url: ['']
  });

  editProjectForm = this.fb.group({
    title: ['', Validators.required],
    description: [''],
    budget: [''],
    team: [''],
    image_url: ['']
  });

  editAppForm = this.fb.group({
    title: ['', Validators.required],
    description: [''],
    web_url: ['', Validators.required],
    image_url: ['']
  });

  formationForm = this.fb.group({
    type: ['job' as FormationType, Validators.required],
    title: [''],
    content: ['', Validators.required],
    level: ['']
  });

  editFormationForm = this.fb.group({
    type: ['job' as FormationType, Validators.required],
    title: [''],
    content: ['', Validators.required],
    level: ['']
  });

  galleryForm = this.fb.group({
    title: [''],
    image_url: ['', Validators.required],
    author: [''],
    tags: ['']
  });

  editGalleryForm = this.fb.group({
    title: [''],
    image_url: ['', Validators.required],
    author: [''],
    tags: ['']
  });

  async signOut(): Promise<void> {
    await this.supabase.auth.signOut();
    this.router.navigate(['/admin/login']);
  }

  async addDoi(): Promise<void> {
    const doi = this.doiInput.trim();
    if (!doi) return;
    this.doiLoading = true;
    this.doiError = '';
    this.doiSuccess = '';
    const { data, error } = await this.publications.addFromDoi(doi);
    this.doiLoading = false;
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al resolver el DOI';
      return;
    }
    this.doiSuccess = data ? `Guardado: ${data.title}` : 'Guardado';
    this.doiInput = '';
    this.publications$ = this.publications.getPublications();
  }

  startEditPublication(pub: { id: string; title: string; authors: string | null; year: number | null; journal: string | null; url: string | null; abstract: string | null; image_url: string | null }): void {
    this.editingPubId = pub.id;
    this.publicationImagePreview = pub.image_url;
    this.editPublicationForm.patchValue({
      title: pub.title,
      authors: pub.authors ?? '',
      year: pub.year,
      journal: pub.journal ?? '',
      url: pub.url ?? '',
      abstract: pub.abstract ?? '',
      image_url: pub.image_url ?? ''
    });
  }

  cancelEditPublication(): void {
    this.editingPubId = null;
    this.publicationImagePreview = null;
  }

  async onPublicationImageFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    this.appImageUploading = true;
    this.doiError = '';
    const path = this.storage.uniquePath('publications', file);
    const { url, error } = await this.storage.uploadImage(path, file);
    this.appImageUploading = false;
    input.value = '';
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al subir la imagen';
      return;
    }
    if (url) {
      this.editPublicationForm.patchValue({ image_url: url });
      this.publicationImagePreview = url;
    }
  }

  clearPublicationImage(): void {
    this.editPublicationForm.patchValue({ image_url: '' });
    this.publicationImagePreview = null;
  }

  async saveEditPublication(): Promise<void> {
    if (!this.editingPubId || this.editPublicationForm.invalid) return;
    const v = this.editPublicationForm.getRawValue();
    const yearVal = v.year;
    const year = yearVal != null && !Number.isNaN(Number(yearVal)) ? Number(yearVal) : null;
    const { error } = await this.publications.update(this.editingPubId, {
      title: v.title,
      authors: v.authors || null,
      year,
      journal: v.journal || null,
      url: v.url || null,
      abstract: v.abstract || null,
      image_url: v.image_url || null
    });
    if (error) this.doiError = (error as { message?: string }).message || 'Error al guardar';
    else {
      this.editingPubId = null;
      this.publicationImagePreview = null;
      this.publications$ = this.publications.getPublications();
    }
  }

  async deletePublication(id: string): Promise<void> {
    if (!confirm('¿Eliminar esta publicación?')) return;
    const { error } = await this.publications.delete(id);
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al eliminar';
      return;
    }
    this.publications$ = this.publications.getPublications();
  }

  async submitApp(): Promise<void> {
    if (this.appForm.invalid || this.appSaving) return;
    this.appSaving = true;
    const { error } = await this.apps.create({
      title: this.appForm.getRawValue().title,
      description: this.appForm.getRawValue().description || undefined,
      image_url: this.appForm.getRawValue().image_url || undefined,
      web_url: this.appForm.getRawValue().web_url
    });
    this.appSaving = false;
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al crear app';
      return;
    }
    this.appForm.reset();
    this.appImagePreview = null;
    this.refreshApps();
  }

  async onAppImageFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    this.appImageUploading = true;
    this.doiError = '';
    const path = this.storage.uniquePath('apps', file);
    const { url, error } = await this.storage.uploadImage(path, file);
    this.appImageUploading = false;
    input.value = '';
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al subir la imagen';
      return;
    }
    if (url) {
      this.appForm.patchValue({ image_url: url });
      this.appImagePreview = url;
    }
  }

  clearAppImage(): void {
    this.appForm.patchValue({ image_url: '' });
    this.appImagePreview = null;
  }

  async deleteApp(id: string): Promise<void> {
    if (!confirm('¿Eliminar esta app?')) return;
    const { error } = await this.apps.delete(id);
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al eliminar';
      return;
    }
    this.refreshApps();
  }

  async onProjectAddImageFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    this.appImageUploading = true;
    this.doiError = '';
    const path = this.storage.uniquePath('projects', file);
    const { url, error } = await this.storage.uploadImage(path, file);
    this.appImageUploading = false;
    input.value = '';
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al subir la imagen';
      return;
    }
    if (url) {
      this.projectForm.patchValue({ image_url: url });
      this.projectAddImagePreview = url;
    }
  }

  clearProjectAddImage(): void {
    this.projectForm.patchValue({ image_url: '' });
    this.projectAddImagePreview = null;
  }

  async onProjectEditImageFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    this.appImageUploading = true;
    this.doiError = '';
    const path = this.storage.uniquePath('projects', file);
    const { url, error } = await this.storage.uploadImage(path, file);
    this.appImageUploading = false;
    input.value = '';
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al subir la imagen';
      return;
    }
    if (url) {
      this.editProjectForm.patchValue({ image_url: url });
      this.projectImagePreview = url;
    }
  }

  clearProjectEditImage(): void {
    this.editProjectForm.patchValue({ image_url: '' });
    this.projectImagePreview = null;
  }

  async submitProject(): Promise<void> {
    if (this.projectForm.invalid || this.projectSaving) return;
    this.projectSaving = true;
    const v = this.projectForm.getRawValue();
    const budgetNum = typeof v.budget === 'string' ? parseFloat(v.budget) : v.budget;
    const budget = budgetNum != null && !Number.isNaN(budgetNum) ? budgetNum : undefined;
    const { error } = await this.projects.create({
      title: v.title,
      description: v.description || undefined,
      budget: budget,
      team: v.team || undefined,
      image_url: v.image_url || undefined
    });
    this.projectSaving = false;
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al crear proyecto';
      return;
    }
    this.projectForm.reset({ title: '', description: '', budget: '', team: '', image_url: '' });
    this.projectAddImagePreview = null;
    this.refreshProjects();
  }

  startEditProject(proj: { id: string; title: string; description: string | null; budget: number | null; team: string | null; image_url: string | null }): void {
    this.editingProjId = proj.id;
    this.projectImagePreview = proj.image_url;
    this.editProjectForm.patchValue({
      title: proj.title,
      description: proj.description ?? '',
      budget: proj.budget != null ? String(proj.budget) : '',
      team: proj.team ?? '',
      image_url: proj.image_url ?? ''
    });
  }

  cancelEditProject(): void {
    this.editingProjId = null;
    this.projectImagePreview = null;
  }

  async saveEditProject(): Promise<void> {
    if (!this.editingProjId || this.editProjectForm.invalid) return;
    const v = this.editProjectForm.getRawValue();
    const budgetNum = typeof v.budget === 'string' ? parseFloat(v.budget) : v.budget;
    const budget = budgetNum != null && !Number.isNaN(budgetNum) ? budgetNum : undefined;
    const { error } = await this.projects.update(this.editingProjId, {
      title: v.title,
      description: v.description || undefined,
      budget,
      team: v.team || undefined,
      image_url: v.image_url || undefined
    });
    if (error) this.doiError = (error as { message?: string }).message || 'Error al guardar';
    else {
      this.editingProjId = null;
      this.projectImagePreview = null;
      this.refreshProjects();
    }
  }

  async deleteProject(id: string): Promise<void> {
    if (!confirm('¿Eliminar este proyecto?')) return;
    const { error } = await this.projects.delete(id);
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al eliminar';
      return;
    }
    this.refreshProjects();
  }

  startEditApp(app: { id: string; title: string; description: string | null; web_url: string; image_url: string | null }): void {
    this.editingAppId = app.id;
    this.editAppForm.patchValue({
      title: app.title,
      description: app.description ?? '',
      web_url: app.web_url,
      image_url: app.image_url ?? ''
    });
    this.appImagePreview = app.image_url;
  }

  cancelEditApp(): void {
    this.editingAppId = null;
    this.appImagePreview = null;
  }

  async saveEditApp(): Promise<void> {
    if (!this.editingAppId || this.editAppForm.invalid) return;
    const v = this.editAppForm.getRawValue();
    const { error } = await this.apps.update(this.editingAppId, {
      title: v.title,
      description: v.description || undefined,
      web_url: v.web_url,
      image_url: v.image_url || undefined
    });
    if (error) this.doiError = (error as { message?: string }).message || 'Error al guardar';
    else {
      this.editingAppId = null;
      this.appImagePreview = null;
      this.refreshApps();
    }
  }

  formationTypeLabel(type: FormationType): string {
    return this.formationTypes.find((t) => t.value === type)?.label ?? type;
  }

  async submitFormation(): Promise<void> {
    if (this.formationForm.invalid || this.formationSaving) return;
    this.formationSaving = true;
    const v = this.formationForm.getRawValue();
    const { error } = await this.formation.create({
      type: v.type,
      title: v.title || undefined,
      content: v.content,
      level: v.level || undefined
    });
    this.formationSaving = false;
    if (error) this.doiError = (error as { message?: string }).message || 'Error al crear formación';
    else {
      this.formationForm.reset({ type: 'job', title: '', content: '', level: '' });
      this.refreshFormation();
    }
  }

  startEditFormation(item: { id: string; type: FormationType; title: string | null; content: string; level: string | null }): void {
    this.editingFormationId = item.id;
    this.editFormationForm.patchValue({
      type: item.type,
      title: item.title ?? '',
      content: item.content,
      level: item.level ?? ''
    });
  }

  cancelEditFormation(): void {
    this.editingFormationId = null;
  }

  async saveEditFormation(): Promise<void> {
    if (!this.editingFormationId || this.editFormationForm.invalid) return;
    const v = this.editFormationForm.getRawValue();
    const { error } = await this.formation.update(this.editingFormationId, {
      type: v.type,
      title: v.title || undefined,
      content: v.content,
      level: v.level || undefined
    });
    if (error) this.doiError = (error as { message?: string }).message || 'Error al guardar';
    else {
      this.editingFormationId = null;
      this.refreshFormation();
    }
  }

  async deleteFormation(id: string): Promise<void> {
    if (!confirm('¿Eliminar este ítem de formación?')) return;
    const { error } = await this.formation.delete(id);
    if (error) this.doiError = (error as { message?: string }).message || 'Error al eliminar';
    else this.refreshFormation();
  }

  async submitGallery(): Promise<void> {
    if (this.galleryForm.invalid || this.gallerySaving) return;
    this.gallerySaving = true;
    const v = this.galleryForm.getRawValue();
    const { error } = await this.gallery.create({
      title: v.title || undefined,
      image_url: v.image_url,
      author: v.author || undefined,
      tags: v.tags || undefined
    });
    this.gallerySaving = false;
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al crear elemento de galería';
      return;
    }
    this.galleryForm.reset({ title: '', image_url: '', author: '', tags: '' });
    this.galleryImagePreview = null;
    this.refreshGallery();
  }

  async onGalleryImageFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    this.appImageUploading = true;
    this.doiError = '';
    const path = this.storage.uniquePath('gallery', file);
    const { url, error } = await this.storage.uploadImage(path, file);
    this.appImageUploading = false;
    input.value = '';
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al subir la imagen';
      return;
    }
    if (url) {
      this.galleryForm.patchValue({ image_url: url });
      this.galleryImagePreview = url;
    }
  }

  clearGalleryImage(): void {
    this.galleryForm.patchValue({ image_url: '' });
    this.galleryImagePreview = null;
  }

  async onGalleryEditImageFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    this.appImageUploading = true;
    this.doiError = '';
    const path = this.storage.uniquePath('gallery', file);
    const { url, error } = await this.storage.uploadImage(path, file);
    this.appImageUploading = false;
    input.value = '';
    if (error) {
      this.doiError = (error as { message?: string }).message || 'Error al subir la imagen';
      return;
    }
    if (url) {
      this.editGalleryForm.patchValue({ image_url: url });
      this.galleryEditImagePreview = url;
    }
  }

  clearGalleryEditImage(): void {
    this.editGalleryForm.patchValue({ image_url: '' });
    this.galleryEditImagePreview = null;
  }

  startEditGallery(item: { id: string; title: string | null; image_url: string; author: string | null; tags: string | null }): void {
    this.editingGalleryId = item.id;
    this.galleryEditImagePreview = item.image_url;
    this.editGalleryForm.patchValue({
      title: item.title ?? '',
      image_url: item.image_url,
      author: item.author ?? '',
      tags: item.tags ?? ''
    });
  }

  cancelEditGallery(): void {
    this.editingGalleryId = null;
    this.galleryEditImagePreview = null;
  }

  async saveEditGallery(): Promise<void> {
    if (!this.editingGalleryId || this.editGalleryForm.invalid) return;
    const v = this.editGalleryForm.getRawValue();
    const { error } = await this.gallery.update(this.editingGalleryId, {
      title: v.title || null,
      image_url: v.image_url,
      author: v.author || null,
      tags: v.tags || null
    });
    if (error) this.doiError = (error as { message?: string }).message || 'Error al guardar';
    else {
      this.editingGalleryId = null;
      this.galleryEditImagePreview = null;
      this.refreshGallery();
    }
  }

  async deleteGallery(id: string): Promise<void> {
    if (!confirm('¿Eliminar este elemento de la galería?')) return;
    const { error } = await this.gallery.delete(id);
    if (error) this.doiError = (error as { message?: string }).message || 'Error al eliminar';
    else this.refreshGallery();
  }
}
