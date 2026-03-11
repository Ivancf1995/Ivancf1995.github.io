export type FormationType = 'job' | 'study' | 'course' | 'language' | 'programming';

export interface FormationItem {
  id: string;
  type: FormationType;
  title: string | null;
  content: string;
  level: string | null;
  order: number;
  created_at: string;
}
