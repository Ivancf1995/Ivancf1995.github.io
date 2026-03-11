export interface Project {
  id: string;
  title: string;
  description: string | null;
  status: string | null;
  url: string | null;
  budget: number | null;
  team: string | null;
  image_url: string | null;
  order: number;
  created_at: string;
}
