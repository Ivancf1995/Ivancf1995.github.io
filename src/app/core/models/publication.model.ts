export interface Publication {
  id: string;
  doi: string;
  title: string;
  authors: string | null;
  year: number | null;
  journal: string | null;
  url: string | null;
  abstract: string | null;
  image_url: string | null;
  created_at: string;
}
