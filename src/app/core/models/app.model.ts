export interface AppItem {
  id: string;
  title: string;
  description: string | null;
  image_url: string | null;
  web_url: string;
  order: number;
  created_at: string;
}
