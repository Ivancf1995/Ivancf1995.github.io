export interface GalleryItem {
  id: string;
  title: string | null;
  image_url: string;
  author: string | null;
  tags: string | null;
  order: number;
  created_at: string;
}
