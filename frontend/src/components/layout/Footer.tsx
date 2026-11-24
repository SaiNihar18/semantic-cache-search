import { API_DOCS, REPO_URL } from "@/lib/api";

const DATASET_URL =
  "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset";

export function Footer() {
  const year = new Date().getFullYear();
  const link = "transition-colors hover:text-foreground";
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-[1040px] flex-wrap items-center justify-between gap-4 px-5 py-6 text-xs text-muted-foreground">
        <p>
          &copy; {year} Built by{" "}
          <a
            href="https://github.com/SaiNihar18"
            target="_blank"
            rel="noopener noreferrer"
            className="text-foreground/80 hover:text-brand"
          >
            Sai Nihar
          </a>
          . 20 Newsgroups, FAISS and SentenceTransformers.
        </p>
        <nav aria-label="Footer" className="flex items-center gap-5">
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className={link}>
            GitHub
          </a>
          <a href={API_DOCS} target="_blank" rel="noopener noreferrer" className={link}>
            API Docs
          </a>
          <a href={DATASET_URL} target="_blank" rel="noopener noreferrer" className={link}>
            Dataset
          </a>
        </nav>
      </div>
    </footer>
  );
}
