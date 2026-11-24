import { Link } from "@tanstack/react-router";
import { ExternalLink, Github, Menu, Zap } from "lucide-react";
import { API_DOCS, REPO_URL } from "@/lib/api";
import { useBackendStatus, type BackendStatus } from "@/lib/hooks";
import { Sheet, SheetClose, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";

const LINKS = [
  { to: "/", label: "Explorer" },
  { to: "/metrics", label: "Metrics" },
  { to: "/about", label: "About Data" },
] as const;

const STATUS: Record<BackendStatus, { label: string; dot: string }> = {
  online: { label: "API online", dot: "bg-brand" },
  connecting: { label: "Connecting", dot: "bg-warning animate-pulse" },
  offline: { label: "API offline", dot: "bg-destructive" },
};

export function StatusPill() {
  const status = useBackendStatus();
  const s = STATUS[status];
  return (
    <span
      role="status"
      className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-2.5 py-1 text-xs text-muted-foreground"
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}

const linkBase =
  "relative py-4 text-sm text-muted-foreground transition-colors hover:text-foreground";

export function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1040px] items-center justify-between px-5">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid h-7 w-7 place-items-center rounded-md bg-brand text-brand-foreground">
              <Zap className="h-4 w-4" aria-hidden />
            </span>
            SemanticCache
          </Link>
          <nav aria-label="Primary" className="hidden items-center gap-6 md:flex">
            {LINKS.map((l) => (
              <Link
                key={l.to}
                to={l.to}
                activeOptions={{ exact: true }}
                className={linkBase}
                activeProps={{
                  className:
                    "text-foreground after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:bg-brand",
                }}
              >
                {l.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:block">
            <StatusPill />
          </div>
          <a
            href={API_DOCS}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground md:inline-flex"
          >
            API Docs <ExternalLink className="h-3 w-3" aria-hidden />
          </a>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub repository"
            className="grid h-8 w-8 place-items-center rounded-md border border-border text-muted-foreground transition-colors hover:text-foreground"
          >
            <Github className="h-4 w-4" />
          </a>

          <Sheet>
            <SheetTrigger
              aria-label="Open menu"
              className="grid h-8 w-8 place-items-center rounded-md border border-border text-muted-foreground md:hidden"
            >
              <Menu className="h-4 w-4" />
            </SheetTrigger>
            <SheetContent side="right" className="w-64 bg-background">
              <SheetTitle className="mb-6 text-base">Menu</SheetTitle>
              <nav className="flex flex-col gap-1" aria-label="Mobile">
                {LINKS.map((l) => (
                  <SheetClose asChild key={l.to}>
                    <Link
                      to={l.to}
                      activeOptions={{ exact: true }}
                      className="rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
                      activeProps={{ className: "bg-accent text-foreground" }}
                    >
                      {l.label}
                    </Link>
                  </SheetClose>
                ))}
                <a
                  href={API_DOCS}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
                >
                  API Docs
                </a>
              </nav>
              <div className="mt-6 px-3">
                <StatusPill />
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}
