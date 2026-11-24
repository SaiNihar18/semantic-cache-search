import { useSyncExternalStore } from "react";

export type CacheOp = {
  id: number;
  time: Date;
  query: string;
  action: "HIT" | "MISS";
  durationMs: number;
};

let ops: CacheOp[] = [];
let nextId = 1;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function recordOp(op: Omit<CacheOp, "id" | "time">) {
  ops = [{ ...op, id: nextId++, time: new Date() }, ...ops].slice(0, 10);
  emit();
}

export function clearOps() {
  ops = [];
  emit();
}

export function useOps() {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => ops,
    () => ops,
  );
}
