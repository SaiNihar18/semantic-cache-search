export type Group = "sci" | "comp" | "rec" | "talk" | "other";

export const GROUP_META: Record<Group, { label: string; dot: string }> = {
  sci: { label: "Science", dot: "bg-emerald-400" },
  comp: { label: "Computing", dot: "bg-violet-400" },
  rec: { label: "Recreation", dot: "bg-amber-400" },
  talk: { label: "Talk / Politics", dot: "bg-rose-400" },
  other: { label: "Miscellaneous", dot: "bg-slate-400" },
};

export function groupOf(category: string): Group {
  if (category.startsWith("sci.")) return "sci";
  if (category.startsWith("comp.")) return "comp";
  if (category.startsWith("rec.")) return "rec";
  if (category.startsWith("talk.")) return "talk";
  return "other";
}

export const NEWSGROUPS: { group: Group; items: { name: string; desc: string }[] }[] = [
  {
    group: "sci",
    items: [
      { name: "sci.space", desc: "Space exploration, astronomy, NASA" },
      { name: "sci.med", desc: "Medicine, health, diseases" },
      { name: "sci.electronics", desc: "Electronics and circuits" },
      { name: "sci.crypt", desc: "Cryptography and security" },
    ],
  },
  {
    group: "comp",
    items: [
      { name: "comp.graphics", desc: "Computer graphics and visualization" },
      { name: "comp.os.ms-windows.misc", desc: "Windows OS discussions" },
      { name: "comp.sys.ibm.pc.hardware", desc: "PC hardware and peripherals" },
      { name: "comp.sys.mac.hardware", desc: "Apple Mac hardware" },
      { name: "comp.windows.x", desc: "X Window System" },
    ],
  },
  {
    group: "rec",
    items: [
      { name: "rec.autos", desc: "Automobiles and motorsport" },
      { name: "rec.motorcycles", desc: "Motorcycles and biking" },
      { name: "rec.sport.baseball", desc: "Baseball and MLB" },
      { name: "rec.sport.hockey", desc: "Hockey and NHL" },
    ],
  },
  {
    group: "talk",
    items: [
      { name: "talk.politics.misc", desc: "General political discussion" },
      { name: "talk.politics.guns", desc: "Gun rights and control" },
      { name: "talk.politics.mideast", desc: "Middle East politics" },
      { name: "talk.religion.misc", desc: "Religion and belief" },
    ],
  },
  {
    group: "other",
    items: [
      { name: "alt.atheism", desc: "Atheism and secular humanism" },
      { name: "soc.religion.christian", desc: "Christianity and church" },
      { name: "misc.forsale", desc: "Buy and sell postings" },
    ],
  },
];
