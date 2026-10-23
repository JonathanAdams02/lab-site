// Fetches the group's publications from OpenAlex (https://openalex.org) and
// saves them to data/publications.json. Runs weekly on GitHub (see
// .github/workflows/update-publications.yml) and can be run by hand with:
//   node scripts/fetch-publications.mjs
//
// Who counts as "the group": everyone in content/people with an ORCID iD
// and/or an OpenAlex author ID filled in.
// Optional environment variables: OPENALEX_API_KEY, OPENALEX_MAILTO,
// OPENALEX_BASE (only used for testing).

import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { load as loadYaml } from "js-yaml";

const ROOT = process.cwd();
const OUT = path.join(ROOT, "data", "publications.json");
const API = (process.env.OPENALEX_BASE || "https://api.openalex.org").replace(/\/$/, "");
const MAX_STORED_AUTHORS = 40;

// ---------- read settings and people ----------
const settingsFile = path.join(ROOT, "content", "publications.yml");
const settings = fs.existsSync(settingsFile) ? loadYaml(fs.readFileSync(settingsFile, "utf8")) || {} : {};
const fromYear = Number(settings.from_year) || 2018;
const requireAffiliation = settings.require_affiliation !== false;
const affiliationWords = (settings.affiliation_keywords || ["Leuven"])
  .map((w) => String(w).trim().toLowerCase())
  .filter(Boolean);

const peopleDir = path.join(ROOT, "content", "people");
const people = fs.existsSync(peopleDir)
  ? fs.readdirSync(peopleDir).filter((f) => f.endsWith(".md"))
      .map((f) => matter(fs.readFileSync(path.join(peopleDir, f), "utf8")).data)
  : [];

const ORCID_RE = /(\d{4}-\d{4}-\d{4}-\d{3}[\dX])/i;
const OAID_RE = /(A\d{4,})/i;
const orcids = new Set();
const authorIds = new Set();
for (const p of people) {
  const o = String(p.orcid || "").match(ORCID_RE);
  if (o) orcids.add(o[1].toUpperCase());
  const a = String(p.openalex_id || "").match(OAID_RE);
  if (a) authorIds.add(a[1].toUpperCase());
}

if (!orcids.size && !authorIds.size) {
  console.log("No ORCID iDs or OpenAlex IDs found in content/people — nothing to fetch.");
  process.exit(0);
}
console.log(`Fetching works for ${orcids.size} ORCID iD(s) and ${authorIds.size} OpenAlex ID(s), from ${fromYear}…`);

// ---------- OpenAlex helpers ----------
const SELECT = [
  "id", "doi", "display_name", "publication_year", "publication_date", "type",
  "primary_location", "authorships", "is_retracted", "open_access",
].join(",");

async function getJson(url, attempt = 1) {
  const res = await fetch(url, { headers: { "User-Agent": "lab-website (publication list)" } });
  if (res.status === 429 || res.status >= 500) {
    if (attempt >= 5) throw new Error(`OpenAlex ${res.status} after ${attempt} tries: ${url}`);
    await new Promise((r) => setTimeout(r, 2000 * attempt));
    return getJson(url, attempt + 1);
  }
  if (!res.ok) throw new Error(`OpenAlex ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

// With an API key we use OpenAlex's "cursor" paging (no limit on results).
// Without one, cursor paging is refused, so we fall back to plain page
// numbers, which OpenAlex allows without a key (up to 10,000 results).
async function fetchAll(filter) {
  const apiKey = process.env.OPENALEX_API_KEY;
  const base = () => {
    const params = new URLSearchParams({ filter, select: SELECT, "per-page": "200" });
    if (process.env.OPENALEX_MAILTO) params.set("mailto", process.env.OPENALEX_MAILTO);
    if (apiKey) params.set("api_key", apiKey);
    return params;
  };
  const works = [];
  if (apiKey) {
    let cursor = "*";
    while (cursor) {
      const params = base();
      params.set("cursor", cursor);
      const data = await getJson(`${API}/works?${params}`);
      works.push(...(data.results || []));
      cursor = data.meta?.next_cursor || null;
      if (!data.results?.length) break;
    }
  } else {
    for (let page = 1; page <= 50; page++) {
      const params = base();
      params.set("page", String(page));
      const data = await getJson(`${API}/works?${params}`);
      const results = data.results || [];
      works.push(...results);
      const total = Number(data.meta?.count ?? 0);
      if (results.length < 200 || works.length >= total) break;
    }
  }
  return works;
}

if (!process.env.OPENALEX_API_KEY) {
  console.log("Note: no OPENALEX_API_KEY set — using page-number paging. Add a free key (openalex.org/settings/api) as a repository secret for best reliability.");
}

// OpenAlex allows up to 100 OR-values per filter; stay well below.
function chunks(arr, n = 50) {
  const out = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
}

const yearFilter = `publication_year:>${fromYear - 1}`;
const raw = [];
for (const group of chunks([...orcids])) {
  raw.push(...(await fetchAll(`author.orcid:${group.join("|")},${yearFilter}`)));
}
for (const group of chunks([...authorIds])) {
  raw.push(...(await fetchAll(`author.id:${group.join("|")},${yearFilter}`)));
}
console.log(`OpenAlex returned ${raw.length} records.`);

// ---------- filter, tidy and de-duplicate ----------
const shortId = (url) => String(url || "").split("/").pop().toUpperCase();
const orcidOf = (url) => (String(url || "").match(ORCID_RE) || [])[1]?.toUpperCase() || "";
const stripTags = (s) => String(s || "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
const normTitle = (s) => stripTags(s).toLowerCase().replace(/[^a-z0-9]+/g, "");

function isMember(authorship) {
  const a = authorship.author || {};
  return authorIds.has(shortId(a.id)) || Boolean(a.orcid && orcids.has(orcidOf(a.orcid)));
}

function hasAffiliation(authorship) {
  const names = [
    ...(authorship.institutions || []).map((i) => i.display_name || ""),
    ...(authorship.raw_affiliation_strings || []),
  ].join(" | ").toLowerCase();
  return affiliationWords.some((w) => names.includes(w));
}

const byKey = new Map();
for (const w of raw) {
  if (!w || !w.display_name || w.is_retracted) continue;
  const members = (w.authorships || []).filter(isMember);
  if (!members.length) continue;
  if (requireAffiliation && !members.some(hasAffiliation)) continue;

  const authorships = w.authorships || [];
  const stored = authorships
    .map((a, i) => ({ name: a.author?.display_name || "", member: isMember(a), i, au: a.author || {} }))
    .filter((a) => a.name && (a.i < MAX_STORED_AUTHORS || a.member))
    // For lab members, also keep their IDs so each profile page can list their papers.
    .map(({ name, member, au }) => (member
      ? { name, member, orcid: orcidOf(au.orcid), openalex: shortId(au.id) }
      : { name, member }));

  const doi = w.doi ? String(w.doi).replace(/^https?:\/\/(dx\.)?doi\.org\//i, "").toLowerCase() : "";
  const item = {
    openalex: shortId(w.id),
    doi,
    title: stripTags(w.display_name),
    year: w.publication_year,
    date: w.publication_date || String(w.publication_year || ""),
    type: w.type || "",
    journal: w.primary_location?.source?.display_name || "",
    url: doi ? `https://doi.org/${doi}` : w.primary_location?.landing_page_url || w.id,
    open_access_url: w.open_access?.is_oa ? w.open_access.oa_url || "" : "",
    authors: stored,
    author_count: authorships.length,
  };

  // Same paper can appear twice (e.g. a journal version and a repository
  // copy). Keep one per DOI / title, preferring the one with a DOI and journal.
  const key = doi || `${normTitle(item.title)}-${item.year}`;
  const titleKey = `${normTitle(item.title)}-${item.year}`;
  const existing = byKey.get(key) || byKey.get(titleKey);
  const score = (x) => (x.doi ? 2 : 0) + (x.journal ? 1 : 0) + (x.type === "preprint" ? -3 : 0);
  if (!existing || score(item) > score(existing)) {
    if (existing) { byKey.delete(existing.doi || ""); byKey.delete(`${normTitle(existing.title)}-${existing.year}`); }
    if (doi) byKey.set(doi, item);
    byKey.set(titleKey, item);
  }
}

const works = [...new Set(byKey.values())].sort(
  (a, b) => String(b.date).localeCompare(String(a.date)) || a.title.localeCompare(b.title)
);

// ---------- write (keep old timestamp if nothing changed, to avoid empty commits) ----------
let previous = null;
try { previous = JSON.parse(fs.readFileSync(OUT, "utf8")); } catch {}
const unchanged = previous && JSON.stringify(previous.works) === JSON.stringify(works);
const output = {
  source: "OpenAlex (https://openalex.org)",
  updated: unchanged ? previous.updated : new Date().toISOString(),
  works,
};
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(output, null, 2) + "\n");
console.log(`${works.length} publications saved to data/publications.json${unchanged ? " (no changes)" : ""}.`);
