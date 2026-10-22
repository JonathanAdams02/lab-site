// Loads everything in the /content folder (the part editors change through
// Pages CMS) and hands it to the templates as `lab`.
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { load as loadYaml } from "js-yaml";
import markdownIt from "markdown-it";

const md = markdownIt({ html: true, linkify: true, breaks: false });
const root = path.resolve("content");

// Read one markdown file: front matter fields + rendered body as `html`.
function readEntry(file) {
  const raw = fs.readFileSync(file, "utf8");
  const { data, content } = matter(raw);
  const body = content.trim();
  return { ...data, html: body ? md.render(body) : "", slug: path.basename(file, ".md") };
}

// Read every .md file in a content folder.
function readFolder(name) {
  const dir = path.join(root, name);
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .sort()
    .map((f) => readEntry(path.join(dir, f)));
}

// Sort by the `order` field (empty order goes last), then by a fallback field.
function byOrder(fallback) {
  return (a, b) => {
    const ao = a.order === undefined || a.order === null || a.order === "" ? Infinity : Number(a.order);
    const bo = b.order === undefined || b.order === null || b.order === "" ? Infinity : Number(b.order);
    if (ao !== bo) return ao - bo;
    return String(a[fallback] ?? "").localeCompare(String(b[fallback] ?? ""));
  };
}

function readYaml(file) {
  return fs.existsSync(file) ? loadYaml(fs.readFileSync(file, "utf8")) || {} : {};
}

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const normTitle = (s) => String(s || "").toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "");
const normDoi = (s) => {
  const m = String(s || "").match(/10\.\d{4,9}\/\S+/);
  return m ? m[0].toLowerCase().replace(/[.,;]+$/, "") : "";
};

// "A, B, C, … , Member, et al." — lab members in bold, and always shown.
function formatAuthors(authors, total, max) {
  const shown = authors.slice(0, max);
  const extraMembers = authors.slice(max).filter((a) => a.member);
  const fmt = (a) => (a.member ? `<strong>${escapeHtml(a.name)}</strong>` : escapeHtml(a.name));
  let html = shown.map(fmt).join(", ");
  if (extraMembers.length) html += ", … " + extraMembers.map(fmt).join(", ");
  if (total > shown.length + extraMembers.length) html += ", et al.";
  return html;
}

function buildPublications(settings, manualEntries) {
  const max = Number(settings.max_authors) || 8;
  const fromYear = Number(settings.from_year) || 0;
  const types = (settings.types || []).map(String);
  const excluded = (settings.exclude || [])
    .map((e) => (typeof e === "string" ? e : e?.id || e?.doi || ""))
    .map((e) => String(e).trim().toLowerCase())
    .filter(Boolean);
  const isExcluded = (p) =>
    excluded.some((e) => (p.doi && normDoi(e) === p.doi) || e.replace(/.*\//, "") === String(p.openalex || "").toLowerCase());

  let auto = [];
  try {
    auto = JSON.parse(fs.readFileSync(path.resolve("data", "publications.json"), "utf8")).works || [];
  } catch {}
  auto = auto
    .filter((p) => !types.length || types.includes(p.type))
    .filter((p) => !fromYear || Number(p.year) >= fromYear)
    .map((p) => ({
      ...p,
      link: p.url,
      authorsHtml: formatAuthors(p.authors || [], p.author_count || 0, max),
    }));

  const manual = manualEntries
    .filter((p) => p.title)
    .map((p) => ({
      ...p,
      doi: normDoi(p.link),
      date: String(p.year ?? ""),
      authorsHtml: escapeHtml(p.authors || ""),
      manual: true,
    }));

  // Hand-made entries win over automatic ones for the same paper.
  const seen = new Set();
  const all = [];
  for (const p of [...manual.sort(byOrder("title")), ...auto]) {
    const keys = [p.doi, `${normTitle(p.title)}-${p.year}`].filter(Boolean);
    if (keys.some((k) => seen.has(k)) || isExcluded(p)) continue;
    keys.forEach((k) => seen.add(k));
    all.push(p);
  }
  return all.sort((a, b) => String(b.date || b.year).localeCompare(String(a.date || a.year)));
}

// Projects: those fetched from FRIS (data/projects.json, refreshed weekly)
// plus any added by hand, filtered by the Project list settings.
function buildProjects(settings, manualEntries) {
  const today = new Date().toISOString().slice(0, 10);
  const completedYears = settings.show_completed_years === undefined || settings.show_completed_years === ""
    ? 2 : Number(settings.show_completed_years) || 0;
  const cutoff = `${new Date().getFullYear() - completedYears}-01-01`;
  const hideInternal = settings.hide_internal !== false;
  const internalWords = (settings.internal_keywords || ["own budget", "eigen budget", "own funds"])
    .map((w) => String(w).toLowerCase()).filter(Boolean);
  const leadRe = /promot|principal|coordinat|supervis|spokes|lead|hoofd/i;
  const maxWords = Number(settings.max_description_words) || 70;
  const excluded = (settings.exclude || [])
    .map((e) => String(typeof e === "string" ? e : e?.id || "").trim().toLowerCase())
    .filter(Boolean);

  let fris = [];
  try {
    fris = JSON.parse(fs.readFileSync(path.resolve("data", "projects.json"), "utf8")).projects || [];
  } catch {}

  const year = (d) => (d ? String(d).slice(0, 4) : "");
  fris = fris
    .filter((p) => !excluded.some((e) => e === String(p.uuid).toLowerCase() || e === p.title.toLowerCase() || (p.url && p.url.toLowerCase().endsWith(e))))
    .filter((p) => {
      if (!hideInternal) return true;
      const codes = (p.funding || []).map((f) => `${f.code} ${f.funders.join(" ")}`.toLowerCase());
      return !codes.length || !codes.every((c) => internalWords.some((w) => c.includes(w)));
    })
    .filter((p) => !p.end || p.end >= today || (completedYears > 0 && p.end >= cutoff))
    .map((p) => {
      const words = (p.abstract || "").split(/\s+/).filter(Boolean);
      const short = words.length > maxWords ? words.slice(0, maxWords).join(" ") + " …" : words.join(" ");
      const people = p.participants || [];
      const main = people.filter((x) => leadRe.test(x.role) && !/\bco-?/i.test(x.role));
      const leads = (main.length ? main : people.filter((x) => leadRe.test(x.role))).map((x) => x.name);
      // "Own budget, for example: patrimony…" -> "Own budget"; "(including WEAVE projects)" dropped.
      const tidy = (t) => String(t).split(/,\s*for example/i)[0].replace(/\s*\((incl\.?|including)[^)]*\)/i, "").trim();
      const funders = (p.funding || []).map((f) => tidy(f.code || f.funders.join(", "))).filter(Boolean);
      return {
        title: String(p.title).replace(/^["“'\s]+|["”'\s]+$/g, ""),
        status: p.start && p.start > today ? "Upcoming" : p.end && p.end < today ? "Completed" : "Ongoing",
        period: p.end ? `${year(p.start)} – ${year(p.end)}` : p.start ? `Since ${year(p.start)}` : "",
        html: short ? `<p>${escapeHtml(short)}</p>` : "",
        lead: [...new Set(leads)].join(", "),
        funder: [...new Set(funders)].join(", "),
        // Older imports built links from FRIS's internal number, which the
        // KU Leuven portal doesn't recognise (…/project/59074371) — skip those.
        url: /research\.kuleuven\.be\/portal\/en\/project\/\d+$/.test(p.url || "") ? "" : p.url,
        start: p.start || "",
      };
    });

  const manual = manualEntries.filter((p) => p.title).sort(byOrder("title"));
  return [...manual, ...fris];
}

function isTrue(value, fallback) {
  if (value === undefined || value === null || value === "") return fallback;
  return value === true || value === "true";
}

export default function () {
  const settings = loadYaml(fs.readFileSync(path.join(root, "settings.yml"), "utf8")) || {};

  const pages = {};
  for (const entry of readFolder("pages")) pages[entry.slug] = entry;

  // People, grouped in the order the groups are listed on the People page.
  const people = readFolder("people").filter((p) => p.name);
  const groupsDefined = pages.people?.groups || [];
  const peopleGroups = groupsDefined
    .map((g) => ({
      id: g.id,
      title: g.title,
      members: people.filter((p) => p.group === g.id).sort(byOrder("name")),
    }))
    .filter((g) => g.members.length);
  // Anyone whose group isn't listed still shows up, under "Other".
  const known = new Set(groupsDefined.map((g) => g.id));
  const orphans = people.filter((p) => !known.has(p.group)).sort(byOrder("name"));
  if (orphans.length) peopleGroups.push({ id: "other", title: "Other", members: orphans });

  // Publications: the automatic list from OpenAlex (data/publications.json,
  // refreshed weekly) plus any entries added by hand, grouped by year.
  const pubSettings = readYaml(path.join(root, "publications.yml"));
  const pubs = buildPublications(pubSettings, readFolder("publications"));
  const years = [...new Set(pubs.map((p) => String(p.year ?? "")))].sort((a, b) => b.localeCompare(a));
  const pubYears = years.map((year) => ({
    year: year || "Undated",
    items: pubs.filter((p) => String(p.year ?? "") === year),
  }));

  const projectSettings = readYaml(path.join(root, "projects-settings.yml"));
  const projects = buildProjects(projectSettings, readFolder("projects"));
  let projectsUpdated = null;
  try {
    const u = JSON.parse(fs.readFileSync(path.resolve("data", "projects.json"), "utf8")).updated;
    if (u) projectsUpdated = new Date(u).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  } catch {}
  const studies = readFolder("studies")
    .filter((s) => s.title && isTrue(s.recruiting, true))
    .sort(byOrder("title"));

  let pubsUpdated = null;
  try {
    const u = JSON.parse(fs.readFileSync(path.resolve("data", "publications.json"), "utf8")).updated;
    if (u) pubsUpdated = new Date(u).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  } catch {}

  return {
    settings,
    pages,
    peopleGroups,
    pubYears,
    pubsUpdated,
    projects,
    projectSettings,
    projectsUpdated,
    studies,
    isTrue,
    year: new Date().getFullYear(),
  };
}
