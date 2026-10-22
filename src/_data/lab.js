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

  // Publications grouped by year, newest year first.
  const pubs = readFolder("publications").filter((p) => p.title);
  const years = [...new Set(pubs.map((p) => String(p.year ?? "")))].sort((a, b) => b.localeCompare(a));
  const pubYears = years.map((year) => ({
    year: year || "Undated",
    items: pubs.filter((p) => String(p.year ?? "") === year).sort(byOrder("title")),
  }));

  const projects = readFolder("projects").filter((p) => p.title).sort(byOrder("title"));
  const studies = readFolder("studies")
    .filter((s) => s.title && isTrue(s.recruiting, true))
    .sort(byOrder("title"));

  return {
    settings,
    pages,
    peopleGroups,
    pubYears,
    projects,
    studies,
    isTrue,
    year: new Date().getFullYear(),
  };
}
